import argparse
import csv
import json
import shutil
import time
from pointdiff.runtime import (
    atomic_checkpoint,
    atomic_json,
    configure_runtime,
    make_dataset,
    model_inputs,
    project_path,
    read_config,
    restore_rng,
    rng_state,
    source_fingerprints,
    implementation_info,
)
import torch
from torch.utils.data import DataLoader
from pointdiff.modeling import build_model, model_config
from pointdiff.data import pointdiff_collate
from pointdiff.evaluation import validate_during_training
from pointdiff.splits import verify_validation_split
from pointdiff.validation import (
    RECALL_METRICS,
    evaluation_settings,
    format_metrics,
    selection_record,
)


def format_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return "%02d:%02d:%02d" % (hours, minutes, seconds)


def estimate_remaining_seconds(durations, remaining_epochs, partial_seconds=0.0):
    if remaining_epochs <= 0:
        return 0.0
    if not durations:
        return None
    average = sum(durations) / len(durations)
    return max(0.0, average * remaining_epochs - partial_seconds)


def make_scheduler(optimizer, training):
    config = training.get("scheduler", {"type": "none"})
    if config["type"] == "none":
        return None
    if config["type"] == "step":
        if config["step_size"] < 1 or not 0 < config["gamma"] < 1:
            raise ValueError("StepLR requires step_size>=1 and 0<gamma<1")
        return torch.optim.lr_scheduler.StepLR(optimizer, config["step_size"], config["gamma"])
    if config["type"] == "multistep":
        milestones = config.get("milestones", [])
        if (
            not milestones
            or any((not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in milestones))
            or sorted(set(milestones)) != milestones
            or (not 0 < config["gamma"] < 1)
        ):
            raise ValueError(
                "MultiStepLR requires increasing positive integer milestones and 0<gamma<1"
            )
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, config["gamma"])
    raise ValueError("Unknown scheduler: " + config["type"])


def gpc_statistics(result):
    if "gpc_diagnostics" not in result:
        return {}
    diag = result["gpc_diagnostics"]
    result = {
        "point_coverage_percent": 100.0 * float(diag["point_contained"].float().mean()),
        "mean_width": float(diag["width"].mean()),
        "single_frame_percent": 100.0 * float(diag["single_frame"].float().mean()),
        "point_below_threshold_percent": 100.0
        * float(diag["point_below_weight_threshold"].float().mean()),
        "point_clamped_percent": 100.0 * float(diag["point_clamped"].float().mean()),
    }
    return result


def condition_statistics(result):
    if "condition_diagnostics" not in result:
        return {}
    diag = result["condition_diagnostics"]
    return {
        "max_peak_inside_pseudo_percent": 100.0
        * float(diag["max_peak_inside_pseudo"].float().mean()),
        "min_peak_inside_pseudo_percent": 100.0
        * float(diag["min_peak_inside_pseudo"].float().mean()),
        "point_cosine_percentile": 100.0 * float(diag["point_cosine_percentile"].mean()),
        "mean_point_cosine": float(diag["point_cosine"].mean()),
        "mean_inside_outside_cosine_margin": float(diag["inside_outside_cosine_margin"].mean()),
        "cosine_margin_valid_percent": 100.0 * float(diag["cosine_margin_valid"].float().mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/charades_i3d.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = read_config(args.config)
    train_config = config["training"]
    for arg, key in [("epochs", "epochs"), ("batch_size", "batch_size")]:
        if getattr(args, arg) is not None:
            train_config[key] = getattr(args, arg)
    if train_config["epochs"] < 1 or train_config["batch_size"] < 1:
        parser.error("epochs and batch size must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max steps must be positive")
    seed = config.get("seed", 25285)
    device = configure_runtime(seed, args.threads, args.device)
    output = project_path(args.output or config["output"])
    if not args.resume and (output / "last.pt").exists():
        raise FileExistsError(
            "Output already contains a checkpoint. Use --resume or a new --output: %s" % output
        )
    output.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset(config, True, args.limit_train)
    if not len(dataset):
        raise RuntimeError("No training queries are available")
    sample = dataset[0]
    if sample["video"].shape[-1] != config["model"]["video_dim"]:
        raise ValueError("Video feature dimension does not match model.video_dim")
    val_config = config.get("validation", {})
    validation_enabled = val_config.get("enabled", False)
    settings = evaluation_settings(config)
    (validation_loader, split_info) = (None, None)
    if args.limit_validation and (not validation_enabled):
        parser.error("--limit-validation requires enabled validation")
    if validation_enabled:
        if val_config.get("every", 1) < 1 or val_config.get("batch_size", 48) < 1:
            parser.error("Validation interval and batch size must be positive")
        validation_data = make_dataset(config, False, args.limit_validation, split="validation")
        split_info = verify_validation_split(config, dataset, validation_data)
        validation_loader = DataLoader(
            validation_data,
            batch_size=val_config.get("batch_size", 48),
            shuffle=False,
            num_workers=args.workers,
            collate_fn=pointdiff_collate,
            pin_memory=device.type == "cuda",
        )
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        betas=tuple(train_config.get("betas", [0.9, 0.99])),
        weight_decay=train_config.get("weight_decay", 1e-05),
    )
    scheduler = make_scheduler(optimizer, train_config)
    generator = torch.Generator()
    (start_epoch, resume_step, global_step) = (1, 0, 0)
    (best_validation, last_validation, epoch_accumulators) = (None, None, None)
    if args.resume:
        checkpoint = torch.load(project_path(args.resume), map_location="cpu")

        def semantic_config(value):
            value = json.loads(json.dumps(value))
            value["model"] = model_config(value["model"])
            value["model"].pop("gpc_device", None)
            for key in [
                "cache_video",
                "video_cache_capacity",
                "cache_text",
                "text_cache_capacity",
                "gpc_video_cache_capacity",
            ]:
                value["data"].pop(key, None)
            return value

        (original, current) = (semantic_config(checkpoint["config"]), semantic_config(config))
        if original["model"] != current["model"] or original["data"] != current["data"]:
            raise ValueError("Resume requires the original model and data configuration")
        if (
            checkpoint.get("train_limit") != args.limit_train
            or checkpoint["config"]["training"]["batch_size"] != train_config["batch_size"]
        ):
            raise ValueError("Resume requires the original training limit and batch size")
        if (
            checkpoint.get("validation_limit") != args.limit_validation
            or checkpoint["config"].get("validation", {}) != val_config
        ):
            raise ValueError("Resume requires the original validation configuration and limit")
        if evaluation_settings(checkpoint["config"]) != settings:
            raise ValueError("Resume requires the original evaluation settings")
        for key in ["lr", "betas", "weight_decay", "grad_clip"]:
            if checkpoint["config"]["training"].get(key) != train_config.get(key):
                raise ValueError("Resume requires the original optimizer setting: " + key)
        if checkpoint["config"]["training"].get("scheduler", {"type": "none"}) != train_config.get(
            "scheduler", {"type": "none"}
        ):
            raise ValueError("Resume requires the original learning rate schedule")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_epoch = checkpoint["next_epoch"]
        resume_step = checkpoint.get("next_batch", 0)
        global_step = checkpoint["global_step"]
        best_validation = checkpoint.get("best_validation")
        last_validation = checkpoint.get("last_validation")
        epoch_accumulators = checkpoint.get("epoch_accumulators")
        if validation_enabled and best_validation:
            previous_best = output / "best.pt"
            if not previous_best.exists():
                previous_best = project_path(args.resume).resolve().parent / "best.pt"
            if not previous_best.exists():
                raise FileNotFoundError(
                    "Cannot preserve the previous best checkpoint: " + str(previous_best)
                )
            saved_best = torch.load(previous_best, map_location="cpu").get("best_validation", {})
            if any((saved_best.get(k) != best_validation[k] for k in ("epoch", "metric", "value"))):
                raise ValueError(
                    "best.pt does not match resume checkpoint; resume the latest coherent run"
                )
            if previous_best != output / "best.pt":
                shutil.copy2(previous_best, output / "best.pt")
        restore_rng(checkpoint["rng"])
    metadata = {
        "method": "PointDiff",
        "config": config,
        "device": str(device),
        "seed": seed,
        "train_queries": len(dataset),
        "train_limit": args.limit_train,
        "validation_limit": args.limit_validation,
        "validation_split": split_info,
        "max_steps": args.max_steps,
        "threads": args.threads,
        "torch_version": torch.__version__,
        "source_sha256": source_fingerprints(),
        "implementation": implementation_info(model),
        "protocol": "Training point labels only; no ground-truth interval passed to the model.",
        "run_type": (
            "verification"
            if args.limit_train
            or args.limit_validation
            or args.max_steps
            or (train_config["epochs"] != 100)
            else "full_100_epoch"
        ),
    }
    atomic_json(output / "config.json", config)
    atomic_json(output / "run_metadata.json", metadata)
    log = (output / "train_log.jsonl").open("a" if args.resume else "w")
    epoch_log = (output / "epoch_log.jsonl").open("a" if args.resume else "w")
    validation_log = (
        (output / "validation_log.jsonl").open("a" if args.resume else "w")
        if validation_enabled
        else None
    )
    csv_path = output / "validation_metrics.csv"
    csv_has_rows = args.resume and csv_path.exists() and (csv_path.stat().st_size > 0)
    csv_handle = (
        csv_path.open("a" if args.resume else "w", newline="") if validation_enabled else None
    )
    csv_fields = [
        "epoch",
        "R@1_IoU=0.5",
        "R@1_IoU=0.7",
        "R@5_IoU=0.5",
        "R@5_IoU=0.7",
        "mIoU",
        "mean_recall",
        "is_best",
        "sampling_steps",
        "postprocessing",
    ]
    candidate_fields = [
        "oracle_R@all_IoU=0.5",
        "oracle_R@all_IoU=0.7",
        "top1_is_best_candidate_percent",
        "candidate_pair_mean_iou",
    ]
    if val_config.get("candidate_diagnostics", False):
        csv_fields += candidate_fields
    csv_writer = csv.DictWriter(csv_handle, fieldnames=csv_fields) if csv_handle else None
    if csv_writer and (not csv_has_rows):
        csv_writer.writeheader()
        csv_handle.flush()
    prior_elapsed = 0.0
    if args.resume:
        summary_path = output / "training_summary.json"
        if not summary_path.exists():
            summary_path = project_path(args.resume).parent / "training_summary.json"
        if summary_path.exists():
            prior_elapsed = json.loads(summary_path.read_text()).get("elapsed_seconds", 0.0)
    epoch_seconds, completed_epoch_seconds = {}, {}
    if args.resume:
        history_path = output / "epoch_log.jsonl"
        if not history_path.exists() or history_path.stat().st_size == 0:
            history_path = project_path(args.resume).parent / "epoch_log.jsonl"
        if history_path.exists():
            with history_path.open() as history:
                for line in history:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record["global_step"] > global_step:
                        continue
                    previous_epoch = record["epoch"]
                    epoch_seconds[previous_epoch] = (
                        epoch_seconds.get(previous_epoch, 0.0) + record["seconds"]
                    )
                    if record["completed"]:
                        completed_epoch_seconds[previous_epoch] = epoch_seconds[previous_epoch]
    started = time.monotonic()
    last_completed = start_epoch - 1
    for epoch in range(start_epoch, train_config["epochs"] + 1):
        model.train()
        model.set_epoch(epoch)
        generator.manual_seed(seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=train_config["batch_size"],
            shuffle=True,
            num_workers=args.workers,
            collate_fn=pointdiff_collate,
            generator=generator,
            pin_memory=device.type == "cuda",
        )
        epoch_started = time.monotonic()
        (sums, gpc_sums, condition_sums, count) = ({}, {}, {}, 0)
        if epoch == start_epoch and resume_step and epoch_accumulators:
            (sums, gpc_sums, count) = (
                dict(epoch_accumulators["losses"]),
                dict(epoch_accumulators.get("gpc", {})),
                epoch_accumulators["queries"],
            )
            condition_sums = dict(epoch_accumulators.get("condition", {}))
        epoch_lr = optimizer.param_groups[0]["lr"]
        (next_epoch, next_batch) = (epoch + 1, 0)
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_step:
                continue
            optimizer.zero_grad(set_to_none=True)
            result = model(model_inputs(batch, device, training=True))
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss at step %d" % global_step)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.get("grad_clip", 1.0)
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradient norm")
            optimizer.step()
            global_step += 1
            size = batch["video"].shape[0]
            values = {key: float(value.detach()) for (key, value) in result["losses"].items()}
            values["total"] = float(loss.detach())
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + size * value
            count += size
            gpc = gpc_statistics(result)
            for key, value in gpc.items():
                gpc_sums[key] = gpc_sums.get(key, 0.0) + size * value
            record = {
                "epoch": epoch,
                "batch": batch_index,
                "step": global_step,
                "queries": size,
                "losses": values,
                "gradient_norm": float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
            }
            if gpc:
                record["gpc"] = gpc
            condition = condition_statistics(result)
            if condition:
                record["condition"] = condition
                for key, value in condition.items():
                    condition_sums[key] = condition_sums.get(key, 0.0) + size * value
            log.write(json.dumps(record) + "\n")
            if args.max_steps and global_step >= args.max_steps:
                if batch_index + 1 < len(loader):
                    (next_epoch, next_batch) = (epoch, batch_index + 1)
                break
        if next_batch == 0:
            last_completed = epoch
        improved = False
        epoch_metrics = None
        if (
            validation_enabled
            and next_batch == 0
            and (epoch % val_config.get("every", 1) == 0 or epoch == train_config["epochs"])
        ):
            metrics = validate_during_training(
                model,
                validation_loader,
                device,
                config["data"]["dataset_name"],
                settings["sampling_steps"],
                val_config.get("seed", seed),
                candidate_diagnostics=val_config.get("candidate_diagnostics", False),
                postprocessing=settings["postprocessing"],
            )
            metrics["mean_recall"] = sum(metrics[key] for key in RECALL_METRICS) / len(
                RECALL_METRICS
            )
            epoch_metrics = metrics
            selection = val_config.get("selection_metric", "R@1_IoU=0.5")
            if selection not in metrics:
                raise ValueError("Unknown validation selection metric: " + selection)
            value = metrics[selection]
            improved = best_validation is None or value > best_validation["value"]
            if improved:
                best_validation = selection_record(epoch, settings, selection, metrics)
            last_validation = {
                "event": "validation",
                "epoch": epoch,
                "global_step": global_step,
                "metrics": metrics,
                "selection_metric": selection,
                "is_best": improved,
                "best_epoch": best_validation["epoch"],
                "best_value": best_validation["value"],
                "seed": val_config.get("seed", seed),
                **settings,
            }
            validation_log.write(json.dumps(last_validation) + "\n")
            csv_metrics = {**metrics, **metrics.get("candidate_diagnostics", {})}
            row = {key: csv_metrics.get(key, "") for key in csv_fields}
            row.update(
                {
                    "epoch": epoch,
                    "is_best": improved,
                    "sampling_steps": settings["sampling_steps"],
                    "postprocessing": settings["postprocessing"]["method"],
                }
            )
            csv_writer.writerow(row)
            validation_log.flush()
            csv_handle.flush()
            atomic_json(output / "validation_latest.json", last_validation)
            atomic_json(output / "best_validation.json", best_validation)
        if scheduler is not None and next_batch == 0:
            scheduler.step()
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "global_step": global_step,
            "rng": rng_state(),
            "train_limit": args.limit_train,
            "completed_epochs": last_completed,
            "validation_limit": args.limit_validation,
            "best_validation": best_validation,
            "last_validation": last_validation,
            "epoch_accumulators": (
                {"losses": sums, "gpc": gpc_sums, "condition": condition_sums, "queries": count}
                if next_batch
                else None
            ),
        }
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        atomic_checkpoint(output / "last.pt", checkpoint)
        if improved:
            atomic_checkpoint(
                output / "best.pt", {**checkpoint, "selected_validation": best_validation}
            )
        if epoch % train_config.get("checkpoint_every", 10) == 0 or epoch == train_config["epochs"]:
            atomic_checkpoint(output / ("epoch_%03d.pt" % epoch), checkpoint)
        summary = {
            "epoch": epoch,
            "completed": next_batch == 0,
            "global_step": global_step,
            "seconds": time.monotonic() - epoch_started,
            "lr": epoch_lr,
            "next_lr": optimizer.param_groups[0]["lr"],
            "losses": {key: value / max(count, 1) for (key, value) in sums.items()},
        }
        if gpc_sums:
            summary["gpc"] = {key: value / max(count, 1) for (key, value) in gpc_sums.items()}
        if condition_sums:
            summary["condition"] = {
                key: value / max(count, 1) for (key, value) in condition_sums.items()
            }
        if validation_enabled:
            summary["validation"] = last_validation
            summary["best_validation"] = best_validation
        epoch_seconds[epoch] = epoch_seconds.get(epoch, 0.0) + summary["seconds"]
        if next_batch == 0:
            completed_epoch_seconds[epoch] = epoch_seconds[epoch]
        remaining_seconds = estimate_remaining_seconds(
            list(completed_epoch_seconds.values()),
            train_config["epochs"] - last_completed,
            epoch_seconds[epoch] if next_batch else 0.0,
        )
        elapsed_seconds = prior_elapsed + time.monotonic() - started
        summary["estimated_remaining_seconds"] = remaining_seconds
        summary["estimated_total_seconds"] = (
            elapsed_seconds + remaining_seconds if remaining_seconds is not None else None
        )
        log.flush()
        epoch_log.write(json.dumps(summary) + "\n")
        epoch_log.flush()
        message = "Epoch %03d/%d | loss %.4f" % (
            epoch,
            train_config["epochs"],
            summary["losses"].get("total", 0.0),
        )
        loss_details = " ".join(
            "%s=%.4f" % (key, value) for key, value in summary["losses"].items() if key != "total"
        )
        if loss_details:
            message += " | " + loss_details
        message += " | time " + format_duration(summary["seconds"])
        if remaining_seconds is not None:
            message += " | est_total %s | remaining %s" % (
                format_duration(summary["estimated_total_seconds"]),
                format_duration(remaining_seconds),
            )
        else:
            message += " | est_total estimating | remaining estimating"
        if epoch_metrics is not None:
            message += " | val " + format_metrics(epoch_metrics)
        if improved:
            message += " [best]"
        if next_batch:
            message += " [partial]"
        print(message, flush=True)
        atomic_json(
            output / "training_summary.json",
            {
                **summary,
                "completed_epochs": last_completed,
                "requested_epochs": train_config["epochs"],
                "train_queries": len(dataset),
                "elapsed_seconds": elapsed_seconds,
                "run_type": metadata["run_type"],
            },
        )
        if args.max_steps and global_step >= args.max_steps:
            break
    log.close()
    epoch_log.close()
    if validation_log:
        validation_log.close()
        csv_handle.close()


if __name__ == "__main__":
    main()
