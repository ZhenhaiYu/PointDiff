import argparse
from pointdiff.runtime import (
    atomic_json,
    configure_runtime,
    make_dataset,
    project_path,
    read_config,
    source_fingerprints,
    implementation_info,
    inference_protocol,
)
import torch
from torch.utils.data import DataLoader
from pointdiff.data import pointdiff_collate
from pointdiff.modeling import build_model, model_config
from pointdiff.evaluation import evaluate
from pointdiff.postprocessing import postprocessing_config
from pointdiff.validation import RECALL_METRICS, evaluation_settings, format_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/charades_i3d.json")
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--postprocessing", "--nms", dest="postprocessing", choices=["none", "hard_nms", "soft_nms"]
    )
    parser.add_argument("--nms-threshold", type=float)
    parser.add_argument("--soft-nms-sigma", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", choices=["eval", "validation", "test"], default="eval")
    parser.add_argument(
        "--candidate-diagnostics",
        action="store_true",
        help="Also report all-candidate recall, selection accuracy and overlap",
    )
    args = parser.parse_args()
    checkpoint_path = project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = read_config(args.config)
    settings = evaluation_settings(config)
    saved_model_config = model_config(checkpoint["config"]["model"])
    proposed_model_config = model_config(config["model"])
    for key in ["sampling_steps", "ddim_eta", "gpc_device"]:
        saved_model_config.pop(key, None)
        proposed_model_config.pop(key, None)
    if saved_model_config != proposed_model_config:
        raise ValueError("Inference model configuration does not match checkpoint")
    val_config = config.get("validation", {}) if args.split == "validation" else {}
    selected = checkpoint.get("selected_validation", {})
    postprocessing = settings["postprocessing"]
    if args.postprocessing is not None and args.postprocessing != postprocessing["method"]:
        postprocessing = postprocessing_config(args.postprocessing)
    if args.nms_threshold is not None:
        if postprocessing["method"] != "hard_nms":
            parser.error("--nms-threshold requires hard_nms")
        postprocessing["iou_threshold"] = args.nms_threshold
    if args.soft_nms_sigma is not None:
        if postprocessing["method"] != "soft_nms":
            parser.error("--soft-nms-sigma requires soft_nms")
        postprocessing["sigma"] = args.soft_nms_sigma
    postprocessing = postprocessing_config(postprocessing)
    batch_size = (
        args.batch_size if args.batch_size is not None else val_config.get("batch_size", 48)
    )
    if batch_size < 1:
        parser.error("Batch size must be positive")
    seed = args.seed if args.seed is not None else val_config.get("seed", config.get("seed", 25285))
    device = configure_runtime(seed, args.threads, args.device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = make_dataset(config, False, args.limit_eval, split=args.split)
    if not len(dataset):
        raise RuntimeError("No evaluation queries are available")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=pointdiff_collate,
        pin_memory=device.type == "cuda",
    )
    if args.split == "validation":
        configure_runtime(seed, args.threads, str(device))
    default_folder = "validation_evaluation" if args.split == "validation" else "evaluation"
    output = project_path(args.output) if args.output else checkpoint_path.parent / default_folder
    output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(
        model,
        loader,
        device,
        config["data"]["dataset_name"],
        predictions_path=output / "predictions.jsonl",
        candidate_diagnostics=args.candidate_diagnostics
        or val_config.get("candidate_diagnostics", False)
        or config.get("evaluation", {}).get("candidate_diagnostics", False),
        postprocessing=postprocessing,
    )
    metrics["mean_recall"] = sum(metrics[key] for key in RECALL_METRICS) / len(RECALL_METRICS)
    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "completed_training_epochs": checkpoint.get("completed_epochs"),
            "sampling_steps": config["model"]["sampling_steps"],
            "seed": seed,
            "batch_size": batch_size,
            "implementation": implementation_info(model),
            "device": str(device),
            "eval_limit": args.limit_eval,
            "source_sha256": source_fingerprints(),
            "protocol": inference_protocol(model),
            "postprocessing": postprocessing,
            "selected_validation": selected or None,
            "reported_units": "Recall and mIoU are percentages; postprocessing is explicitly recorded.",
        }
    )
    atomic_json(output / "metrics.json", metrics)
    print("%s | %s" % (args.split.capitalize(), format_metrics(metrics)), flush=True)
    print("Results: %s" % (output / "metrics.json"), flush=True)


if __name__ == "__main__":
    main()
