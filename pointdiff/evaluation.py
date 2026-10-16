import json
import random
import time
import numpy as np
import torch
from .runtime import model_inputs, rng_state, restore_rng
from .postprocessing import postprocessing_config, rerank_spans


def temporal_iou(spans, target):
    intersection = (
        torch.minimum(spans[:, 1], target[1]) - torch.maximum(spans[:, 0], target[0])
    ).clamp(min=0)
    union = spans[:, 1] - spans[:, 0] + (target[1] - target[0]) - intersection
    return intersection / union.clamp(min=1e-08)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    dataset_name,
    sampling_steps=None,
    predictions_path=None,
    progress_every=0,
    candidate_diagnostics=False,
    postprocessing=None,
):
    postprocessing = postprocessing_config(postprocessing)
    processed = postprocessing["method"] != "none"
    thresholds = [0.5, 0.7] if dataset_name == "charades" else [0.3, 0.5, 0.7]
    hits = {"R@%d_IoU=%.1f" % (rank, threshold): 0 for rank in [1, 5] for threshold in thresholds}
    (raw_hits, raw_sum_iou, retained, fewer_than_five) = (dict(hits), 0.0, 0, 0)
    (count, sum_iou) = (0, 0.0)
    (clipped, total, terminal_maxabs, consistency) = (0, 0, 0.0, 0.0)
    oracle_hits = {threshold: 0 for threshold in thresholds}
    missed_top1 = {threshold: 0 for threshold in thresholds}
    (top1_best, pair_iou_sum, pair_count, num_candidates) = (0, 0.0, 0, None)
    started = time.monotonic()
    handle = predictions_path.open("w") if predictions_path is not None else None
    try:
        for batch_index, batch in enumerate(loader):
            result = model(
                model_inputs(batch, device, training=False), sampling_steps=sampling_steps
            )
            if "diffusion_diagnostics" in result:
                diag = result["diffusion_diagnostics"]
                clipped += int(diag["clipped_coordinates"])
                total += diag["total_coordinates"]
                terminal_maxabs = max(terminal_maxabs, float(diag["terminal_latent_maxabs"]))
                consistency = max(consistency, float(diag["terminal_span_consistency_max_error"]))
            (spans, scores) = (result["spans"].cpu(), result["scores"].cpu())
            if not torch.isfinite(spans).all() or not torch.isfinite(scores).all():
                raise FloatingPointError("Non-finite predictions")
            if (spans < 0).any() or (spans > 1).any() or (spans[..., 1] <= spans[..., 0]).any():
                raise ValueError("Predicted normalized spans must satisfy 0 <= start < end <= 1")
            for index in range(spans.shape[0]):
                order = torch.argsort(scores[index], descending=True)
                duration = float(batch["duration"][index])
                ranked = spans[index, order] * duration
                ranked_scores = scores[index, order]
                target = batch["target"][index].cpu()
                ious = temporal_iou(ranked, target)
                if candidate_diagnostics:
                    if num_candidates is not None and len(ranked) != num_candidates:
                        raise ValueError(
                            "Candidate diagnostics require a consistent proposal count"
                        )
                    num_candidates = len(ranked)
                    best_iou = ious.max()
                    top1_best += int(ious[0] >= best_iou - 1e-06)
                    for threshold in thresholds:
                        oracle_hits[threshold] += int(best_iou >= threshold)
                        missed_top1[threshold] += int(
                            (ious[0] < threshold) & (best_iou >= threshold)
                        )
                    if num_candidates > 1:
                        intersection = (
                            torch.minimum(ranked[:, None, 1], ranked[None, :, 1])
                            - torch.maximum(ranked[:, None, 0], ranked[None, :, 0])
                        ).clamp_min(0)
                        widths = ranked[:, 1] - ranked[:, 0]
                        pairwise = intersection / (
                            widths[:, None] + widths[None, :] - intersection
                        ).clamp_min(1e-08)
                        indices = torch.triu_indices(num_candidates, num_candidates, offset=1)
                        values = pairwise[indices[0], indices[1]]
                        pair_iou_sum += float(values.sum())
                        pair_count += len(values)
                if processed:
                    for rank in [1, 5]:
                        for threshold in thresholds:
                            raw_hits["R@%d_IoU=%.1f" % (rank, threshold)] += int(
                                ious[:rank].max() >= threshold
                            )
                    raw_sum_iou += float(ious[0])
                    (selected, selected_scores) = rerank_spans(
                        ranked, ranked_scores, postprocessing
                    )
                    (original_ranked, original_scores) = (ranked, ranked_scores)
                    (ranked, ranked_scores, ious) = (
                        ranked[selected],
                        selected_scores,
                        ious[selected],
                    )
                    retained += len(selected)
                    fewer_than_five += int(len(selected) < 5)
                for rank in [1, 5]:
                    for threshold in thresholds:
                        hits["R@%d_IoU=%.1f" % (rank, threshold)] += int(
                            ious[:rank].max() >= threshold
                        )
                count += 1
                sum_iou += float(ious[0])
                if handle is not None:
                    prediction = {
                        "query_id": batch["query_id"][index],
                        "video_id": batch["video_id"][index],
                        "query": batch["query"][index],
                        "duration": duration,
                        "predicted_spans": ranked.tolist(),
                        "scores": ranked_scores.tolist(),
                        "target": target.tolist(),
                        "top1_iou": float(ious[0]),
                    }
                    if processed:
                        prediction.update(
                            {
                                "postprocessing": postprocessing,
                                "original_ranked_indices": selected.tolist(),
                                "raw_predicted_spans": original_ranked.tolist(),
                                "raw_scores": original_scores.tolist(),
                            }
                        )
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            if progress_every and (batch_index + 1) % progress_every == 0:
                print(
                    json.dumps({"evaluated_queries": count, "seconds": time.monotonic() - started}),
                    flush=True,
                )
    finally:
        if handle is not None:
            handle.close()
    if count == 0:
        raise ValueError("Evaluation requires at least one query")
    metrics = {key: 100.0 * value / count for (key, value) in hits.items()}
    metrics.update(
        {
            "mIoU": 100.0 * sum_iou / count,
            "queries": count,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    if processed:
        metrics.update(
            {
                "postprocessing": postprocessing,
                "raw_metrics": {
                    **{key: 100.0 * value / count for (key, value) in raw_hits.items()},
                    "mIoU": 100.0 * raw_sum_iou / count,
                    "queries": count,
                },
                "postprocessing_diagnostics": {
                    "mean_retained_candidates": retained / count,
                    "fewer_than_five_percent": 100.0 * fewer_than_five / count,
                },
            }
        )
    if total:
        metrics["diffusion_diagnostics"] = {
            "denoiser_input_clipped_percent": 100.0 * clipped / total,
            "terminal_latent_maxabs": terminal_maxabs,
            "terminal_span_consistency_max_error": consistency,
        }
    if candidate_diagnostics:
        metrics["candidate_diagnostics"] = {
            "num_candidates": num_candidates,
            **{
                "oracle_R@all_IoU=%.1f" % threshold: 100.0 * value / count
                for (threshold, value) in oracle_hits.items()
            },
            **{
                "top1_missed_with_good_candidate_IoU=%.1f" % threshold: 100.0 * value / count
                for (threshold, value) in missed_top1.items()
            },
            "top1_is_best_candidate_percent": 100.0 * top1_best / count,
            "candidate_pair_mean_iou": pair_iou_sum / max(pair_count, 1),
        }
        if processed:
            metrics["candidate_diagnostics"]["pool"] = "all_generated_candidates_before_NMS"
    return metrics


def validate_during_training(
    model,
    loader,
    device,
    dataset_name,
    sampling_steps,
    seed,
    candidate_diagnostics=False,
    postprocessing=None,
):
    (state, was_training) = (rng_state(), model.training)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model.eval()
        return evaluate(
            model,
            loader,
            device,
            dataset_name,
            sampling_steps=sampling_steps,
            candidate_diagnostics=candidate_diagnostics,
            postprocessing=postprocessing,
        )
    finally:
        model.train(was_training)
        restore_rng(state)
