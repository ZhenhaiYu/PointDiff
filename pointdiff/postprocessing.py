import math
import numpy as np
import torch


def postprocessing_config(config=None):
    if config is None:
        config = {"method": "none"}
    if isinstance(config, str):
        config = {"method": config}
    method = config.get("method", "none")
    if method not in ("none", "hard_nms", "soft_nms"):
        raise ValueError("Unknown postprocessing method: " + str(method))
    result = {"method": method}
    if method == "hard_nms":
        threshold = float(config.get("iou_threshold", 0.5))
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("NMS IoU threshold must be in [0, 1]")
        result["iou_threshold"] = threshold
    elif method == "soft_nms":
        sigma = float(config.get("sigma", 0.5))
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Gaussian Soft-NMS requires finite sigma > 0")
        result.update({"sigma": sigma, "score_cutoff": None})
    return result


def rerank_spans(ranked_spans, ranked_scores, config=None):
    config = postprocessing_config(config)
    if config["method"] == "none":
        return (torch.arange(len(ranked_spans)), ranked_scores)
    spans = ranked_spans.detach().cpu().numpy().astype(np.float64)
    scores = ranked_scores.detach().cpu().numpy().astype(np.float64).copy()
    (remaining, kept, selected_scores) = (list(range(len(spans))), [], [])
    while remaining:
        chosen = remaining.pop(int(np.argmax(scores[remaining])))
        kept.append(chosen)
        selected_scores.append(float(scores[chosen]))
        if not remaining:
            break
        rest = np.asarray(remaining)
        intersection = np.maximum(
            0.0,
            np.minimum(spans[chosen, 1], spans[rest, 1])
            - np.maximum(spans[chosen, 0], spans[rest, 0]),
        )
        union = spans[chosen, 1] - spans[chosen, 0] + spans[rest, 1] - spans[rest, 0] - intersection
        overlaps = intersection / np.maximum(union, 1e-08)
        if config["method"] == "hard_nms":
            remaining = rest[overlaps <= config["iou_threshold"]].tolist()
        else:
            scores[rest] *= np.exp(-(overlaps**2) / config["sigma"])
    return (
        torch.tensor(kept, dtype=torch.long),
        torch.tensor(selected_scores, dtype=torch.float64),
    )
