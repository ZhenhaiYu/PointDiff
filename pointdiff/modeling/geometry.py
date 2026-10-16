import torch


def cxw_to_xx(spans):
    (center, width) = spans.unbind(-1)
    return torch.stack((center - width / 2, center + width / 2), dim=-1)


def xx_to_cxw(spans):
    (start, end) = spans.unbind(-1)
    return torch.stack(((start + end) / 2, end - start), dim=-1)


def temporal_iou(predictions, targets):
    intersection = (
        torch.minimum(predictions[..., 1], targets[..., 1])
        - torch.maximum(predictions[..., 0], targets[..., 0])
    ).clamp_min(0)
    union = (
        predictions[..., 1] - predictions[..., 0] + targets[..., 1] - targets[..., 0] - intersection
    )
    return intersection / union.clamp_min(1e-08)
