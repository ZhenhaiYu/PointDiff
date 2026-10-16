import torch
from torch import nn
from torch.nn import functional as F
from .geometry import cxw_to_xx, temporal_iou


class PointDiffObjectives(nn.Module):

    def __init__(
        self,
        matching_method="momentdiff_hungarian",
        noise_loss_mode="auxiliary",
        cost_span=10.0,
        cost_giou=1.0,
        cost_class=4.0,
        background_weight=0.1,
    ):
        super().__init__()
        if matching_method not in ("momentdiff_hungarian",):
            raise ValueError("matching_method must be best_iou or momentdiff_hungarian")
        if noise_loss_mode not in ("auxiliary",):
            raise ValueError("noise_loss_mode must be auxiliary or consistent")
        if min(cost_span, cost_giou, cost_class) < 0 or cost_span + cost_giou + cost_class <= 0:
            raise ValueError("Matching costs must be nonnegative with at least one positive")
        if background_weight < 0:
            raise ValueError("Background weight must be nonnegative")
        self.matching_method = matching_method
        self.noise_loss_mode = noise_loss_mode
        (self.cost_span, self.cost_giou, self.cost_class) = (cost_span, cost_giou, cost_class)
        self.register_buffer("class_weights", torch.tensor([1.0, background_weight]))

    def forward(self, predictions, target_cxw, noise, epsilon_sampler, compute_noise=True):
        spans = cxw_to_xx(predictions["pred_cxw"])
        target = cxw_to_xx(target_cxw)
        ious = temporal_iou(spans, target[:, None])
        rows = torch.arange(spans.shape[0], device=spans.device)
        logits = predictions["confidence_logits"]
        enclosing = torch.maximum(spans[..., 1], target[:, None, 1]) - torch.minimum(
            spans[..., 0], target[:, None, 0]
        )
        intersection = (
            torch.minimum(spans[..., 1], target[:, None, 1])
            - torch.maximum(spans[..., 0], target[:, None, 0])
        ).clamp_min(0)
        union = (
            spans[..., 1] - spans[..., 0] + target[:, None, 1] - target[:, None, 0] - intersection
        )
        giou = ious - (enclosing - union) / enclosing.clamp_min(1e-08)
        l1 = (predictions["pred_cxw"] - target_cxw[:, None]).abs().sum(-1)
        cost = (
            self.cost_span * l1
            - self.cost_giou * giou
            - self.cost_class * logits.softmax(-1)[..., 0]
        )
        matched = cost.detach().argmin(1)
        classes = torch.ones(spans.shape[:2], dtype=torch.long, device=spans.device)
        classes[rows, matched] = 0
        ce = F.cross_entropy(
            logits.transpose(1, 2), classes, weight=self.class_weights, reduction="none"
        ).mean()
        losses = {
            "l1": F.l1_loss(predictions["pred_cxw"][rows, matched], target_cxw),
            "iou": (1 - ious[rows, matched]).mean(),
            "ce": ce,
        }
        if compute_noise:
            eps = predictions["pred_noise_aux"]
            losses = {"mse": F.mse_loss(eps, noise), **losses}
        return (losses, {"matched_indices": matched, "loss_spans": spans, "candidate_ious": ious})
