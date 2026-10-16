from collections.abc import Mapping
import torch
from torch.nn import functional as F
from torch import nn
from .gaussian import GaussianPriorConstraint
from .geometry import cxw_to_xx, xx_to_cxw
from .decoder import ConditionalDenoiser
from .diffusion import SpanDiffusion
from .objectives import PointDiffObjectives
from .fusion import FeatureFusion, video_position_embedding


class PointDiff(nn.Module):

    def __init__(self, config):
        if not isinstance(config, Mapping):
            raise TypeError("PointDiff config must be a mapping")
        nn.Module.__init__(self)
        self.config = dict(config)
        self.sampling_method = config.get("sampling_method", "momentdiff_ddim")
        if self.sampling_method != "momentdiff_ddim":
            raise ValueError("PointDiff requires consistent MomentDiff DDIM")
        self.loss_span_bounds = config.get("loss_span_bounds", "unclipped")
        if self.loss_span_bounds != "unclipped":
            raise ValueError("PointDiff losses use unclipped center/width predictions")
        self.gpc_device = config.get("gpc_device", "cpu")
        if self.gpc_device not in ("input", "cpu"):
            raise ValueError("gpc_device must be input or cpu")
        self.condition_prior = config.get("condition_prior", "none")
        if self.condition_prior != "none":
            raise ValueError("condition_prior must be none")
        self.record_condition_diagnostics = bool(config.get("record_condition_diagnostics", False))
        self.noise_loss_mode = config.get("noise_loss_mode", "auxiliary")
        self.matching_method = config.get("matching_method", "momentdiff_hungarian")
        self.auxiliary_loss = bool(config.get("auxiliary_loss", True))
        self.auxiliary_loss_weight = float(config.get("auxiliary_loss_weight", 1.0))
        if not 0 <= self.auxiliary_loss_weight < float("inf"):
            raise ValueError("Auxiliary loss weight must be finite and nonnegative")
        self.gpc_feature_source = "native_raw_saliency_learned_dga"
        dimension = int(config.get("hidden_dim", 256))
        heads = int(config.get("heads", 8))
        if dimension < 8 or dimension % 4 or dimension % heads:
            raise ValueError("hidden_dim must be >=8 and divisible by 4 and heads")
        self.hidden_dim = dimension
        self.num_spans = int(config.get("num_spans", 8))
        if self.num_spans < 2:
            raise ValueError("At least two proposals are required")
        dropout = float(config.get("dropout", 0.1))
        ffn_dim = int(config.get("ffn_dim", 2048))
        self.video_projection = nn.Linear(int(config.get("video_dim", 1024)), dimension)
        self.text_projection = nn.Linear(int(config.get("text_dim", 768)), dimension)
        self.input_dropout = nn.Dropout(dropout)
        self.gpc = GaussianPriorConstraint(
            sigma=config.get("gaussian_sigma", 0.3),
            window=config.get("dga_window", 5),
            dga_threshold=config.get("dga_threshold", 0.9),
            threshold=config.get("gpc_threshold", 0.7),
            lambda1=config.get("lambda1", 0.1),
            lambda2=config.get("lambda2", 0.9),
            weight_threshold=config.get("weight_threshold", 0.6),
            gaussian_coordinates=config.get("gaussian_coordinates", "frame_index"),
            boundary_mode=config.get("gpc_boundary_mode", "point_component"),
        )
        if self.gpc.boundary_mode != "point_component":
            raise ValueError("PointDiff require annotation-anchored point_component boundaries")
        if self.gpc.gaussian_coordinates != "frame_index":
            raise ValueError("PointDiff requires consistent frame-index Gaussian coordinates")
        self.objectives = PointDiffObjectives(
            self.matching_method,
            self.noise_loss_mode,
            config.get("match_cost_span", 10.0),
            config.get("match_cost_giou", 1.0),
            config.get("match_cost_class", 4.0),
            config.get("background_weight", 0.1),
        )
        self.fusion = FeatureFusion(dimension, heads, ffn_dim, dropout)
        self.denoiser = ConditionalDenoiser(
            dimension,
            heads,
            ffn_dim,
            dropout,
            layers=int(config.get("decoder_layers", 2)),
            time_embed_scale=config.get("time_embed_scale", 0.1),
            confidence_mode=self.matching_method,
            noise_loss_mode=self.noise_loss_mode,
            auxiliary_loss=self.auxiliary_loss,
        )
        self.diffusion = SpanDiffusion(
            timesteps=int(config.get("timesteps", 1000)),
            sampling_steps=int(config.get("sampling_steps", 100)),
            eta=float(config.get("ddim_eta", 1.0)),
            scale=2.0,
        )
        self.loss_weights = {
            "mse": float(config.get("loss_mse_weight", 0.5)),
            "l1": float(config.get("loss_l1_weight", 0.5)),
            "iou": float(config.get("loss_iou_weight", 0.1)),
            "ce": float(config.get("loss_ce_weight", 0.2)),
        }
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def _prior(self, video, mask, points, sample_ids=None, update_memory=False, dynamic_video=None):
        if self.gpc_device == "cpu" and video.device.type != "cpu":
            result = self.gpc(
                video.detach().cpu(),
                mask.cpu(),
                points.detach().cpu(),
                sample_ids,
                update_memory=update_memory,
                dynamic_video=dynamic_video.detach().cpu() if dynamic_video is not None else None,
            )
            return {key: value.to(video.device) for (key, value) in result.items()}
        return self.gpc(
            video,
            mask,
            points,
            sample_ids,
            update_memory=update_memory,
            dynamic_video=dynamic_video,
        )

    def _condition(self, features):
        (raw, video, text, video_mask, text_mask, similarity, positions) = features
        points = None
        weights = video_mask.to(video.dtype)
        source = "video_query_only"

        condition = self.fusion(video, text, video_mask, text_mask, weights, similarity, positions)
        return (
            condition,
            {
                "condition_weights": weights,
                "condition_seed_points": points,
                "condition_point_source": source,
            },
        )

    def _training_prior(self, batch):
        if "gpc_video" not in batch or "gpc_video_mask" not in batch:
            raise ValueError("PointDiff training requires native gpc_video and gpc_video_mask")
        video = batch["gpc_video"]
        if video.ndim != 3 or video.shape[-1] != self.video_projection.in_features:
            raise ValueError("Native GPC video must be [B,T_native,video_dim]")
        mask = self._valid_mask(batch["gpc_video_mask"], video, "gpc_video_mask")
        points = batch.get("point")
        if points is None or points.shape != (video.shape[0],):
            raise ValueError("Training requires one normalized annotation point per query")
        if not bool(torch.isfinite(points).all()) or not bool((points >= 0).all()):
            raise ValueError("Training points must be finite and nonnegative")
        sample_ids = batch.get("query_id", batch.get("sample_ids"))
        with torch.no_grad():
            dynamic = self.video_projection(video).relu()
        return self._prior(video, mask, points, sample_ids, True, dynamic)

    @staticmethod
    @torch.no_grad()
    def _condition_diagnostics(features, prior, points):
        (similarity, mask) = (features[5].detach(), features[3])
        lengths = mask.sum(1)
        denominator = (lengths - 1).clamp_min(1)
        positions = torch.arange(mask.shape[1], device=mask.device)[None] / denominator[:, None]
        spans = prior["pseudo_spans"].detach()
        inside = mask & (positions >= spans[:, :1]) & (positions <= spans[:, 1:])
        outside = mask & ~inside
        high = similarity.masked_fill(~mask, float("-inf")).argmax(1)
        low = similarity.masked_fill(~mask, float("inf")).argmin(1)
        point_index = (points.detach().clamp(0, 1) * denominator).round().long()
        point_index = torch.minimum(point_index, lengths - 1)
        point_cosine = similarity.gather(1, point_index[:, None]).squeeze(1)
        (inside_count, outside_count) = (inside.sum(1), outside.sum(1))
        margin_valid = (inside_count > 0) & (outside_count > 0)
        margin = (similarity * inside).sum(1) / inside_count.clamp_min(1) - (
            similarity * outside
        ).sum(1) / outside_count.clamp_min(1)
        return {
            "max_peak_inside_pseudo": inside.gather(1, high[:, None]).squeeze(1),
            "min_peak_inside_pseudo": inside.gather(1, low[:, None]).squeeze(1),
            "point_cosine_percentile": ((similarity <= point_cosine[:, None]) & mask).sum(1)
            / lengths,
            "point_cosine": point_cosine,
            "inside_outside_cosine_margin": margin.masked_fill(~margin_valid, 0),
            "cosine_margin_valid": margin_valid,
        }

    @staticmethod
    def _prediction_output(predictions):
        epsilon = max(1e-06, torch.finfo(predictions["pred_cxw"].dtype).eps)
        spans = cxw_to_xx(predictions["pred_cxw"].clamp(epsilon, 1 - epsilon)).clamp(0, 1)
        scores = predictions["confidence_logits"].softmax(-1)[..., 0]
        best = scores.argmax(1)
        rows = torch.arange(spans.shape[0], device=spans.device)
        return {
            "spans": spans,
            "scores": scores,
            "pred_cxw": predictions["pred_cxw"],
            "confidence_logits": predictions["confidence_logits"],
            "top_spans": spans[rows, best],
            "top_scores": scores[rows, best],
        }

    def forward(self, batch, sampling_steps=None):
        if not self.training:
            return self.sample(batch, sampling_steps)
        features = self._features(batch)
        (condition, condition_info) = self._condition(features)
        prior = self._training_prior(batch)

        # Annotation-driven Gaussian fusion.
        # (_, video, text, video_mask, text_mask, similarity, positions) = features
        # weights = video.new_zeros(video_mask.shape)
        # for row in range(video.shape[0]):
        #     length = int(batch["gpc_video_mask"][row].sum().item())
        #     clips = int(video_mask[row].sum().item())
        #     native_weights = prior["multi_weights"][row, :length]
        #     edges = torch.linspace(0, length, clips + 1).round().long()
        #     for clip in range(clips):
        #         start, end = int(edges[clip]), int(edges[clip + 1])
        #         weights[row, clip] = (
        #             native_weights[start:end].mean()
        #             if start < end
        #             else native_weights[min(start, length - 1)]
        #         )
        # condition = self.fusion(
        #     video, text, video_mask, text_mask, weights, similarity, positions
        # )
        # condition_info = {
        #     "condition_weights": weights,
        #     "condition_seed_points": batch["point"],
        #     "condition_point_source": "training_annotation_shared_dga",
        # }

        target_cxw = xx_to_cxw(prior["pseudo_spans"])
        prepared = self.diffusion.prepare_targets(target_cxw, self.num_spans)
        predictions = self.denoiser(
            prepared["noisy_spans"], prepared["steps"], condition, features[-1], features[3]
        )
        epsilon_sampler = self.diffusion.predict_noise_from_start(
            prepared["latent"], prepared["steps"], self.diffusion.expand(predictions["pred_cxw"])
        )
        (losses, match_info) = self.objectives(
            predictions, target_cxw, prepared["noise"], epsilon_sampler
        )
        total_loss = sum((self.loss_weights[key] * value for (key, value) in losses.items()))
        auxiliary_matches = []
        for layer_index, intermediate in enumerate(predictions.get("aux_outputs", [])):
            (aux_losses, aux_info) = self.objectives(
                intermediate, target_cxw, None, None, compute_noise=False
            )
            total_loss = total_loss + self.auxiliary_loss_weight * sum(
                (self.loss_weights[key] * value for (key, value) in aux_losses.items())
            )
            losses.update(
                {key + "_aux_%d" % layer_index: value for (key, value) in aux_losses.items()}
            )
            auxiliary_matches.append(aux_info["matched_indices"])
        result = self._prediction_output(predictions)
        result.update(
            {
                "loss": total_loss,
                "losses": losses,
                "pseudo_spans": prior["pseudo_spans"],
                "prior_weights": prior["constraint_weights"],
                "diffusion_targets": prepared["clean_candidates"],
                "point_source": "training_annotation_target_only",
                **condition_info,
                **match_info,
            }
        )
        if self.auxiliary_loss:
            result["aux_matched_indices"] = auxiliary_matches
        point = batch["point"].clamp(0, 1)
        width = prior["pseudo_spans"][:, 1] - prior["pseudo_spans"][:, 0]
        native_length = batch["gpc_video_mask"].sum(1)
        result["gpc_diagnostics"] = {
            "point_contained": (prior["pseudo_spans"][:, 0] <= point)
            & (point <= prior["pseudo_spans"][:, 1]),
            "width": width,
            "single_frame": (native_length > 1)
            & (width <= 1.01 / (native_length - 1).clamp_min(1)),
            "point_below_weight_threshold": prior["point_below_weight_threshold"],
            "point_clamped": batch["point"] != point,
        }
        if self.record_condition_diagnostics:
            result["condition_diagnostics"] = self._condition_diagnostics(
                features, prior, batch["point"]
            )
        return result

    @torch.no_grad()
    def sample(self, batch, sampling_steps=None):
        features = self._features(batch)
        (video, video_mask, positions) = (features[1], features[3], features[-1])
        (condition, condition_info) = self._condition(features)
        latent = torch.randn(
            video.shape[0], self.num_spans, 2, device=video.device, dtype=video.dtype
        )
        clipped = latent.new_zeros((), dtype=torch.long)
        elements = 0
        for step, next_step in self.diffusion.time_pairs(sampling_steps):
            clipped += (latent.abs() > self.diffusion.scale).sum()
            elements += latent.numel()
            steps = torch.full((video.shape[0],), step, dtype=torch.long, device=video.device)
            predictions = self.denoiser(
                self.diffusion.normalize(latent), steps, condition, positions, video_mask
            )
            latent = self.diffusion.ddim_step(latent, predictions["pred_cxw"], step, next_step)
        result = self._prediction_output(predictions)
        result.update(
            {
                **condition_info,
                "point_source": "not_used_for_sampling",
                "sampling_method": self.sampling_method,
                "diffusion_diagnostics": {
                    "clipped_coordinates": clipped,
                    "total_coordinates": elements,
                    "terminal_latent_maxabs": latent.abs().max(),
                    "terminal_span_consistency_max_error": (
                        latent - self.diffusion.expand(predictions["pred_cxw"])
                    )
                    .abs()
                    .max(),
                },
            }
        )
        return result

    def set_epoch(self, epoch):
        self.gpc.set_epoch(epoch)

    @staticmethod
    def _valid_mask(mask, features, name):
        if mask.shape != features.shape[:2]:
            raise ValueError("{} must match the first two feature dimensions".format(name))
        mask = mask.to(device=features.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every sample must contain at least one valid {} item".format(name))
        expected = (
            torch.arange(mask.shape[1], device=mask.device)[None, :] < mask.sum(dim=1)[:, None]
        )
        if not torch.equal(mask, expected):
            raise ValueError("{} must describe right-padded contiguous valid tokens".format(name))
        return mask

    def _features(self, batch):
        (video, text) = (batch["video"], batch["text"])
        if video.ndim != 3 or text.ndim != 3 or video.shape[0] != text.shape[0]:
            raise ValueError("video/text must have shape [B,T,Dv]/[B,L,Dq] with the same B")
        video_mask = self._valid_mask(batch["video_mask"], video, "video_mask")
        text_mask = self._valid_mask(batch["text_mask"], text, "text_mask")
        projected_video = self.input_dropout(self.video_projection(video))
        projected_text = self.input_dropout(self.text_projection(text))
        token_cosine = torch.matmul(
            F.normalize(projected_video, dim=-1),
            F.normalize(projected_text, dim=-1).transpose(1, 2),
        )
        similarity = (token_cosine * text_mask[:, None, :]).sum(-1) / text_mask.sum(-1)[:, None]
        similarity = similarity * video_mask
        positions = video_position_embedding(video_mask, self.hidden_dim, projected_video.dtype)
        return (
            video,
            projected_video,
            projected_text,
            video_mask,
            text_mask,
            similarity,
            positions,
        )
