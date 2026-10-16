import math
import torch
from torch import nn
from torch.nn import functional as F


class GaussianPriorConstraint(nn.Module):

    def __init__(
        self,
        sigma=0.3,
        window=5,
        dga_threshold=0.9,
        threshold=0.7,
        lambda1=0.1,
        lambda2=0.9,
        weight_threshold=0.6,
        gaussian_coordinates="frame_index",
        boundary_mode="point_component",
    ):
        super().__init__()
        if sigma <= 0 or window < 0:
            raise ValueError("Gaussian sigma must be positive; DGA window must be nonnegative")
        if not -1 <= threshold <= 1 or not -1 <= dga_threshold <= 1:
            raise ValueError("Cosine thresholds must be in [-1, 1]")
        if lambda1 < 0 or lambda2 < 0 or lambda1 + lambda2 <= 0:
            raise ValueError("Gaussian mixture coefficients must be nonnegative and nonzero")
        if not 0 < weight_threshold <= 1:
            raise ValueError("weight_threshold must be in (0, 1]")
        if gaussian_coordinates not in ("frame_index",):
            raise ValueError("gaussian_coordinates must be frame_index")
        if boundary_mode not in ("point_component",):
            raise ValueError("boundary_mode must be point_component")
        self.sigma = float(sigma)
        self.window = int(window)
        self.dga_threshold = float(dga_threshold)
        self.threshold = float(threshold)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.weight_threshold = float(weight_threshold)
        self.gaussian_coordinates = gaussian_coordinates
        self.boundary_mode = boundary_mode
        self.epoch = 0
        self.memory = {}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def get_extra_state(self):
        return {"version": 1, "epoch": self.epoch, "memory": self.memory}

    def set_extra_state(self, state):
        if state is None:
            (self.epoch, self.memory) = (0, {})
        elif state.get("version") == 1:
            self.epoch = int(state["epoch"])
            self.memory = {
                key: {
                    "counter": value["counter"].detach().cpu().clone(),
                    "similarity": value["similarity"].detach().cpu().clone(),
                    "epoch": int(value["epoch"]),
                    "point_index": int(value["point_index"]),
                }
                for (key, value) in state["memory"].items()
            }
        else:
            raise ValueError("Unsupported PointDiff DGA memory version")

    @staticmethod
    def _sample_key(value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        return str(value)

    def _initial_counter(self, length, point_index, device):
        indices = torch.arange(length, device=device)
        counter = torch.full((length,), -1.0, device=device)
        counter[(indices - point_index).abs() <= self.window] = 0
        counter[point_index] = 3
        return counter

    def _update_dga(self, similarity, point_index, sample_id, update):
        length = similarity.numel()
        state = self.memory.get(self._sample_key(sample_id)) if update else None
        if state is not None:
            if state["counter"].numel() != length or state["point_index"] != point_index:
                raise ValueError("A query's clip length/point changed within persistent DGA memory")
            if state["epoch"] == self.epoch:
                return (
                    state["counter"].to(similarity.device),
                    state["similarity"].to(similarity.device),
                )
            counter = state["counter"].to(similarity.device)
            similarity = 0.7 * similarity + 0.3 * state["similarity"].to(similarity.device)
        else:
            counter = self._initial_counter(length, point_index, similarity.device)
        positive = (counter >= 0) & (similarity >= self.dga_threshold)
        counter[positive] += 1
        negative = (counter >= 1) & (similarity < self.dga_threshold)
        counter[negative] -= 1
        if update:
            self.memory[self._sample_key(sample_id)] = {
                "counter": counter.detach().cpu(),
                "similarity": similarity.detach().cpu(),
                "epoch": self.epoch,
                "point_index": point_index,
            }
        return (counter, similarity)

    def _gaussian(self, positions, means):
        weights = torch.exp(
            -(positions.unsqueeze(0) - means.unsqueeze(-1)).square() / (2 * self.sigma**2)
        )
        weights = weights / (math.sqrt(2 * math.pi) * self.sigma)
        return weights / weights.amax(dim=-1, keepdim=True).clamp_min(1e-08)

    @torch.no_grad()
    def forward(
        self, video, video_mask, points, sample_ids=None, update_memory=False, dynamic_video=None
    ):
        (batch_size, padded_length, _) = video.shape
        if update_memory and (sample_ids is None or len(sample_ids) != batch_size):
            raise ValueError("Training requires one stable query_id (or sample_ids) per sample")
        if not update_memory:
            sample_ids = [None] * batch_size
        lengths = video_mask.sum(dim=1)
        raw_normalized = F.normalize(video.detach(), dim=-1)
        if dynamic_video is not None and dynamic_video.shape[:2] != video.shape[:2]:
            raise ValueError("DGA features must share the native saliency frame grid")
        dynamic_normalized = (
            F.normalize(dynamic_video.detach(), dim=-1)
            if dynamic_video is not None
            else raw_normalized
        )
        multi = video.new_zeros((batch_size, padded_length))
        constrained = torch.zeros_like(multi)
        saliency = torch.zeros_like(multi)
        salient_weights = torch.zeros_like(multi)
        pseudo = video.new_zeros((batch_size, 2))
        (centers, boundaries, point_indices, point_below_threshold) = ([], [], [], [])
        for row in range(batch_size):
            length = int(lengths[row].item())
            point_index = min(
                max(int(torch.round(points[row] * (length - 1)).item()), 0), length - 1
            )
            similarity = torch.matmul(
                raw_normalized[row, :length], raw_normalized[row, point_index]
            )
            similarity[point_index] = 1
            indices = torch.arange(length, device=video.device)
            candidates = similarity >= self.threshold
            distances = (indices - point_index).abs().masked_fill(~candidates, -1)
            boundary = int(distances.argmax().item())
            center = (point_index + boundary) / 2
            positions = torch.linspace(-1, 1, length, device=video.device, dtype=video.dtype)
            denominator = max(length - 1, 1)
            dga_similarity = (
                similarity
                if dynamic_video is None
                else torch.matmul(
                    dynamic_normalized[row, :length], dynamic_normalized[row, point_index]
                )
            )
            dga_similarity[point_index] = 1
            (counter, dynamic_similarity) = self._update_dga(
                dga_similarity, point_index, sample_ids[row], update_memory
            )
            active = counter >= 1
            means = indices.to(video.dtype) / denominator * 2 - 1
            gaussians = self._gaussian(positions, means[active])
            wmulti = (gaussians * dynamic_similarity[active, None]).sum(0) / active.sum()
            wg = self._gaussian(positions, video.new_tensor([center / denominator * 2 - 1]))[0]
            wc = self.lambda1 * wg + self.lambda2 * wmulti
            support = wc >= wc.max() * self.weight_threshold
            seed = point_index
            point_below_threshold.append(not bool(support[point_index]))
            support[seed] = True
            (start, end) = (seed, seed)
            while start > 0 and bool(support[start - 1]):
                start -= 1
            while end < length - 1 and bool(support[end + 1]):
                end += 1
            point = float(points[row].clamp(0, 1))
            left = max(0.0, (start - 0.5) / (length - 1)) if length > 1 else 0.0
            right = min(1.0, (end + 0.5) / (length - 1)) if length > 1 else 1.0
            pseudo[row] = video.new_tensor([min(left, point), max(right, point)])
            (multi[row, :length], constrained[row, :length]) = (wmulti, wc)
            saliency[row, :length] = similarity
            salient_weights[row, :length] = wg
            centers.append(center)
            boundaries.append(boundary)
            point_indices.append(point_index)
        result = {
            "multi_weights": multi,
            "constraint_weights": constrained,
            "pseudo_spans": pseudo,
            "saliency_similarity": saliency,
            "salient_weights": salient_weights,
            "salient_centers": video.new_tensor(centers),
            "salient_boundaries": torch.tensor(boundaries, device=video.device),
            "point_indices": torch.tensor(point_indices, device=video.device),
            "point_below_weight_threshold": torch.tensor(
                point_below_threshold, device=video.device
            ),
        }
        return result
