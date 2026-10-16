import math
from torch import nn
import torch


class SpanDiffusion(nn.Module):

    def __init__(self, timesteps=1000, sampling_steps=100, eta=1.0, scale=2.0):
        super().__init__()
        if timesteps < 2 or not 1 <= sampling_steps <= timesteps:
            raise ValueError("Require timesteps >= 2 and 1 <= sampling_steps <= timesteps")
        if not 0 <= eta <= 1:
            raise ValueError("DDIM eta must be in [0, 1]")
        self.timesteps = int(timesteps)
        self.sampling_steps = int(sampling_steps)
        self.eta = float(eta)
        self.scale = float(scale)
        betas = cosine_beta_schedule(timesteps)
        cumulative = (1 - betas).cumprod(0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", cumulative.float())
        self.register_buffer("sqrt_alphas_cumprod", cumulative.sqrt().float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - cumulative).sqrt().float())
        self.register_buffer("sqrt_recip_alphas_cumprod", (1 / cumulative).sqrt().float())
        self.register_buffer("sqrt_recipm1_alphas_cumprod", (1 / cumulative - 1).sqrt().float())

    def predict_noise_from_start(self, raw_xt, steps, expanded_x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, steps, raw_xt.shape) * raw_xt - expanded_x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, steps, raw_xt.shape)

    def prepare_targets(self, pseudo_cxw, num_proposals):
        if pseudo_cxw.ndim != 2 or pseudo_cxw.shape[-1] != 2 or num_proposals < 1:
            raise ValueError("Expected one [B,2] GPC target and positive proposal count")
        (latents, noises, times, candidates) = ([], [], [], [])
        for target in pseudo_cxw:
            step = torch.randint(self.timesteps, (1,), device=target.device)
            noise = torch.randn(num_proposals, 2, device=target.device, dtype=target.dtype)
            placeholders = (
                torch.randn(num_proposals - 1, 2, device=target.device, dtype=target.dtype) / 6
                + 0.5
            )
            placeholders[:, 1].clamp_(min=0.01)
            clean = torch.cat((target[None], placeholders), dim=0)
            latent = self.q_sample(clean[None], step, noise[None])[0]
            latents.append(latent)
            noises.append(noise)
            times.append(step)
            candidates.append(clean)
        raw_xt = torch.stack(latents)
        return {
            "latent": raw_xt,
            "noisy_spans": self.normalize(raw_xt),
            "noise": torch.stack(noises),
            "steps": torch.cat(times),
            "clean_candidates": torch.stack(candidates),
        }

    def ddim_step(self, raw_xt, predicted_cxw, step, next_step, noise=None):
        if not 0 <= step < self.timesteps or not -1 <= next_step < step:
            raise ValueError("Require -1 <= next_step < step < timesteps")
        x0 = self.expand(predicted_cxw).clamp(-self.scale, self.scale)
        if next_step < 0:
            return x0
        steps = torch.full((raw_xt.shape[0],), step, device=raw_xt.device, dtype=torch.long)
        epsilon = self.predict_noise_from_start(raw_xt, steps, x0)
        if noise is None:
            noise = torch.randn_like(raw_xt)
        alpha = self.alphas_cumprod[step]
        next_alpha = self.alphas_cumprod[next_step]
        sigma = (
            self.eta
            * ((1 - alpha / next_alpha) * (1 - next_alpha) / (1 - alpha)).clamp_min(0).sqrt()
        )
        coefficient = (1 - next_alpha - sigma.square()).clamp_min(0).sqrt()
        return next_alpha.sqrt() * x0 + coefficient * epsilon + sigma * noise

    def normalize(self, latent):
        return (latent.clamp(-self.scale, self.scale) / self.scale + 1) / 2

    def expand(self, spans):
        return (spans * 2 - 1) * self.scale

    def q_sample(self, spans, steps, noise):
        x0 = self.expand(spans)
        return (
            extract(self.sqrt_alphas_cumprod, steps, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, steps, x0.shape) * noise
        )

    def time_pairs(self, sampling_steps=None):
        count = self.sampling_steps if sampling_steps is None else int(sampling_steps)
        if not 1 <= count <= self.timesteps:
            raise ValueError("sampling_steps must be in [1, timesteps]")
        times = torch.linspace(-1, self.timesteps - 1, count + 1).long().tolist()[::-1]
        return list(zip(times[:-1], times[1:]))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    cumulative = torch.cos((steps / timesteps + s) / (1 + s) * math.pi / 2) ** 2
    cumulative = cumulative / cumulative[0]
    return (1 - cumulative[1:] / cumulative[:-1]).clamp(0, 0.999)


def extract(values, steps, shape):
    return values.gather(0, steps).reshape(steps.shape[0], *(1,) * (len(shape) - 1))
