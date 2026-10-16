import math
import torch
from torch import nn


def scalar_position_embedding(values, dimension):
    if dimension % 2:
        raise ValueError("Sinusoidal embedding dimension must be even")
    frequencies = torch.arange(dimension, device=values.device, dtype=torch.float32)
    frequencies = 10000 ** (2 * torch.div(frequencies, 2, rounding_mode="floor") / dimension)
    angles = values.float().unsqueeze(-1) * (2 * math.pi) / frequencies
    return torch.stack((angles[..., 0::2].sin(), angles[..., 1::2].cos()), dim=-1).flatten(-2)


def video_position_embedding(mask, dimension, dtype):
    counts = mask.sum(dim=1, keepdim=True)
    positions = mask.to(torch.float32).cumsum(dim=1) / counts.clamp_min(1)
    return scalar_position_embedding(positions, dimension).to(dtype) * mask.unsqueeze(-1)


class FeatureFusion(nn.Module):

    def __init__(self, dimension, heads, ffn_dim, dropout):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(dimension, heads, dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dimension),
        )
        self.ffn_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        self.condition_encoder = nn.TransformerEncoderLayer(
            dimension, heads, ffn_dim, dropout, activation="relu", batch_first=True
        )

    def forward(self, video, text, video_mask, text_mask, gaussian_prior, similarity, positions):
        weighted_video = video * (gaussian_prior * similarity).unsqueeze(-1)
        (attended, _) = self.cross_attention(
            weighted_video, text, text, key_padding_mask=~text_mask, need_weights=False
        )
        fusion = self.cross_norm(weighted_video + self.dropout(attended))
        fusion = self.ffn_norm(fusion + self.dropout(self.ffn(fusion)))
        fusion = fusion * video_mask.unsqueeze(-1)
        condition = self.condition_encoder(fusion + positions, src_key_padding_mask=~video_mask)
        return condition * video_mask.unsqueeze(-1)
