# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# See licenses/MomentDiff_LICENSE.
import copy
import math
from typing import Optional
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .attention import MultiheadAttention


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            (nn.Linear(n, k) for (n, k) in zip([input_dim] + h, h + [output_dim]))
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x, eps=0.001):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_sineembed_for_position(pos_tensor, dimension=256):
    scale = 2 * math.pi
    dim_t = torch.arange(dimension // 2, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / (dimension // 2))
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        frequencies = torch.exp(torch.arange(half_dim) * (-math.log(10000) / (half_dim - 1)))
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = self.frequencies.to(time.device)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffDecoder(nn.Module):

    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=2,
        keep_query_pos=False,
        query_scale_type="cond_elewise",
        modulate_t_attn=False,
        bbox_embed_diff_each_layer=False,
        time_embed_scale=0.1,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim
        assert query_scale_type in ["cond_elewise", "cond_scalar", "fix_elewise"]
        self.query_scale_type = query_scale_type
        if query_scale_type == "cond_elewise":
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == "cond_scalar":
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == "fix_elewise":
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList(
                [MLP(d_model, d_model, 2, 3) for i in range(num_layers)]
            )
        else:
            self.bbox_embed = MLP(d_model, d_model, 2, 3)
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        self.d_model = d_model
        self.modulate_t_attn = modulate_t_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        if modulate_t_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 1, 2)
        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None
        self.ref_point_head = MLP(d_model, d_model, d_model, 1)
        self.time_mlp = SinusoidalPositionEmbeddings(d_model)
        self.time_embed_scale = float(time_embed_scale)
        self.noise_embed = MLP(2, d_model, d_model, 1)

    def forward(
        self,
        tgt,
        span,
        memory,
        txt_src,
        time,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        time_emb = self.time_mlp(time)
        span = inverse_sigmoid(span).sigmoid()
        output = self.noise_embed(
            span.to(torch.float32)
        ) + self.time_embed_scale * time_emb.unsqueeze(1).repeat(1, span.shape[1], 1)
        output = output.permute(1, 0, 2)
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            obj_center = span[..., : self.query_dim].permute(1, 0, 2).to(torch.float32)
            query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model)
            query_pos = self.ref_point_head(query_sine_embed)
            if self.query_scale_type != "fix_elewise":
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]
            query_sine_embed = query_sine_embed * pos_transformation
            output = layer(
                output,
                memory,
                time_emb,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                query_sine_embed=query_sine_embed,
                is_first=layer_id == 0,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            if self.bbox_embed is not None:
                return torch.stack(intermediate).transpose(1, 2)
        return output.unsqueeze(0)


class DiffDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        keep_query_pos=False,
        rm_self_attn_decoder=False,
    ):
        super().__init__()
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = MLP(d_model, d_model, d_model, 3)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        time_emb,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_sine_embed=None,
        is_first=False,
    ):
        (nr_boxes, N) = (tgt.shape[0], tgt.shape[1])
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)
        (num_queries, bs, n_model) = q_content.shape
        (hw, _, _) = k_content.shape
        k_pos = self.ca_kpos_proj(pos)
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content
        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)
        tgt2 = self.cross_attn(
            query=q, key=k, value=v, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class ConditionalDenoiser(nn.Module):

    def __init__(
        self,
        dimension=256,
        heads=8,
        ffn_dim=2048,
        dropout=0.1,
        layers=2,
        time_embed_scale=0.1,
        confidence_mode="best_iou",
        noise_loss_mode="auxiliary",
        auxiliary_loss=False,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("At least one diffusion decoder layer is required")
        layer = DiffDecoderLayer(
            dimension, heads, ffn_dim, dropout, activation="prelu", rm_self_attn_decoder=True
        )
        self.decoder = DiffDecoder(
            layer,
            layers,
            nn.LayerNorm(dimension),
            return_intermediate=True,
            d_model=dimension,
            time_embed_scale=time_embed_scale,
        )
        self.span_head = MLP(dimension, dimension, 2, 1)
        self.noise_head = nn.Linear(dimension, 2) if noise_loss_mode == "auxiliary" else None
        self.confidence_head = nn.Linear(dimension, 1 if confidence_mode == "best_iou" else 2)
        self.auxiliary_loss = bool(auxiliary_loss)

    def forward(self, spans, steps, condition, frame_positions, video_mask):
        (batch, proposals, _) = spans.shape
        tgt = condition.new_zeros(proposals, batch, condition.shape[-1])
        all_hidden = self.decoder(
            tgt,
            spans,
            condition.transpose(0, 1),
            None,
            steps,
            memory_key_padding_mask=~video_mask,
            pos=frame_positions.transpose(0, 1),
        )
        hidden = all_hidden[-1]
        logits = self.confidence_head(hidden)
        result = {
            "pred_cxw": self.span_head(hidden).sigmoid(),
            "pred_noise_aux": self.noise_head(hidden) if self.noise_head is not None else None,
            "confidence_logits": logits.squeeze(-1) if logits.shape[-1] == 1 else logits,
            "hidden": hidden,
        }
        if self.auxiliary_loss and self.training:
            result["aux_outputs"] = []
            for intermediate in all_hidden[:-1]:
                aux_logits = self.confidence_head(intermediate)
                result["aux_outputs"].append(
                    {
                        "pred_cxw": self.span_head(intermediate).sigmoid(),
                        "confidence_logits": (
                            aux_logits.squeeze(-1) if aux_logits.shape[-1] == 1 else aux_logits
                        ),
                    }
                )
        return result
