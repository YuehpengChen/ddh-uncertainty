"""Inference-only transcription of the evaluated research architecture.

No surrogate layers, retraining, quantization, or learned-parameter changes.
The inactive annotation encoder/optimizer are intentionally not loaded.
Import and execution are reserved for GitHub Actions, never the author's Mac.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
import timm


class FrozenBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = timm.create_model(
            cfg["guide_backbone"], pretrained=False, features_only=True,
            out_indices=[0, 1, 2, 3], img_size=cfg["img_size"],
        )
        self.channels = [self.model.feature_info[i]["num_chs"] for i in range(4)]

    def forward(self, image):
        return tuple(
            feature.contiguous() if feature.shape[1] == channel else
            feature.permute(0, 3, 1, 2).contiguous()
            for feature, channel in zip(self.model(image), self.channels)
        )


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, value):
        return self.net(value)


class MixerLayer(nn.Module):
    def __init__(self, hidden_dim, hidden_inter_dim, token_dim, token_inter_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.token_mlp = FeedForward(token_dim, token_inter_dim, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.channel_mlp = FeedForward(hidden_dim, hidden_inter_dim, dropout)

    def forward(self, value):
        value = value + self.token_mlp(self.ln1(value).transpose(1, 2)).transpose(1, 2)
        return value + self.channel_mlp(self.ln2(value))


class ResMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, hidden_dim, dropout)

    def forward(self, value):
        return value + self.ff(self.norm(value))


class FrozenDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer("embedding", torch.zeros(cfg["codebook"], cfg["token_dim"]))
        hidden = cfg["dec_hidden_dim"]
        middle = cfg["dec_hidden_inter_dim"]
        dropout = cfg["dropout"]
        self.decoder_token_mlp = nn.Linear(cfg["num_tokens"], 5)
        self.decoder_start = nn.Sequential(nn.Linear(cfg["token_dim"], hidden),
            nn.GELU(), nn.Dropout(dropout))
        self.decoder = nn.ModuleList([
            MixerLayer(hidden, middle, 5, cfg["dec_token_inter_dim"], dropout)
            for _ in range(cfg["dec_num_blocks"])
        ])
        self.decoder_tail = nn.Sequential(ResMLP(hidden, middle, dropout), nn.LayerNorm(hidden))
        self.head_mu = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.head_logvar = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, logits):
        value = logits.softmax(-1) @ self.embedding
        value = self.decoder_token_mlp(value.transpose(1, 2)).transpose(1, 2)
        value = self.decoder_start(value)
        for block in self.decoder:
            value = block(value)
        value = self.decoder_tail(value)
        return self.head_mu(value), self.head_logvar(value)


class MultiScaleFPN(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, width, 1, bias=False), nn.BatchNorm2d(width), nn.GELU())
            for c in channels
        ])

    def forward(self, features):
        fused = None
        size = features[0].shape[-2:]
        for projection, feature in zip(self.projs, features):
            value = projection(feature)
            if value.shape[-2:] != size:
                value = F.interpolate(value, size=size, mode="bilinear", align_corners=False)
            fused = value if fused is None else fused + value
        return fused


class Stage2Predictor(nn.Module):
    """Exact archived parameter names permit strict state_dict validation."""
    def __init__(self, cfg, dropout=0.3):
        super().__init__()
        width = cfg["conv_hidden"]
        self.fpn = MultiScaleFPN(cfg["in_channels_list"], width)
        self.modulator = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width),
            nn.GELU(), nn.Dropout2d(dropout),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, width, 7, 7))
        self.query_embed = nn.Parameter(torch.zeros(1, cfg["M"], width))
        layer = nn.TransformerDecoderLayer(d_model=width, nhead=cfg["num_heads"],
            dim_feedforward=width * 4, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=cfg["num_layers"])
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, cfg["N"]), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(cfg["N"], cfg["N"]))
        self.classifier = nn.Linear(cfg["N"], cfg["V"])

    def forward(self, features):
        value = self.fpn(features)
        value = self.modulator(value)
        value = value + F.interpolate(self.pos_embed, size=value.shape[-2:], mode="bilinear", align_corners=False)
        memory = value.flatten(2).transpose(1, 2).contiguous()
        query = self.query_embed.expand(value.shape[0], -1, -1).contiguous()
        value = self.transformer_decoder(tgt=query, memory=memory)
        return self.classifier(self.ffn(self.norm(value)))


def dropout_mask_specs(cfg, image_size=448):
    """One independent Bernoulli mask per archived dropout site.

    Each mask is binary uint8, with one=retained and P(one)=0.7. The graph
    applies inverse-keep-probability scaling. Spatial Dropout2d uses a mask
    with singleton spatial dimensions, exactly matching its shared-channel law.
    """
    width, count, heads = cfg["conv_hidden"], cfg["M"], cfg["num_heads"]
    memory = (image_size // 4) ** 2
    specs = [{"name": "mask_modulator", "shape": [1, width, 1, 1]}]
    for index in range(cfg["num_layers"]):
        prefix = f"mask_layer{index}_"
        specs.extend([
            {"name": prefix + "self_attention", "shape": [1, heads, count, count]},
            {"name": prefix + "self_residual", "shape": [1, count, width]},
            {"name": prefix + "cross_attention", "shape": [1, heads, count, memory]},
            {"name": prefix + "cross_residual", "shape": [1, count, width]},
            {"name": prefix + "feed_forward", "shape": [1, count, width * 4]},
            {"name": prefix + "feed_forward_residual", "shape": [1, count, width]},
        ])
    specs.append({"name": "mask_ffn", "shape": [1, count, cfg["N"]]})
    return [{**spec, "dtype": "uint8", "probability": 0.3,
             "keepProbability": 0.7} for spec in specs]


def dropout(value, probability, stochastic, mask):
    """Explicit binary mask input prevents ONNX optimizers removing MC noise."""
    one = torch.ones((), dtype=value.dtype, device=value.device)
    effective = torch.where(stochastic, mask.to(dtype=value.dtype), one)
    scale = torch.where(stochastic, one / (1.0 - probability), one)
    return value * effective * scale


def attention(module, query, key, value, stochastic, mask):
    """Standard scaled dot-product attention, retaining probability dropout.

    The archived PyTorch decoder uses no masks, no causal attention, and equal
    Q/K/V embedding sizes. Explicit elementary ops avoid export fast-path drift.
    """
    width = module.embed_dim
    heads = module.num_heads
    depth = width // heads
    weight_q, weight_k, weight_v = module.in_proj_weight.chunk(3, 0)
    bias_q, bias_k, bias_v = module.in_proj_bias.chunk(3, 0)
    q = F.linear(query, weight_q, bias_q).reshape(query.shape[0], -1, heads, depth).transpose(1, 2)
    k = F.linear(key, weight_k, bias_k).reshape(key.shape[0], -1, heads, depth).transpose(1, 2)
    v = F.linear(value, weight_v, bias_v).reshape(value.shape[0], -1, heads, depth).transpose(1, 2)
    weights = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(depth), dim=-1)
    weights = dropout(weights, module.dropout, stochastic, mask)
    output = (weights @ v).transpose(1, 2).reshape(query.shape[0], -1, width)
    return module.out_proj(output)


class PortableHead(nn.Module):
    def __init__(self, predictor, decoder, cfg):
        super().__init__()
        self.predictor = predictor
        self.decoder = decoder
        self.log_min = 2 * cfg["log_sigma_min"]
        self.log_max = 2 * cfg["log_sigma_max"]

    def forward(self, feature0, feature1, feature2, feature3, stochastic, *mask_inputs):
        masks = iter(mask_inputs)
        predictor = self.predictor
        value = predictor.fpn((feature0, feature1, feature2, feature3))
        for index, module in enumerate(predictor.modulator):
            value = dropout(value, module.p, stochastic, next(masks)) if index == 3 else module(value)
        value = value + F.interpolate(predictor.pos_embed, size=value.shape[-2:], mode="bilinear", align_corners=False)
        memory = value.flatten(2).transpose(1, 2).contiguous()
        query = predictor.query_embed.expand(value.shape[0], -1, -1).contiguous()
        for layer in predictor.transformer_decoder.layers:
            sa = attention(layer.self_attn, query, query, query, stochastic, next(masks))
            query = layer.norm1(query + dropout(sa, layer.dropout1.p, stochastic, next(masks)))
            ca = attention(layer.multihead_attn, query, memory, memory, stochastic, next(masks))
            query = layer.norm2(query + dropout(ca, layer.dropout2.p, stochastic, next(masks)))
            ff = layer.linear2(dropout(layer.activation(layer.linear1(query)), layer.dropout.p, stochastic, next(masks)))
            query = layer.norm3(query + dropout(ff, layer.dropout3.p, stochastic, next(masks)))
        value = predictor.norm(query)
        for module in predictor.ffn:
            value = dropout(value, module.p, stochastic, next(masks)) if isinstance(module, nn.Dropout) else module(value)
        mean, logvar = self.decoder(predictor.classifier(value))
        return mean, torch.exp(torch.clamp(logvar, self.log_min, self.log_max))
