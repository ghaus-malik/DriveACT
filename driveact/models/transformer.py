"""DETR-style Transformer encoder / decoder as used by ACT (Zhao et al., 2023).

Written for this project following the structure of the reference implementation (``detr/models/transformer.py``
in github.com/tonyzhaozh/act, which is derived from DETR): positional embeddings are added to the queries and keys
of every attention layer (not to the values), layers are post-norm by default, the decoder starts from zero
targets and its learned queries enter as ``query_pos``.

Difference to the reference code: the reference decoder stacks the outputs of all layers and the policy then reads
index 0, i.e. only the *first* decoder layer influences the actions (documented in LeRobot's ACT configuration and
in tonyzhaozh/act issue #25). Here the output of the *last* decoder layer is used, so ``dec_layers`` is effective.
``dec_layers: 1`` reproduces the effective depth of the reference model.

Tensors are batch-first: (B, L, D).
"""

from __future__ import annotations

import torch
from torch import nn


def sinusoid_table(n_position: int, d_hid: int) -> torch.Tensor:
    """Fixed sine/cosine table (1, n_position, d_hid) (the CVAE encoder's positional encoding in ACT)."""
    pos = torch.arange(n_position, dtype=torch.float64)[:, None]
    i = torch.arange(d_hid, dtype=torch.float64)[None, :]
    angle = pos / torch.pow(10000.0, 2 * torch.div(i, 2, rounding_mode="floor") / d_hid)
    table = torch.zeros(n_position, d_hid, dtype=torch.float64)
    table[:, 0::2] = torch.sin(angle[:, 0::2])
    table[:, 1::2] = torch.cos(angle[:, 1::2])
    return table.float()[None]


def _with_pos(x: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
    return x if pos is None else x + pos


class EncoderLayer(nn.Module):
    def __init__(self, d: int, nheads: int, dim_ff: int, dropout: float, pre_norm: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nheads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, dim_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_ff, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, x, pos=None, key_padding_mask=None):
        if self.pre_norm:
            h = self.norm1(x)
            q = k = _with_pos(h, pos)
            x = x + self.drop1(self.attn(q, k, h, key_padding_mask=key_padding_mask, need_weights=False)[0])
            return x + self.drop2(self.ff(self.norm2(x)))
        q = k = _with_pos(x, pos)
        x = self.norm1(x + self.drop1(self.attn(q, k, x, key_padding_mask=key_padding_mask, need_weights=False)[0]))
        return self.norm2(x + self.drop2(self.ff(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d: int, nheads: int, dim_ff: int, dropout: float, pre_norm: bool = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nheads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, nheads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, dim_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_ff, d))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop1, self.drop2, self.drop3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self, tgt, memory, pos=None, query_pos=None, memory_key_padding_mask=None):
        if self.pre_norm:
            h = self.norm1(tgt)
            q = k = _with_pos(h, query_pos)
            tgt = tgt + self.drop1(self.self_attn(q, k, h, need_weights=False)[0])
            h = self.norm2(tgt)
            tgt = tgt + self.drop2(self.cross_attn(_with_pos(h, query_pos), _with_pos(memory, pos), memory,
                                                   key_padding_mask=memory_key_padding_mask, need_weights=False)[0])
            return tgt + self.drop3(self.ff(self.norm3(tgt)))
        q = k = _with_pos(tgt, query_pos)
        tgt = self.norm1(tgt + self.drop1(self.self_attn(q, k, tgt, need_weights=False)[0]))
        tgt = self.norm2(tgt + self.drop2(self.cross_attn(_with_pos(tgt, query_pos), _with_pos(memory, pos), memory,
                                                          key_padding_mask=memory_key_padding_mask, need_weights=False)[0]))
        return self.norm3(tgt + self.drop3(self.ff(tgt)))


class Encoder(nn.Module):
    def __init__(self, d, nheads, dim_ff, num_layers, dropout, pre_norm=False):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(d, nheads, dim_ff, dropout, pre_norm) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d) if pre_norm else nn.Identity()

    def forward(self, x, pos=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, pos, key_padding_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d, nheads, dim_ff, num_layers, dropout, pre_norm=False):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(d, nheads, dim_ff, dropout, pre_norm) for _ in range(num_layers))
        self.norm = nn.LayerNorm(d)  # as in DETR / ACT, a final norm is applied to the decoder output

    def forward(self, tgt, memory, pos=None, query_pos=None, memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, pos, query_pos, memory_key_padding_mask)
        return self.norm(tgt)


class Transformer(nn.Module):
    """Encoder over the conditioning tokens + decoder from learned action queries."""

    def __init__(self, d=256, nheads=8, dim_ff=1024, enc_layers=4, dec_layers=4, dropout=0.1, pre_norm=False):
        super().__init__()
        self.encoder = Encoder(d, nheads, dim_ff, enc_layers, dropout, pre_norm)
        self.decoder = Decoder(d, nheads, dim_ff, dec_layers, dropout, pre_norm)
        self.d = d
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos, query_embed):
        """src, pos (B, L, D); query_embed (Q, D) -> decoder output (B, Q, D)."""
        memory = self.encoder(src, pos)
        query_pos = query_embed[None].expand(src.shape[0], -1, -1)
        tgt = torch.zeros_like(query_pos)
        return self.decoder(tgt, memory, pos=pos, query_pos=query_pos)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

