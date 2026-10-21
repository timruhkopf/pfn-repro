"""A compact TabPFN-v2-style surrogate (own implementation, after Hollmann et al. 2025, Nature; TabPFN v2.5's
thinking rows). Same call signature and output as `surrogates.pfn.PFN`, so trainers, acquisitions and
evaluation work with either.

The difference to the PFNs4BO-style `PFN` (one token per row, attention between rows only):
- **every cell is a token**: a row with D features becomes D + 1 tokens (the features and the target);
- each block alternates **feature attention** (the cells of one row attend to each other) and **item
  attention** (the cells of one column attend across rows: context rows to context rows, test rows to
  context rows only, never to each other), then an MLP on every cell;
- **thinking rows**: `n_thinking_rows` learned rows prepended to the context. They take part in item
  attention like observed rows and give the model scratch space independent of the context size;
- the test rows' target cell holds a learned "unknown" embedding; its final state is decoded into
  bar-distribution logits.

Deliberate simplifications against TabPFN v2: a learned embedding per feature index instead of
random per-forward feature embeddings (in BO a column keeps its meaning, so symmetry between features
need not be learned away), no feature grouping, no preprocessing ensemble. Padded features (index
>= n_features) are masked as keys of feature attention, so they never reach the real columns.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..objectives.bar_distribution import BarDistribution, uniform_bin_borders


class _Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self, q_in: torch.Tensor, kv_in: torch.Tensor | None = None, key_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """q_in [S, Tq, d], kv_in [S, Tk, d] (default q_in), key_mask [S, Tk] bool (True = attend)."""
        kv_in = q_in if kv_in is None else kv_in
        S, Tq, d = q_in.shape
        Tk = kv_in.shape[1]
        q = F.linear(q_in, self.qkv.weight[:d], self.qkv.bias[:d])  # one projection, also for cross-attention
        k, v = F.linear(kv_in, self.qkv.weight[d:], self.qkv.bias[d:]).chunk(2, -1)
        q, k, v = (t.view(S, -1, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))
        mask = None if key_mask is None else key_mask.view(S, 1, 1, Tk)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(out.transpose(1, 2).reshape(S, Tq, d))


class TabBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln_feat, self.feat_attn = nn.LayerNorm(d_model), _Attention(d_model, n_heads)
        self.ln_item, self.item_attn = nn.LayerNorm(d_model), _Attention(d_model, n_heads)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(
        self, ctx: torch.Tensor, test: torch.Tensor, col_mask: torch.Tensor, row_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ctx [B, R, C, d] (thinking + context rows), test [B, M, C, d], col_mask [B, C] (real columns),
        row_mask [B, R] (real context rows) or None."""
        B, R, C, d = ctx.shape
        M = test.shape[1]
        # 1. feature attention: within each row, over its cells (padded feature columns are not keys)
        cells = torch.cat([ctx, test], 1).reshape(B * (R + M), C, d)
        keys = col_mask[:, None].expand(B, R + M, C).reshape(B * (R + M), C)
        cells = cells + self.feat_attn(self.ln_feat(cells), key_mask=keys)
        cells = cells.view(B, R + M, C, d)
        ctx, test = cells[:, :R], cells[:, R:]
        # 2. item attention: within each column, over rows; test rows read the context only
        ctx_c = ctx.transpose(1, 2).reshape(B * C, R, d)
        test_c = test.transpose(1, 2).reshape(B * C, M, d)
        rows = None if row_mask is None else row_mask[:, None].expand(B, C, R).reshape(B * C, R)
        ctx_n = self.ln_item(ctx_c)
        ctx_c = ctx_c + self.item_attn(ctx_n, key_mask=rows)
        test_c = test_c + self.item_attn(self.ln_item(test_c), kv_in=ctx_n, key_mask=rows)
        ctx = ctx_c.view(B, C, R, d).transpose(1, 2)
        test = test_c.view(B, C, M, d).transpose(1, 2)
        # 3. MLP on every cell
        ctx = ctx + self.ff(self.ln_ff(ctx))
        test = test + self.ff(self.ln_ff(test))
        return ctx, test


class TabPFN(nn.Module):
    def __init__(
        self,
        max_x_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 256,
        n_bins: int = 1000,
        n_thinking_rows: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_x_dim = max_x_dim
        self.n_thinking_rows = n_thinking_rows
        self.x_embed = nn.Linear(1, d_model)
        self.y_embed = nn.Linear(1, d_model)
        self.column_embed = nn.Parameter(torch.randn(max_x_dim + 1, d_model) * 0.02)  # features..., target
        self.y_unknown = nn.Parameter(torch.zeros(d_model))
        self.thinking = nn.Parameter(torch.randn(n_thinking_rows, max_x_dim + 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([TabBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, n_bins)
        self.bar_dist = BarDistribution(uniform_bin_borders(n_bins))

    def _cells(self, x: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        """x [B, N, max_x_dim], y [B, N] or None -> [B, N, max_x_dim + 1, d_model]."""
        feats = self.x_embed(x[..., None])
        target = self.y_embed(y[..., None, None]) if y is not None else self.y_unknown.expand(*x.shape[:2], 1, -1)
        return torch.cat([feats, target], 2) + self.column_embed

    def forward(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        n_features: torch.Tensor | None = None,
        train_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x_train [B, N, D], y_train [B, N], x_test [B, M, D] with D <= max_x_dim; n_features [B] real
        feature count (default D); train_key_padding_mask [B, N] (True = real row) -> logits [B, M, n_bins]."""
        B, _, D = x_train.shape
        if n_features is None:
            n_features = x_train.new_full((B,), D, dtype=torch.long)
        pad = self.max_x_dim - D
        if pad:
            x_train, x_test = (F.pad(t, (0, pad)) for t in (x_train, x_test))
        col = torch.arange(self.max_x_dim + 1, device=x_train.device)
        col_mask = (col[None] < n_features[:, None]) | (col[None] == self.max_x_dim)  # real features + target
        feature_mask = col_mask[:, None, :-1].to(x_train.dtype)
        ctx = self._cells(x_train * feature_mask, y_train)
        test = self._cells(x_test * feature_mask, None)
        row_mask = train_key_padding_mask
        if self.n_thinking_rows:
            ctx = torch.cat([self.thinking.expand(B, -1, -1, -1), ctx], 1)
            if row_mask is not None:
                row_mask = torch.cat([row_mask.new_ones(B, self.n_thinking_rows), row_mask], 1)
        for block in self.blocks:
            ctx, test = block(ctx, test, col_mask, row_mask)
        return self.out_head(self.out_ln(test[:, :, -1]))
