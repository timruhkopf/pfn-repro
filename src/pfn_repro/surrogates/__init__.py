"""Surrogates: transformers that map (context, query x) to bar-distribution logits over y.

Both take `forward(x_train [B,N,D], y_train [B,N], x_test [B,M,D], n_features [B], train_key_padding_mask)`
and return logits [B, M, n_bins]; both own their output distribution as `.bar_dist`.

- `pfn.PFN`: PFNs4BO-style, one token per row, attention between rows (frozen: the registered
  checkpoints are PFNs).
- `tabpfn.TabPFN`: TabPFN-v2-style, one token per cell, feature and item attention, thinking rows.
"""

from .pfn import PFN
from .tabpfn import TabPFN

SURROGATES = {"pfn_repro.surrogates.pfn.PFN": PFN, "pfn_repro.surrogates.tabpfn.TabPFN": TabPFN}

__all__ = ["PFN", "SURROGATES", "TabPFN"]
