"""Surrogates: transformers that map (context, query x) to bar-distribution logits over y.

Both take `forward(x_train [B,N,D], y_train [B,N], x_test [B,M,D], n_features [B], train_key_padding_mask)`
and return logits [B, M, n_bins]; both own their output distribution as `.bar_dist`.

- `pfn.PFN`: PFNs4BO-style, one token per row, attention between rows (frozen: the registered
  checkpoints are PFNs).
- `two_head_pfn.TwoHeadPFN`: the PFN with a latent and a noisy bar head (noisy priors; forward = latent).
- `tabpfn.TabPFN`: TabPFN-v2-style, one token per cell, feature and item attention, thinking rows.
"""

from .pfn import PFN
from .tabpfn import TabPFN
from .two_head_pfn import TwoHeadPFN

SURROGATES = {
    "pfn_repro.surrogates.pfn.PFN": PFN,
    "pfn_repro.surrogates.tabpfn.TabPFN": TabPFN,
    "pfn_repro.surrogates.two_head_pfn.TwoHeadPFN": TwoHeadPFN,
}

__all__ = ["PFN", "SURROGATES", "TabPFN", "TwoHeadPFN"]
