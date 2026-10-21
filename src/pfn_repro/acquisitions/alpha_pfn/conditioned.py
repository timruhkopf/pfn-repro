# Ported from pfn-acqpolicy@980fad9 src/pfn_acqpolicy/models/conditioned_pfn.py, unchanged apart from import paths.
# FROZEN: the registered checkpoints depend on this code; add behaviour elsewhere (docs/decisions).
"""Stage 1 of alpha-PFN: a PFN that can additionally condition on the optimum,
q(y | x, D, I) with I in {none, x*, f*, (x*, f*)}.

As in the paper, I enters as one extra context token with its own encoder. It
wraps an unmodified `PFN` (models/pfn.py stays frozen) and reuses its blocks:
- token = opt_embed([x* * has_x, f* * has_f, has_x, has_f]); opt_embed starts as
  a copy of train_embed with zero weights on the two flags, so at init "both
  given" reads like an observation at (x*, f*);
- with I = none the token is masked out of every attention key, so that case is
  exactly the wrapped PFN. Initialized from a trained PFN, the model starts as
  that PFN and only has to learn what x*/f* add.
"""
import torch
import torch.nn as nn

from pfn_repro.surrogates.pfn import PFN, _pad_and_rescale_features

CASES = ("none", "x", "f", "xf")  # what I contains; x = PES, f = MES, xf = JES


def case_flags(case: str, B: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    has_x = torch.full((B,), "x" in case, dtype=torch.bool, device=device)
    has_f = torch.full((B,), "f" in case, dtype=torch.bool, device=device)
    return has_x, has_f


class OptimumConditionedPFN(nn.Module):
    def __init__(self, pfn: PFN):
        super().__init__()
        assert not pfn.causal
        self.pfn = pfn
        D, d_model = pfn.max_x_dim, pfn.train_embed.out_features
        self.opt_embed = nn.Linear(D + 3, d_model)
        with torch.no_grad():
            self.opt_embed.weight.zero_()
            self.opt_embed.weight[:, :D + 1] = pfn.train_embed.weight
            self.opt_embed.bias.copy_(pfn.train_embed.bias)

    @property
    def bar_dist(self):
        return self.pfn.bar_dist

    def forward(self, x_train, y_train, x_test, x_opt, f_opt, has_x, has_f, n_features):
        """x_opt [B, D], f_opt/has_x/has_f/n_features [B] -> logits [B, Nte, n_bins]."""
        p = self.pfn
        B, n = y_train.shape
        x_train = _pad_and_rescale_features(x_train, p.max_x_dim, n_features)
        x_test = _pad_and_rescale_features(x_test, p.max_x_dim, n_features)
        x_opt = _pad_and_rescale_features(x_opt[:, None], p.max_x_dim, n_features)[:, 0]
        hx, hf = has_x.to(x_opt.dtype), has_f.to(x_opt.dtype)
        opt_in = torch.cat([x_opt * hx[:, None], (f_opt * hf)[:, None], hx[:, None], hf[:, None]], -1)
        train_tok = torch.cat([p.train_embed(torch.cat([x_train, y_train[..., None]], -1)),
                               self.opt_embed(opt_in)[:, None]], 1)
        test_tok = p.test_x_embed(x_test) + p.test_placeholder.view(1, 1, -1)
        keep = torch.cat([torch.ones(B, n, dtype=torch.bool, device=y_train.device), (has_x | has_f)[:, None]], 1)
        for block in p.blocks:
            train_tok, test_tok = block(train_tok, test_tok, keep, None)
        return p.out_head(p.out_ln(test_tok))
