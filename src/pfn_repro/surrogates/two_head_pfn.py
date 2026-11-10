"""Two-headed PFN: a latent bar head p(f(x) | D) and a separately trained noisy bar head p(y | x, D).

Issue #3; the decoupled-head idea follows Decoupled PFNs (Bergna, Depeweg & Hernandez-Lobato 2026,
arXiv:2605.06413), which trains a latent head on the clean signal and a noise head on the noise
variance, with the generator's privileged targets. One difference: their observation predictive convolves
the latent distribution with a plug-in noise variance; here the observation predictive is a *separately
trained* bar head, because (a) the exact predictive is a mixture over the unknown noise level, not a
plug-in convolution, and (b) the noise families of `priors/noisy_bnn.py` (Student-t, contamination,
skew) are not Gaussian. Var_noisy - Var_latent is then an *estimate* of E[sigma^2 | D] rather than a
constraint, and the gap between the two heads is a free consistency diagnostic.

Same backbone, attention pattern and call signature as the frozen `PFN` (subclassed, nothing in it
changed): `forward(...)` returns the **latent** logits with `.bar_dist` (borders [0, 1]), so existing
acquisition code uses the latent head unchanged, which is the right default for acquisition and for the
recommendation (latent incumbent; no winner's curse). `forward_both(...)` returns (latent, noisy);
`.noisy_bar_dist` covers [noisy_lo, noisy_hi]. Context y is clipped to that range before embedding.
"""

from typing import cast

import torch
import torch.nn as nn

from ..objectives.bar_distribution import BarDistribution, uniform_bin_borders
from .pfn import PFN, MaskedMHA, PFNBlock, _pad_and_rescale_features


class TwoHeadPFN(PFN):
    def __init__(
        self,
        max_x_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        n_bins: int = 64,
        noisy_n_bins: int | None = None,
        noisy_lo: float = -1.0,
        noisy_hi: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__(max_x_dim, d_model, n_heads, n_layers, d_ff, n_bins, dropout, causal=False)
        self.noisy_lo, self.noisy_hi = noisy_lo, noisy_hi
        # same bin width as the latent head by default
        noisy_n_bins = noisy_n_bins or round(n_bins * (noisy_hi - noisy_lo))
        self.noisy_ln = nn.LayerNorm(d_model)
        self.noisy_head = nn.Linear(d_model, noisy_n_bins)
        self.noisy_bar_dist = BarDistribution(uniform_bin_borders(noisy_n_bins, noisy_lo, noisy_hi))

    def _clip_y(self, y: torch.Tensor) -> torch.Tensor:
        return y.clamp(self.noisy_lo, self.noisy_hi)

    def _test_tokens(self, x_train, y_train, x_test, n_features, train_key_padding_mask) -> torch.Tensor:
        """The frozen PFN's forward up to the final test tokens (same blocks, same masks)."""
        if n_features is None:
            n_features = x_train.new_full((x_train.shape[0],), x_train.shape[-1])
        x_train = _pad_and_rescale_features(x_train, self.max_x_dim, n_features)
        x_test = _pad_and_rescale_features(x_test, self.max_x_dim, n_features)
        train_tok = self.train_embed(torch.cat([x_train, self._clip_y(y_train).unsqueeze(-1)], dim=-1))
        test_tok = self.test_x_embed(x_test) + self.test_placeholder.view(1, 1, -1)
        for block in self.blocks:
            train_tok, test_tok = block(train_tok, test_tok, train_key_padding_mask, None)
        return test_tok

    def forward_both(self, x_train, y_train, x_test, n_features=None, train_key_padding_mask=None):
        """-> (latent logits [B, M, n_bins], noisy logits [B, M, noisy_n_bins])."""
        tok = self._test_tokens(x_train, y_train, x_test, n_features, train_key_padding_mask)
        return self.out_head(self.out_ln(tok)), self.noisy_head(self.noisy_ln(tok))

    def forward(self, x_train, y_train, x_test, n_features=None, train_key_padding_mask=None, return_hidden=False):
        """Latent logits only: the drop-in surrogate interface (see module docstring)."""
        assert not return_hidden, "TwoHeadPFN does not export hidden states; use context_kv"
        return self.forward_both(x_train, y_train, x_test, n_features, train_key_padding_mask)[0]

    def context_kv_with_grad(self, x_train, y_train, n_features, train_key_padding_mask=None):
        return super().context_kv_with_grad(x_train, self._clip_y(y_train), n_features, train_key_padding_mask)

    def heads_from_context_kv(self, kv, x_test, n_features, train_key_padding_mask=None):
        """Both heads from a cached context (`context_kv`): one encoding, many query batches."""
        x = _pad_and_rescale_features(x_test, self.max_x_dim, n_features)
        tok = self.test_x_embed(x) + self.test_placeholder.view(1, 1, -1)
        B, Q, Dm = tok.shape
        for block, (K, V) in zip(self.blocks, kv, strict=True):
            blk = cast(PFNBlock, block)
            a = cast(MaskedMHA, blk.attn)
            N = K.shape[1]
            q = a.q_proj(blk.ln1(tok)).view(B, Q, a.h, a.dk).transpose(1, 2)
            k = K.view(B, N, a.h, a.dk).transpose(1, 2)
            v = V.view(B, N, a.h, a.dk).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (a.dk**0.5)
            if train_key_padding_mask is not None:
                scores = scores.masked_fill(~train_key_padding_mask.view(B, 1, 1, N), torch.finfo(scores.dtype).min)
            att = torch.nan_to_num(scores.softmax(dim=-1), nan=0.0)
            tok = tok + a.out_proj((att @ v).transpose(1, 2).reshape(B, Q, Dm))
            tok = tok + blk.ff(blk.ln2(tok))
        return self.out_head(self.out_ln(tok)), self.noisy_head(self.noisy_ln(tok))


if __name__ == "__main__":
    from ..priors.noisy_bnn import NoisyBNNPrior

    torch.manual_seed(0)
    prior = NoisyBNNPrior(batch_size=4, x_dim=2, seed=0)
    e = prior.sample_noisy_episode(20, 32)
    model = TwoHeadPFN(max_x_dim=2, d_model=32, n_heads=4, n_layers=2, d_ff=64, n_bins=32)
    lat, noi = model.forward_both(e.x_train, e.y_train, e.x_test, prior.active_dim)
    print("latent", tuple(lat.shape), "noisy", tuple(noi.shape))
    print("forward == latent head:", torch.allclose(model(e.x_train, e.y_train, e.x_test, prior.active_dim), lat))
    kv = model.context_kv(e.x_train, e.y_train, prior.active_dim)
    lat2, noi2 = model.heads_from_context_kv(kv, e.x_test, prior.active_dim)
    print("cached path max diff:", (lat - lat2).abs().max().item(), (noi - noi2).abs().max().item())
    perm = torch.randperm(20)
    lat3 = model(e.x_train[:, perm], e.y_train[:, perm], e.x_test, prior.active_dim)
    print("permutation max diff:", (lat - lat3).abs().max().item())
    print(
        "untrained NLL latent / noisy:",
        model.bar_dist(lat, e.f_test).mean().item(),
        model.noisy_bar_dist(noi, e.y_test).mean().item(),
    )
