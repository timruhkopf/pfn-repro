# Ported from pfn-acqpolicy@980fad9 src/pfn_acqpolicy/es/targets.py, unchanged apart from import paths.
"""Stage-2 targets and the alpha-PFN's output support.

alpha-PFN's target (their Eq. for alpha~) is the entropy reduction from one draw of
the optimum information I, under the stage-1 model:
    alpha~(D, x, I) = H[q(y | x, D)] - H[q(y | x, D, I)]
For a function drawn from the prior, its x* is a draw from p(x* | D). So the
mean of alpha~ given (D, x) is the mutual information I(y; I | x, D): PES for
I = x* (which equals the original Entropy Search by symmetry of MI), MES for
I = f*, JES for both. The alpha-PFN predicts a bar distribution over alpha~;
its mean is the acquisition value. A single alpha~ can be negative.
"""
import torch

from pfn_repro.acquisitions.alpha_pfn.episodes import Episode
from pfn_repro.acquisitions.alpha_pfn.conditioned import OptimumConditionedPFN, case_flags

VARIANTS = {"pes": "x", "es": "x", "mes": "f", "jes": "xf"}


@torch.no_grad()
def info_gain(cpfn: OptimumConditionedPFN, ep: Episode, info: str) -> torch.Tensor:
    """-> alpha~ [B, n_test] in nats, from two fp32 passes of the stage-1 model."""
    B, dev = ep.y_tr.shape[0], ep.y_tr.device
    args = (ep.x_tr, ep.y_tr, ep.x_te, ep.x_opt, ep.f_opt)
    h0 = cpfn.bar_dist.entropy(cpfn(*args, *case_flags("none", B, dev), ep.n_features))
    h1 = cpfn.bar_dist.entropy(cpfn(*args, *case_flags(info, B, dev), ep.n_features))
    return h0 - h1


def fit_borders(samples: torch.Tensor, n_bins: int, margin: float = 0.1) -> torch.Tensor:
    """Quantile borders (equal mass per bin on `samples`, as alpha-PFN fits its borders
    from the corpus), with a minimum bin width so point masses (e.g. alpha~ = 0 far from
    the data) can't produce zero-width bins, and outer borders widened by `margin`
    of the range for values beyond the calibration sample."""
    s = samples.flatten().float()
    q = torch.quantile(s[torch.randperm(len(s))[:2 ** 24]], torch.linspace(0, 1, n_bins + 1, device=s.device))
    span = float(q[-1] - q[0]) or 1.0
    q[0], q[-1] = q[0] - margin * span, q[-1] + margin * span
    min_w = 1e-4 * span
    b = q.clone()
    for i in range(1, len(b)):
        b[i] = torch.maximum(b[i], b[i - 1] + min_w)
    return b.cpu()
