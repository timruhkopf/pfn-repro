import torch

from pfn_repro.acquisitions.alpha_pfn.conditioned import OptimumConditionedPFN, case_flags
from pfn_repro.acquisitions.standard import arm_scores
from pfn_repro.checkpoints import load


def _context():
    g = torch.Generator().manual_seed(0)
    return torch.rand(3, 6, 2, generator=g), torch.rand(3, 6, generator=g), torch.rand(3, 40, 2, generator=g)


def test_standard_acquisitions_are_finite_and_shaped() -> None:
    pfn, _ = load("pfn_smoke")
    x_tr, y_tr, x_te = _context()
    logits = pfn(x_tr, y_tr, x_te, n_features=torch.full((3,), 2))
    inc = y_tr.min(1).values
    for arm in ("pi", "ei", "ucb", "mean", "std", "mes", "ei0.01", "ucb0.99"):
        s = arm_scores(arm, pfn.bar_dist, logits, inc, ucb_quantile=0.95)
        assert s.shape == (3, 40) and bool(torch.isfinite(s).all()), arm
    assert bool((arm_scores("ei", pfn.bar_dist, logits, inc, 0.95) >= 0).all())


def test_conditioned_pfn_without_information_is_its_base() -> None:
    """With I = none the optimum token is masked out: at initialization the model equals the wrapped PFN."""
    pfn, _ = load("pfn_smoke")
    cond = OptimumConditionedPFN(pfn).eval()
    x_tr, y_tr, x_te = _context()
    nf = torch.full((3,), 2)
    x_opt, f_opt = torch.rand(3, 2), torch.rand(3)
    out = cond(x_tr, y_tr, x_te, x_opt, f_opt, *case_flags("none", 3, "cpu"), nf)
    torch.testing.assert_close(out, pfn(x_tr, y_tr, x_te, n_features=nf), atol=1e-5, rtol=1e-5)
