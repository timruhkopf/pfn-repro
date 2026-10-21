"""The alpha-PFN checkpoints: training histories, the two stage gates, 1-d panels and agreement with standard
acquisitions (what pfn-acqpolicy's notebook 09 computed inline; read in `notebooks/04_*_alpha_pfn.ipynb`).

Gates (the rules stated before the original runs, pfn-acqpolicy `gates.py`):
  conditioned  at n = 20, every d: x, f and xf below none in the final validation; none within +0.02 nats
               of its first logged value (step 0, where the model equals its base PFN)
  alpha        final validation at n = 20, every d: r2 > 0 and rho >= 0.1
Tables:
  history_conditioned / history_alpha    validation curves from the checkpoints
  gates              (stage, d, criterion, value, threshold, pass)
  panel_lines        per 1-d function and context size: f, base PFN mean/band, conditioned (x*) mean,
                     alpha~ (target, from the true x*), alpha-PFN mean/band, EI, MES, UCB0.99, std
  panel_context / panel_optimum / panel_density / panel_borders   the rest of the 1-d panels
  agreement          (d, n, vs, spearman, same_argmax, argmax_distance) on fresh contexts
"""

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from ..acquisitions.alpha_pfn.conditioned import case_flags
from ..acquisitions.alpha_pfn.episodes import find_minimum
from ..acquisitions.alpha_pfn.targets import VARIANTS
from ..acquisitions.standard import mes_scores
from ..checkpoints import REGISTRY, load, make_prior
from ..tracking.sink import Run
from .common import to_dict
from .eval_pfn import history_frame, rebin


def gates(hc: pd.DataFrame, ha: pd.DataFrame, dims_c: list[int], dims_a: list[int], n: int = 20) -> pd.DataFrame:
    rows = []
    vc = hc.dropna(subset=[f"nll/val_d{dims_c[0]}_n{n}_none"])
    first, last = vc.iloc[0], vc.iloc[-1]
    for d in dims_c:
        none = last[f"nll/val_d{d}_n{n}_none"]
        for case in ("x", "f", "xf"):
            v = last[f"nll/val_d{d}_n{n}_{case}"]
            rows.append(
                dict(
                    stage="conditioned",
                    d=d,
                    criterion=f"{case} below none",
                    value=v - none,
                    threshold=0.0,
                    passed=bool(v < none),
                )
            )
        drift = none - first[f"nll/val_d{d}_n{n}_none"]
        rows.append(
            dict(
                stage="conditioned",
                d=d,
                criterion="none drift from step 0",
                value=drift,
                threshold=0.02,
                passed=bool(drift <= 0.02),
            )
        )
    va = ha.dropna(subset=[f"r2/val_d{dims_a[0]}_n{n}"]).iloc[-1]
    for d in dims_a:
        r2, rho = va[f"r2/val_d{d}_n{n}"], va[f"rho/val_d{d}_n{n}"]
        rows.append(dict(stage="alpha", d=d, criterion="r2 > 0", value=r2, threshold=0.0, passed=bool(r2 > 0)))
        rows.append(dict(stage="alpha", d=d, criterion="rho >= 0.1", value=rho, threshold=0.1, passed=bool(rho >= 0.1)))
    return pd.DataFrame(rows)


@torch.no_grad()
def _acquisitions(base, alpha, x_tr, y_tr, x_te) -> dict[str, torch.Tensor]:
    nf = torch.full((x_tr.shape[0],), x_tr.shape[-1], dtype=torch.long)
    lg = base(x_tr, y_tr, x_te, n_features=nf).float()
    bd = base.bar_dist
    inc = y_tr.min(1).values
    la = alpha(x_tr, y_tr, x_te, n_features=nf).float()
    return {
        "mean": bd.mean(lg),
        "std": bd.variance(lg).clamp(min=0).sqrt(),
        "ei": bd.ei(lg, inc[:, None]),
        "ucb0.99": -bd.ucb(lg, rest_prob=0.01),
        "mes": mes_scores(bd, lg, inc),
        "q05": bd.icdf(lg, 0.05),
        "q95": bd.icdf(lg, 0.95),
        "es": alpha.bar_dist.mean(la),
        "es_q05": alpha.bar_dist.icdf(la, 0.05),
        "es_q95": alpha.bar_dist.icdf(la, 0.95),
        "_lg": lg,
        "_la": la,
    }


@torch.no_grad()
def panels(base, cond, alpha, info: str, prior_kwargs: dict, p: DictConfig) -> dict[str, pd.DataFrame]:
    prior = make_prior(1, p.n_functions, seed=p.seed, **prior_kwargs)
    prior.reset()
    grid = torch.linspace(0, 1, p.n_grid)[None, :, None].expand(p.n_functions, -1, -1).contiguous()
    f = prior.evaluate(grid, noise=False)
    x_opt, f_opt = find_minimum(prior, extra_x=grid, seed=p.seed, refine_steps=0)
    keep_x = np.linspace(0, p.n_grid - 1, p.density_x).round().astype(int)
    xs = grid[0, :, 0].numpy()
    out: dict[str, list[pd.DataFrame]] = {k: [] for k in ("lines", "context", "density", "borders")}
    opt = pd.DataFrame({"function": np.arange(p.n_functions), "x_opt": x_opt[:, 0].numpy(), "f_opt": f_opt.numpy()})
    for n in p.contexts:
        g = torch.Generator().manual_seed(p.seed + n)
        x_tr = torch.rand(p.n_functions, n, 1, generator=g)
        y_tr = prior.evaluate(x_tr, noise=False)
        a = _acquisitions(base, alpha, x_tr, y_tr, grid)
        nf = torch.ones(p.n_functions, dtype=torch.long)
        lc = cond(x_tr, y_tr, grid, x_opt, f_opt, *case_flags(info, p.n_functions, x_tr.device), nf).float()
        l0 = cond(x_tr, y_tr, grid, x_opt, f_opt, *case_flags("none", p.n_functions, x_tr.device), nf).float()
        target = cond.bar_dist.entropy(l0) - cond.bar_dist.entropy(lc)  # alpha~ with the true optimum
        for b in range(p.n_functions):
            out["lines"].append(
                pd.DataFrame(
                    {
                        "function": b,
                        "n": n,
                        "x": xs,
                        "f_true": f[b].numpy(),
                        "pfn_mean": a["mean"][b].numpy(),
                        "pfn_q05": a["q05"][b].numpy(),
                        "pfn_q95": a["q95"][b].numpy(),
                        "cond_mean": cond.bar_dist.mean(lc)[b].numpy(),
                        "alpha_target": target[b].numpy(),
                        "alpha_mean": a["es"][b].numpy(),
                        "alpha_q05": a["es_q05"][b].numpy(),
                        "alpha_q95": a["es_q95"][b].numpy(),
                        **{k: a[k][b].numpy() for k in ("ei", "mes", "ucb0.99", "std")},
                    }
                )
            )
            out["context"].append(
                pd.DataFrame({"function": b, "n": n, "x": x_tr[b, :, 0].numpy(), "y": y_tr[b].numpy()})
            )
            for kind, logits, bd in (
                ("pfn", a["_lg"][b], base.bar_dist),
                ("conditioned", lc[b], cond.bar_dist),
                ("alpha", a["_la"][b], alpha.bar_dist),
            ):
                dens, nb = rebin(torch.softmax(logits, -1) / bd.bucket_widths, bd.borders, p.max_bins)
                dens = dens[keep_x].numpy()
                xi, bi = np.meshgrid(keep_x, np.arange(dens.shape[1]), indexing="ij")
                out["density"].append(
                    pd.DataFrame(
                        {
                            "kind": kind,
                            "function": b,
                            "n": n,
                            "x": xs[xi.ravel()],
                            "bin": bi.ravel(),
                            "density": dens.ravel().astype(np.float16),  # small: the snapshot is committed
                        }
                    )
                )
                if b == 0 and n == p.contexts[0]:
                    out["borders"].append(
                        pd.DataFrame(
                            {"kind": kind, "bin": np.arange(len(nb) - 1), "lo": nb[:-1].numpy(), "hi": nb[1:].numpy()}
                        )
                    )
    tables = {f"panel_{k}": pd.concat(v, ignore_index=True) for k, v in out.items()}
    tables["panel_optimum"] = opt
    return tables


def _spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ra, rb = a.argsort(-1).argsort(-1).float(), b.argsort(-1).argsort(-1).float()
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1)).clamp_min(1e-12)


@torch.no_grad()
def agreement(base, alpha, prior_kwargs: dict, a: DictConfig) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(a.seed)
    for d in a.dims:
        prior = make_prior(d, a.n_functions, seed=a.seed + d, **prior_kwargs)
        mean_dist = np.linalg.norm(rng.random((20000, d)) - rng.random((20000, d)), axis=1).mean()
        for mult in a.context_multiples:
            n = mult * d
            prior.reset()
            x_tr, y_tr, x_te, _ = prior.sample_episode(n, a.n_candidates)
            acq = _acquisitions(base, alpha, x_tr, y_tr, x_te)
            es = acq["es"]
            r = torch.arange(es.shape[0])
            for k in ("ei", "mes", "ucb0.99", "std", "mean"):
                v = acq[k] if k != "mean" else -acq["mean"]
                dist = (x_te[r, es.argmax(1)] - x_te[r, v.argmax(1)]).norm(dim=-1).mean().item()
                rows.append(
                    dict(
                        d=d,
                        n=n,
                        vs=k,
                        spearman=_spearman(es, v).mean().item(),
                        same_argmax=(es.argmax(1) == v.argmax(1)).float().mean().item(),
                        argmax_distance=dist / mean_dist,
                    )
                )
    return pd.DataFrame(rows)


def eval_alpha(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    torch.manual_seed(cfg.seed)
    if t.threads:
        torch.set_num_threads(t.threads)
    prior_kwargs = {k: v for k, v in to_dict(cfg.prior).items() if k != "name"}
    base, _ = load(t.base, "cpu")
    cond, cond_ckpt = load(t.conditioned, "cpu")
    alpha, alpha_ckpt = load(t.alpha, "cpu")
    info = VARIANTS[REGISTRY[t.alpha].source["variant"]]
    hc = history_frame(cond_ckpt, t.val_every)
    ha = history_frame(alpha_ckpt, t.val_every)
    run.log_table("history_conditioned", hc)
    run.log_table("history_alpha", ha)
    g = gates(hc, ha, list(t.gate_dims), list(t.gate_dims))
    run.log_table("gates", g)
    for name, df in panels(base, cond, alpha, info, prior_kwargs, t.panels).items():
        run.log_table(name, df)
    ag = agreement(base, alpha, prior_kwargs, t.agreement)
    run.log_table("agreement", ag)
    run.log_summary(
        **{"final/gates_passed": float(g.passed.all()), "final/agreement_ei": float(ag[ag.vs == "ei"].spearman.mean())}
    )
    return float(g.passed.mean())
