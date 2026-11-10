"""Stage-0 checks of a two-headed PFN on the noisy prior (issue #3; notebook `*_noisy_pfn`).

Tables (all on fresh functions from `data_seed`, disjoint from training and its validation):

- `calibration`: per function x context size: NLL of f under the latent head and of y under the noisy
  head, central-interval coverage (50 / 90 %) of each head for its own target, squared error of the
  latent mean. Control: the *noisy* head used as a predictive for f (an observation-level PFN's view of
  f) — the latent head should beat it, increasingly with noise.
- `pit_hist`: PIT histograms per head, regime and context size (uniform iff calibrated).
- `total_variance`: Var_noisy - Var_latent against the true noise variance at the test points, for the
  families with finite variance and no contamination (law of total variance, doc section 4.2).
- `martingale`: the self-test of doc section 3.1: sample y ~ noisy head at x, re-encode D + (x, y),
  average the new latent means at probe points x', compare with the current latent mean.
- `identifiability` / `probe_features`: can the noise regime, heteroscedasticity, family and SNR be read
  from one encoding as the context grows? Interpretable features from both heads (doc section 7.4,
  group E), linear probes (logistic / ridge) fitted on half the functions, scored on the other half,
  with the majority-class / mean predictor as the null control.
- `slices` / `slice_context`: 1-d cuts through a few functions for the figures.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from ..checkpoints import load_pfn_file, sha256
from ..evaluation.predictive import pit
from ..priors.noisy_bnn import ASSETS, FAMILIES, REGIMES, NoisyBNNPrior
from ..tracking.sink import Run
from ..utils.seeding import seed_everything
from .common import device_of

# ---------------------------------------------------------------- bar-distribution helpers


def central_interval(bd, logits: torch.Tensor, level: float) -> tuple[torch.Tensor, torch.Tensor]:
    return bd.icdf(logits, (1 - level) / 2), bd.icdf(logits, (1 + level) / 2)


def bar_moments(bd, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """mean, variance (exact for the piecewise-uniform density), skewness and kurtosis (bin centres)."""
    p = torch.softmax(logits.float(), -1)
    c = bd.borders[:-1] + bd.bucket_widths / 2
    mean = p @ c
    var = bd.variance(logits.float()).clamp(min=1e-12)
    z = (c - mean.unsqueeze(-1)) / var.sqrt().unsqueeze(-1)
    return mean, var, (p * z**3).sum(-1), (p * z**4).sum(-1)


# ---------------------------------------------------------------- sections


def _prior(prior_cfg: dict, batch: int, seed: int, dev: str, ecdf: torch.Tensor) -> NoisyBNNPrior:
    kw = {k: v for k, v in prior_cfg.items() if k not in ("name", "x_dim")}
    return NoisyBNNPrior(batch_size=batch, x_dim=prior_cfg["x_dim"], seed=seed, device=dev, ecdf_sorted=ecdf, **kw)


def _props_frame(prior: NoisyBNNPrior, offset: int) -> pd.DataFrame:
    p = {k: v.detach().cpu().numpy() for k, v in prior.properties().items()}
    df = pd.DataFrame(p)
    df["regime"] = [REGIMES[i] for i in p["regime"]]
    df["family"] = [FAMILIES[i] for i in p["family"]]
    df["bo_context"] = prior.episode_info["bo_context"].cpu().numpy()
    df["replicate_frac"] = prior.episode_info["replicate_frac"].cpu().numpy()
    df["fn"] = np.arange(len(df)) + offset
    return df


@torch.no_grad()
def calibration(model, prior_cfg, ecdf, t, dev, seed) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, pits, tv = [], [], []
    n_max = max(t.context_sizes)
    bins = np.linspace(0, 1, t.pit_bins + 1)
    for r_i, regime in enumerate(REGIMES):
        for c in range(t.n_functions // t.chunk):
            prior = _prior(prior_cfg, t.chunk, seed + 1000 * r_i + c, dev, ecdf)
            prior.reset(regime=regime)
            e = prior.sample_noisy_episode(n_max, t.n_test)
            props = _props_frame(prior, offset=(r_i * 10_000 + c * t.chunk))
            var_true = prior.noise_var(e.x_test, e.f_test)
            for n in t.context_sizes:
                lat, noi = model.forward_both(e.x_train[:, :n], e.y_train[:, :n], e.x_test, prior.active_dim)
                lat, noi = lat.float(), noi.float()
                bd, nbd = model.bar_dist, model.noisy_bar_dist
                out = {
                    "nll_latent": bd(lat, e.f_test).mean(1),
                    "nll_noisy": nbd(noi, e.y_test).mean(1),
                    "nll_f_under_noisy": nbd(noi, e.f_test).mean(1),  # control: observation-level view of f
                    "mse_latent": (bd.mean(lat) - e.f_test).square().mean(1),
                    "mse_noisy_mean_vs_f": (nbd.mean(noi) - e.f_test).square().mean(1),
                }
                for level in (0.5, 0.9):
                    lo, hi = central_interval(bd, lat, level)
                    out[f"cov{int(level * 100)}_latent"] = ((e.f_test >= lo) & (e.f_test <= hi)).float().mean(1)
                    lo, hi = central_interval(nbd, noi, level)
                    out[f"cov{int(level * 100)}_noisy"] = ((e.y_test >= lo) & (e.y_test <= hi)).float().mean(1)
                frame = props.copy()
                frame["n"] = n
                for k, v in out.items():
                    frame[k] = v.cpu().numpy()
                rows.append(frame)
                for head, b, lg, y in (("latent", bd, lat, e.f_test), ("noisy", nbd, noi, e.y_test)):
                    h, _ = np.histogram(pit(b, lg, y).cpu().numpy().ravel(), bins=bins)
                    pits.append(
                        pd.DataFrame({"head": head, "regime": regime, "n": n, "bin": np.arange(t.pit_bins), "count": h})
                    )
                if regime != "free":
                    v_est = nbd.variance(noi) - bd.variance(lat)
                    ok = (prior.family != FAMILIES.index("contaminated")).cpu().numpy()
                    est, tru = v_est.cpu().numpy(), var_true.cpu().numpy()
                    for b in np.flatnonzero(ok):
                        pos = est[b] > 1e-10
                        logr = np.log(est[b][pos] / tru[b][pos]) if pos.any() else np.array([np.nan])
                        rho = (
                            pd.Series(est[b]).corr(pd.Series(tru[b]), method="spearman")
                            if regime == "hetero"
                            else np.nan
                        )
                        tv.append(
                            {
                                "fn": int(props.fn[b]),
                                "regime": regime,
                                "family": props.family[b],
                                "n": n,
                                "snr": float(props.snr[b]),
                                "median_log_ratio": float(np.median(logr)),
                                "frac_nonpositive": float(1 - pos.mean()),
                                "spearman": float(rho),
                                "true_var_mean": float(tru[b].mean()),
                                "est_var_mean": float(est[b].mean()),
                            }
                        )
    return pd.concat(rows, ignore_index=True), pd.concat(pits, ignore_index=True), pd.DataFrame(tv)


@torch.no_grad()
def martingale(model, prior_cfg, ecdf, t, dev, seed) -> pd.DataFrame:
    """Per function: mean over probe x' of (avg_k mu_{D+(x,y_k)}(x') - mu_D(x')), with its MC error."""
    m = t.martingale
    rows = []
    for c in range(m.n_functions // m.chunk):
        prior = _prior(prior_cfg, m.chunk, seed + 50_000 + c, dev, ecdf)
        prior.reset()
        e = prior.sample_noisy_episode(max(m.context_sizes), 1)
        props = _props_frame(prior, offset=c * m.chunk)
        B = m.chunk
        g = torch.Generator().manual_seed(seed + c)
        x_probe = torch.rand(B, m.n_probe, prior.d, generator=g).to(dev) * prior.active_dim_mask.unsqueeze(1)
        for n in m.context_sizes:
            x_tr, y_tr = e.x_train[:, :n], e.y_train[:, :n]
            x_q = torch.rand(B, 1, prior.d, generator=g).to(dev) * prior.active_dim_mask.unsqueeze(1)
            lat0 = model(x_tr, y_tr, x_probe, prior.active_dim)
            mu0 = model.bar_dist.mean(lat0.float())  # [B, P]
            _, noi = model.forward_both(x_tr, y_tr, x_q, prior.active_dim)
            ys = model.noisy_bar_dist.sample(noi.float(), m.n_fantasies, generator=g)[:, 0]  # [B, K]
            K = m.n_fantasies
            xt = torch.cat([x_tr.repeat_interleave(K, 0), x_q.repeat_interleave(K, 0)], 1)
            yt = torch.cat([y_tr.repeat_interleave(K, 0), ys.reshape(B * K, 1)], 1)
            mu1 = model.bar_dist.mean(
                model(xt, yt, x_probe.repeat_interleave(K, 0), prior.active_dim.repeat_interleave(K)).float()
            ).view(B, K, -1)
            diff = mu1.mean(1) - mu0  # [B, P]
            se = mu1.std(1) / math.sqrt(K)  # MC standard error per probe point
            spread = mu1.std(1).mean(1)  # how much one observation moves the mean
            frame = props.copy()
            frame["n"] = n
            frame["bias_mean"] = diff.mean(1).cpu().numpy()
            frame["bias_abs"] = diff.abs().mean(1).cpu().numpy()
            frame["mc_se"] = se.mean(1).cpu().numpy()
            frame["update_spread"] = spread.cpu().numpy()
            frame["z_mean"] = (diff.mean(1) / (se.square().mean(1) / m.n_probe).sqrt().clamp(min=1e-12)).cpu().numpy()
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


FEATURES = [
    "noise_std_mean",
    "noise_std_log",
    "noise_cv",
    "latent_std_mean",
    "noisy_std_mean",
    "mu_spread",
    "snr_log_est",
    "resid_abs_mean",
    "resid_kurt",
    "resid_frac3",
    "resid_max",
    "noisy_skew",
    "noisy_kurt",
    "noise_at_best",
]


@torch.no_grad()
def probe_features(model, prior_cfg, ecdf, t, dev, seed) -> pd.DataFrame:
    """One row per function x context size: the interpretable state features (doc 7.4, group E)."""
    p = t.probes
    rows = []
    for c in range(p.n_functions // p.chunk):
        prior = _prior(prior_cfg, p.chunk, seed + 90_000 + c, dev, ecdf)
        prior.reset()
        e = prior.sample_noisy_episode(max(p.context_sizes), 1)
        props = _props_frame(prior, offset=c * p.chunk)
        g = torch.Generator().manual_seed(seed + 7 + c)
        x_p = torch.rand(p.chunk, p.n_probe, prior.d, generator=g).to(dev) * prior.active_dim_mask.unsqueeze(1)
        for n in p.context_sizes:
            x_tr, y_tr = e.x_train[:, :n], e.y_train[:, :n]
            xq = torch.cat([x_p, x_tr], 1)
            lat, noi = model.forward_both(x_tr, y_tr, xq, prior.active_dim)
            lat, noi = lat.float(), noi.float()
            mu_l, var_l, _, _ = bar_moments(model.bar_dist, lat)
            _, var_n, sk_n, ku_n = bar_moments(model.noisy_bar_dist, noi)
            nstd = (var_n - var_l).clamp(min=0).sqrt()
            P = p.n_probe
            nstd_p = nstd[:, :P]
            resid = (y_tr - mu_l[:, P:]) / var_n[:, P:].sqrt()
            best = mu_l[:, P:].argmin(1, keepdim=True)
            f = {
                "noise_std_mean": nstd_p.mean(1),
                "noise_std_log": nstd_p.mean(1).clamp(min=1e-4).log(),
                "noise_cv": nstd_p.std(1) / nstd_p.mean(1).clamp(min=1e-4),
                "latent_std_mean": var_l[:, :P].sqrt().mean(1),
                "noisy_std_mean": var_n[:, :P].sqrt().mean(1),
                "mu_spread": mu_l[:, :P].std(1),
                "snr_log_est": (mu_l[:, :P].std(1) / nstd_p.mean(1).clamp(min=1e-4)).log(),
                "resid_abs_mean": resid.abs().mean(1),
                "resid_kurt": resid.pow(4).mean(1) / resid.square().mean(1).clamp(min=1e-8).square(),
                "resid_frac3": (resid.abs() > 3).float().mean(1),
                "resid_max": resid.abs().max(1).values,
                "noisy_skew": sk_n[:, :P].mean(1),
                "noisy_kurt": ku_n[:, :P].mean(1),
                "noise_at_best": nstd[:, P:].gather(1, best)[:, 0],
            }
            frame = props.copy()
            frame["n"] = n
            for k, v in f.items():
                frame[k] = v.cpu().numpy()
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _standardize(tr: np.ndarray, te: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    tr, te = np.nan_to_num(tr, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(te, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    return torch.tensor((tr - mu) / sd, dtype=torch.float64), torch.tensor((te - mu) / sd, dtype=torch.float64)


def fit_logistic(x_tr, y_tr, x_te, n_classes: int, l2: float = 1e-2) -> np.ndarray:
    xa, xb = _standardize(x_tr, x_te)
    W = torch.zeros(xa.shape[1] + 1, n_classes, dtype=torch.float64, requires_grad=True)
    ya = torch.tensor(y_tr)
    opt = torch.optim.LBFGS([W], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = torch.cat([xa, torch.ones(len(xa), 1, dtype=torch.float64)], 1) @ W
        loss = torch.nn.functional.cross_entropy(logits, ya) + l2 * W[:-1].square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (torch.cat([xb, torch.ones(len(xb), 1, dtype=torch.float64)], 1) @ W).argmax(1).numpy()


def fit_ridge(x_tr, y_tr, x_te, l2: float = 1e-2) -> np.ndarray:
    xa, xb = _standardize(x_tr, x_te)
    xa = torch.cat([xa, torch.ones(len(xa), 1, dtype=torch.float64)], 1)
    xb = torch.cat([xb, torch.ones(len(xb), 1, dtype=torch.float64)], 1)
    A = xa.T @ xa + l2 * len(xa) * torch.eye(xa.shape[1], dtype=torch.float64)
    w = torch.linalg.solve(A, xa.T @ torch.tensor(y_tr, dtype=torch.float64))
    return (xb @ w).numpy()


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def identifiability(feats: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Linear probes per context size; train on even functions, score on odd ones."""
    rows = []
    feats = feats.copy()
    feats["noisy"] = (feats.regime != "free").astype(int)
    feats["hetero"] = (feats.regime == "hetero").astype(int)
    feats["log10_snr"] = np.log10(feats.snr)
    targets = [
        ("regime", "class", feats.regime.map({r: i for i, r in enumerate(REGIMES)}), None),
        ("noisy_vs_free", "class", feats.noisy, None),
        ("hetero_vs_homo", "class", feats.hetero, "noisy"),
        ("family", "class", feats.family.map({f: i for i, f in enumerate(FAMILIES)}), "noisy"),
        ("log10_snr", "reg", feats.log10_snr, "noisy"),
        ("d_eff", "reg", feats.d_eff.astype(float), None),
    ]
    for n, df in feats.groupby("n"):
        split = (df.fn % 2 == 0).to_numpy()
        X = df[FEATURES].to_numpy()
        for name, kind, y_all, subset in targets:
            keep = np.ones(len(df), bool) if subset is None else (df[subset] == 1).to_numpy()
            y = y_all.loc[df.index].to_numpy()
            a, b = split & keep, ~split & keep
            if kind == "class":
                k = int(y_all.max()) + 1
                pred = fit_logistic(X[a], y[a].astype(int), X[b], k)
                maj = np.bincount(y[a].astype(int), minlength=k).argmax()
                rows.append(
                    {
                        "n": n,
                        "target": name,
                        "metric": "balanced_accuracy",
                        "value": balanced_accuracy(y[b].astype(int), pred),
                        "null": balanced_accuracy(y[b].astype(int), np.full(b.sum(), maj)),
                        "accuracy": float(np.mean(pred == y[b])),
                        "n_test": int(b.sum()),
                    }
                )
            else:
                pred = fit_ridge(X[a], y[a], X[b])
                ss = np.square(y[b] - y[b].mean()).sum()
                r2 = 1 - np.square(y[b] - pred).sum() / ss
                r2_null = 1 - np.square(y[b] - y[a].mean()).sum() / ss
                rows.append(
                    {
                        "n": n,
                        "target": name,
                        "metric": "r2",
                        "value": float(r2),
                        "null": float(r2_null),
                        "accuracy": np.nan,
                        "n_test": int(b.sum()),
                    }
                )
    return pd.DataFrame(rows)


@torch.no_grad()
def slices(model, prior_cfg, ecdf, t, dev, seed) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1-d cuts along input 0 through the best reference point, a few functions per regime."""
    s = t.slices
    rows, ctx = [], []
    for r_i, regime in enumerate(REGIMES):
        prior = _prior(prior_cfg, s.n_functions, seed + 70_000 + r_i, dev, ecdf)
        prior.reset(regime=regime)
        e = prior.sample_noisy_episode(max(s.contexts), 1)
        x_best = prior.x_ref.expand(prior.B, -1, -1).gather(
            1, prior.latent(prior.x_ref.expand(prior.B, -1, -1)).argmin(1).view(-1, 1, 1).expand(-1, 1, prior.d)
        )
        grid = torch.linspace(0, 1, s.n_grid, device=dev)
        xg = x_best.expand(-1, s.n_grid, -1).clone()
        xg[..., 0] = grid
        f, sc = prior.latent(xg), prior.noise_scale(xg)
        for n in s.contexts:
            lat, noi = model.forward_both(e.x_train[:, :n], e.y_train[:, :n], xg, prior.active_dim)
            lat, noi = lat.float(), noi.float()
            q = {}
            for name, bd, lg in (("latent", model.bar_dist, lat), ("noisy", model.noisy_bar_dist, noi)):
                q[f"{name}_mean"] = bd.mean(lg)
                q[f"{name}_q05"], q[f"{name}_q95"] = central_interval(bd, lg, 0.9)
            for b in range(prior.B):
                df = pd.DataFrame(
                    {
                        "x": grid.cpu().numpy(),
                        "f": f[b].cpu().numpy(),
                        "s": sc[b].cpu().numpy(),
                        **{k: v[b].cpu().numpy() for k, v in q.items()},
                    }
                )
                df["regime"], df["fn"], df["n"] = regime, b, n
                df["family"], df["snr"] = FAMILIES[int(prior.family[b])], float(prior.snr[b])
                rows.append(df)
                dist = (e.x_train[b, :n, 1:] - x_best[b, :, 1:]).norm(dim=-1) if prior.d > 1 else torch.zeros(n)
                ctx.append(
                    pd.DataFrame(
                        {
                            "x": e.x_train[b, :n, 0].cpu().numpy(),
                            "y": e.y_train[b, :n].cpu().numpy(),
                            "dist_off_slice": dist.cpu().numpy(),
                            "regime": regime,
                            "fn": b,
                            "n": n,
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True), pd.concat(ctx, ignore_index=True)


def eval_noisy_pfn(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    dev = device_of(t.device)
    seed_everything(cfg.seed)
    if t.threads:
        torch.set_num_threads(t.threads)
    model, ckpt = load_pfn_file(t.checkpoint, dev)
    prior_cfg = dict(ckpt["prior"])
    ecdf_file = ASSETS / ckpt["ecdf_file"]
    assert sha256(ecdf_file) == ckpt["ecdf_sha256"], "the latent ECDF asset differs from the one used in training"
    ecdf = torch.load(ecdf_file, map_location=dev)
    run.set_tags(checkpoint_sweep=str(ckpt.get("sweep_id")), checkpoint_commit=str(ckpt.get("git_commit")))
    seed = int(t.data_seed)

    cal, pits, tv = calibration(model, prior_cfg, ecdf, t, dev, seed)
    run.log_table("calibration", cal)
    run.log_table("pit_hist", pits)
    run.log_table("total_variance", tv)
    mart = martingale(model, prior_cfg, ecdf, t, dev, seed)
    run.log_table("martingale", mart)
    feats = probe_features(model, prior_cfg, ecdf, t, dev, seed)
    run.log_table("probe_features", feats)
    ident = identifiability(feats, seed)
    run.log_table("identifiability", ident)
    sl, sc = slices(model, prior_cfg, ecdf, t, dev, seed)
    run.log_table("slices", sl)
    run.log_table("slice_context", sc)
    hist = pd.DataFrame({k: pd.Series(v) for k, v in ckpt["history"].items()})
    run.log_table("history", hist)

    summary: dict[str, Any] = {}
    for _, g in cal.groupby(["regime", "n"]):
        regime, n = g.regime.iloc[0], int(g.n.iloc[0])
        summary[f"nll_latent_{regime}_n{n}"] = float(g.nll_latent.mean())
        summary[f"cov90_latent_{regime}_n{n}"] = float(g.cov90_latent.mean())
        summary[f"cov90_noisy_{regime}_n{n}"] = float(g.cov90_noisy.mean())
    for _, r in ident.iterrows():
        summary[f"probe_{r.target}_n{r.n}"] = float(r.value)
    run.log_summary(**summary)
    return float(cal.nll_latent.mean())
