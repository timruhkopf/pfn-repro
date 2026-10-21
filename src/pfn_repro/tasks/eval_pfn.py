"""Predictive quality of registered PFN checkpoints on held-out BNN-prior draws (pfn-acqpolicy's eval_pfn,
ported; read in `notebooks/03_*_pfn_quality.ipynb`).

Per model, d and context size n (nested prefixes of one context, so all n share functions and test
points): NLL, squared error of the mean, CRPS, PIT calibration, interval coverage, sharpness,
per-function Spearman and selection regret (`evaluation/predictive.py`). Plus CPU forward cost, the
checkpoint's own training history, and 1-d panels (true f, context, the PPD density on the bar
distribution's intervals, mean and 5-95% band) for a few draws. Tables (run sink, then snapshot):
  metrics            (model, d, n, metric, mean, lo, hi), 95% bootstrap over functions
  per_function       one row per (model, d, n, function)
  pit_hist           (model, d, n, bin, freq), 20 PIT bins
  timing             (model, d, n, batch, ms) median forward wall time
  history            (model, step, metric columns...) from each checkpoint
  checkpoints        registry entry + checkpoint metadata per model
  slice_lines / slice_context / slice_density / slice_borders   the 1-d panels
"""

import json
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from ..checkpoints import REGISTRY, load, make_prior
from ..evaluation.predictive import function_metrics, point_metrics
from ..tracking.sink import Run
from .common import to_dict

POINT = ("nll", "sq_err", "crps", "std", "cov50", "cov90")
FUNC = ("spearman", "select_regret")


def held_out_sets(dims, prior_kwargs, n_max, n_test, n_functions, seed, device):
    return {
        d: make_prior(d, n_functions, seed=seed + d, device=device, **prior_kwargs).sample_episode(n_max, n_test)
        for d in dims
    }


@torch.no_grad()
def evaluate(model, sets, context_sizes, chunk) -> tuple[pd.DataFrame, pd.DataFrame]:
    bd = model.bar_dist
    rows, pit_rows = [], []
    for d, (x_tr, y_tr, x_te, y_te) in sets.items():
        for n in context_sizes:
            parts = []
            for s in range(0, x_tr.shape[0], chunk):
                sl = slice(s, s + chunk)
                nf = torch.full((x_tr[sl].shape[0],), d, dtype=torch.long)
                logits = model(x_tr[sl, :n], y_tr[sl, :n], x_te[sl], n_features=nf)
                pm = point_metrics(bd, logits, y_te[sl])
                fm = function_metrics(pm["mean"], y_te[sl])
                u = pm["pit"]
                parts.append(
                    {
                        **{k: pm[k].mean(-1) for k in ("nll", "sq_err", "crps", "std")},
                        "cov50": ((u > 0.25) & (u < 0.75)).float().mean(-1),
                        "cov90": ((u > 0.05) & (u < 0.95)).float().mean(-1),
                        **fm,
                        "pit": u,
                    }
                )
            cat = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
            df = pd.DataFrame({k: cat[k].numpy() for k in POINT + FUNC})
            df["d"], df["n"], df["function"] = d, n, np.arange(len(df))
            rows.append(df)
            h, _ = np.histogram(cat["pit"].numpy().ravel(), bins=20, range=(0, 1))
            pit_rows.append(pd.DataFrame({"d": d, "n": n, "bin": np.arange(20), "freq": h / h.sum()}))
    return pd.concat(rows, ignore_index=True), pd.concat(pit_rows, ignore_index=True)


def summarize(per_fn: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for key, g in per_fn.groupby(["model", "d", "n"]):
        model, d, n = key  # type: ignore[misc]
        idx = rng.integers(0, len(g), (n_boot, len(g)))
        for m in POINT + FUNC:
            v = g[m].to_numpy()
            b = v[idx].mean(1)
            out.append(
                dict(model=model, d=d, n=n, metric=m, mean=v.mean(), lo=np.quantile(b, 0.025), hi=np.quantile(b, 0.975))
            )
    return pd.DataFrame(out)


@torch.no_grad()
def time_forward(model, dims, context_sizes, n_test, batch, reps) -> pd.DataFrame:
    rows = []
    for d in dims:
        for n in context_sizes:
            x_tr, y_tr, x_te = torch.rand(batch, n, d), torch.rand(batch, n), torch.rand(batch, n_test, d)
            nf = torch.full((batch,), d, dtype=torch.long)
            model(x_tr, y_tr, x_te, n_features=nf)  # warm-up
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                model(x_tr, y_tr, x_te, n_features=nf)
                ts.append(time.perf_counter() - t0)
            rows.append(dict(d=d, n=n, batch=batch, n_test=n_test, ms=1000 * float(np.median(ts))))
    return pd.DataFrame(rows)


def count_local_minima(f: torch.Tensor) -> torch.Tensor:
    """f [B, G] on a 1-d grid -> number of strict interior local minima."""
    c = (f[:, 1:-1] < f[:, :-2]) & (f[:, 1:-1] < f[:, 2:])
    depth = torch.minimum(f[:, :-2], f[:, 2:]) - f[:, 1:-1]
    return (c & (depth >= 0)).sum(-1)


def history_frame(ckpt: dict, val_every: int | None = None) -> pd.DataFrame:
    """Train metrics every log step; validation metrics on their own (sparser) steps, aligned by cadence
    (inferred if not given: the smallest log step whose multiples match the number of validation entries)."""
    h = ckpt["history"]
    steps = np.asarray(h["step"])
    df = pd.DataFrame({k: v for k, v in h.items() if len(v) == len(steps)})
    val = {k: v for k, v in h.items() if len(v) != len(steps)}
    if val:
        n_val = len(next(iter(val.values())))
        if val_every:
            val_steps = steps[steps % val_every == 0]
        else:
            val_steps = next(s for e in steps[steps > 0] if len(s := steps[steps % e == 0]) == n_val)
        assert len(val_steps) == n_val, (len(val_steps), n_val)
        vdf = pd.DataFrame(val)
        vdf["step"] = val_steps
        df = df.merge(vdf, on="step", how="left")
    return df


def checkpoint_record(name: str, ckpt: dict, model: torch.nn.Module) -> dict:
    spec = REGISTRY[name]
    meta = {k: v for k, v in ckpt.items() if k not in ("model_state", "train_state", "history")}
    return {
        "model": name,
        "file": spec.file,
        "sha256": spec.sha256,
        "kind": spec.kind,
        "train_n_min": spec.train_n[0],
        "train_n_max": spec.train_n[1],
        "description": spec.description,
        "n_params": sum(p.numel() for p in model.parameters()),
        "source": json.dumps(spec.source),
        "metadata": json.dumps(meta, default=str),
    }


def rebin(density: torch.Tensor, borders: torch.Tensor, max_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge consecutive bins (exact mass sums) so at most `max_bins` remain: keeps committed tables small."""
    nb = density.shape[-1]
    k = max(1, -(-nb // max_bins))
    if k == 1:
        return density, borders
    widths = borders[1:] - borders[:-1]
    mass = density * widths
    m = nb // k * k
    mass = mass[..., :m].reshape(*mass.shape[:-1], -1, k).sum(-1)
    new_b = torch.cat([borders[:m:k], borders[m : m + 1]])
    return mass / (new_b[1:] - new_b[:-1]), new_b


@torch.no_grad()
def slices_1d(models: dict, prior_kwargs: dict, s: DictConfig) -> dict[str, pd.DataFrame]:
    """The `n_functions` 'most interesting' 1-d draws (most local minima among draws spanning more than
    half of [0, 1]); per model and nested context size: PPD density on the bar intervals, mean, 5-95%."""
    prior = make_prior(1, s.n_candidates, seed=s.seed, **prior_kwargs)
    coarse = torch.linspace(0, 1, 64).view(1, -1, 1).expand(s.n_candidates, -1, -1)
    fc = prior.evaluate(coarse, noise=False)
    span = fc.max(-1).values - fc.min(-1).values
    score = count_local_minima(fc).float() - 100.0 * (span < 0.5)
    picks = score.argsort(descending=True)[: s.n_functions].tolist()
    grid = torch.linspace(0, 1, s.n_grid).view(1, -1, 1).expand(s.n_candidates, -1, -1)
    f_true = prior.evaluate(grid, noise=False)
    g = torch.Generator().manual_seed(s.seed + 1)
    xc = torch.rand(1, max(s.contexts), 1, generator=g).expand(s.n_candidates, -1, -1).contiguous()
    yc = prior.evaluate(xc, noise=True)
    lines, ctx, dens, borders = [], [], [], []
    xs = grid[0, :, 0].numpy()
    keep_x = np.linspace(0, s.n_grid - 1, s.density_x).round().astype(int)
    for fi, b in enumerate(picks):
        ctx.append(
            pd.DataFrame(
                {"function": fi, "order": np.arange(max(s.contexts)), "x": xc[b, :, 0].numpy(), "y": yc[b].numpy()}
            )
        )
        for name, model in models.items():
            bd = model.bar_dist
            for n in s.contexts:
                logits = model(
                    xc[b : b + 1, :n], yc[b : b + 1, :n], grid[b : b + 1], n_features=torch.ones(1, dtype=torch.long)
                )[0].float()
                p = torch.softmax(logits, -1)
                lines.append(
                    pd.DataFrame(
                        {
                            "model": name,
                            "function": fi,
                            "n": n,
                            "x": xs,
                            "f_true": f_true[b].numpy(),
                            "mean": bd.mean(logits).numpy(),
                            "q05": bd.icdf(logits, 0.05).numpy(),
                            "q95": bd.icdf(logits, 0.95).numpy(),
                        }
                    )
                )
                d, nb = rebin(p / bd.bucket_widths, bd.borders, s.max_bins)
                d = d[keep_x].numpy()
                xi, bi = np.meshgrid(keep_x, np.arange(d.shape[1]), indexing="ij")
                dens.append(
                    pd.DataFrame(
                        {
                            "model": name,
                            "function": fi,
                            "n": n,
                            "x": xs[xi.ravel()],
                            "bin": bi.ravel(),
                            "density": d.ravel().astype(np.float32),
                        }
                    )
                )
                if fi == 0 and n == s.contexts[0]:
                    borders.append(
                        pd.DataFrame(
                            {"model": name, "bin": np.arange(len(nb) - 1), "lo": nb[:-1].numpy(), "hi": nb[1:].numpy()}
                        )
                    )
    return {
        "slice_lines": pd.concat(lines),
        "slice_context": pd.concat(ctx),
        "slice_density": pd.concat(dens),
        "slice_borders": pd.concat(borders),
    }


def eval_pfn(cfg: DictConfig, run: Run) -> float:
    t = cfg.task
    torch.manual_seed(cfg.seed)
    if t.threads:
        torch.set_num_threads(t.threads)
    prior_kwargs = {k: v for k, v in to_dict(cfg.prior).items() if k != "name"}
    sets = held_out_sets(t.dims, prior_kwargs, max(t.context_sizes), t.n_test, t.n_functions, t.data_seed, "cpu")
    per_fn, pits, timing, histories, records, models = [], [], [], [], [], {}
    for name in t.models:
        model, ckpt = load(name, "cpu")
        models[name] = model
        lo, hi = REGISTRY[name].train_n
        pf, ph = evaluate(model, sets, t.context_sizes, t.chunk)
        per_fn.append(pf.assign(model=name, in_train_range=(pf.n >= lo) & (pf.n <= hi)))
        pits.append(ph.assign(model=name))
        timing.append(
            time_forward(model, t.dims, t.context_sizes, t.timing.n_test, t.timing.batch, t.timing.reps).assign(
                model=name, threads=torch.get_num_threads()
            )
        )
        histories.append(history_frame(ckpt, t.val_every.get(name)).assign(model=name))
        records.append(checkpoint_record(name, ckpt, model))
    per_fn_df = pd.concat(per_fn, ignore_index=True)
    run.log_table("per_function", per_fn_df)
    run.log_table("metrics", summarize(per_fn_df, t.n_boot, cfg.seed))
    run.log_table("pit_hist", pd.concat(pits, ignore_index=True))
    run.log_table("timing", pd.concat(timing, ignore_index=True))
    run.log_table("history", pd.concat(histories, ignore_index=True))
    run.log_table("checkpoints", pd.DataFrame(records))
    for table, df in slices_1d(models, prior_kwargs, t.slice).items():
        run.log_table(table, df.reset_index(drop=True))
    nll20 = per_fn_df[(per_fn_df.n == 20) & (per_fn_df.model == t.models[0])].nll.mean()
    run.log_summary(**{"final/nll_n20": float(nll20)})
    return float(nll20)
