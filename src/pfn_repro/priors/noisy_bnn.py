"""Noisy BNN prior: the PFNs4BO-style BNN as the latent signal, plus an explicit observation-noise model.

Issue #3. Follows the generator design of Decoupled PFNs (Bergna, Depeweg & Hernandez-Lobato 2026,
arXiv:2605.06413): the generator knows the noise-free signal f and the noise scale at every point, so
both become supervised targets of a two-headed PFN (`surrogates/two_head_pfn.py`), and the
epistemic/aleatoric split is identified *relative to this prior* (law of total variance). Differences
from that paper (BNN signal instead of RFF-GP/SCM, SNR-relative noise, non-Gaussian families, a
separately trained noisy head) are listed in `research_log/entries/*_noisy-pfn.md`.

Per function omega = (BNN weights, relevance, noise parameters):

- **Latent** f(x) = F(g(x)) in [0, 1], g the frozen `BNNPrior` network with its own pre-activation and
  output noise switched off, F this prior's own ECDF (`fit_ecdf`; relevance changes the marginal, so
  the frozen checkpoints' ECDF does not apply). Minimization, as everywhere in this repo.
- **Effective dimension.** d_eff ~ U{1..d}; d_eff random inputs get relevance 1, the others 0 or (with
  prob. `p_weak`) `weak_relevance`. Relevance scales the input row of the first BNN layer.
- **Noise scale relative to the function's own range.** On `n_ref` Sobol points: f*_ref = min,
  med_ref = median, Delta = med_ref - f*_ref; s_bar = Delta / SNR, log10 SNR ~ U[`log10_snr_range`].
  An absolute noise level would make narrow-range functions hopeless and wide ones noise-free.
- **Regime** (`regime_probs`): noise-free (s = 0), homoscedastic (s = s_bar), heteroscedastic
  s(x) = s_bar * exp(a * h(x)), h a width-16 one-hidden-layer tanh net standardized over the reference
  points (optionally mixed with standardized f: `coupling_range`), a ~ U[0, `hetero_amp_max`].
- **Family** (`family_probs`): Gaussian, Student-t (nu ~ U[`t_df_range`]), contaminated Gaussian
  (with prob. ~ U[`contam_rate_range`] y is replaced by U[`contam_range`]), skew-normal (alpha ~
  U[`skew_alpha_range`], heavy side towards *worse*, i.e. larger y). Every family is centred (mean 0)
  and scaled to unit *robust* scale (IQR / 1.349), so families at equal SNR are comparable. Noise is
  drawn by inverse CDF from a per-function quantile table, so it is a deterministic function of a
  uniform u: pass `generator=` (or `u=`) for common random numbers.

Contexts (`sample_noisy_episode`): per function, either uniform x, or a BO-like sequence (a few uniform
points, then rounds of perturbations around the best *observed* points), and optionally exact
replicates of earlier points with fresh noise. Every context location depends only on earlier
observations, never on omega directly: such designs are ignorable, so the PFN cannot learn to read the
optimum off where the points are. Test points: uniform, near context points, and exact context points.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from .bnn import BNNPrior

ASSETS = Path(__file__).resolve().parent.parent / "assets"

REGIMES = ("free", "homo", "hetero")
FAMILIES = ("gauss", "student_t", "contaminated", "skew")

# Quantile table of the unit noise: u on a logit-spaced grid, dense in the tails (1e-4 .. 1 - 1e-4).
_U_GRID = torch.sigmoid(torch.linspace(-9.21, 9.21, 2049, dtype=torch.float64))


@dataclass
class NoisyEpisode:
    x_train: torch.Tensor  # [B, n, d]
    y_train: torch.Tensor  # [B, n] noisy observations
    f_train: torch.Tensor  # [B, n] latent values (privileged)
    x_test: torch.Tensor  # [B, m, d]
    y_test: torch.Tensor  # [B, m] fresh noisy observations
    f_test: torch.Tensor  # [B, m] latent values (privileged)
    s_test: torch.Tensor  # [B, m] robust noise scale at x_test (privileged)


class NoisyBNNPrior:
    def __init__(
        self,
        batch_size: int,
        x_dim: int,
        seed: int | None = None,
        device: str = "cpu",
        variable_dim_min: int | None = None,
        # latent BNN family (BNNPrior's own arguments; its noise is switched off)
        depth_range: tuple[int, int] = (8, 32),
        width_range: tuple[int, int] = (32, 128),
        log_amp_range: tuple[float, float] = (8.0, 20.0),
        bias_std: float = 0.1,
        sparseness: float = 0.145,
        # effective dimension
        weak_relevance: float = 0.05,
        p_weak: float = 0.5,
        # noise
        regime_probs: tuple[float, float, float] = (0.2, 0.4, 0.4),
        family_probs: tuple[float, float, float, float] = (0.6, 0.2, 0.1, 0.1),
        log10_snr_range: tuple[float, float] = (math.log10(2.0), 2.0),
        t_df_range: tuple[float, float] = (3.0, 10.0),
        contam_rate_range: tuple[float, float] = (0.01, 0.1),
        contam_range: tuple[float, float] = (-0.5, 1.5),
        skew_alpha_range: tuple[float, float] = (3.0, 10.0),
        hetero_amp_max: float = math.log(10.0) / 4,
        hetero_width: int = 16,
        coupling_range: tuple[float, float] = (0.0, 0.0),
        n_ref: int = 512,
        # contexts
        p_bo_context: float = 0.5,
        bo_init_frac: tuple[float, float] = (0.1, 0.3),
        bo_rounds: int = 6,
        p_local: float = 0.7,
        local_scales: tuple[float, ...] = (0.01, 0.03, 0.1),
        p_replicates: float = 0.4,
        replicate_frac_max: float = 0.3,
        test_mix: tuple[float, float, float] = (0.5, 0.4, 0.1),  # uniform, near context, at context x
        # ECDF of the latent: None -> load the packaged asset for this config, fit (and save) if missing
        ecdf_sorted: torch.Tensor | None = None,
        ecdf_n_draws: int = 50,
        ecdf_samples_per_draw: int = 300,
        ecdf_n_samples: int = 1000,
    ):
        self.B, self.d, self.device = batch_size, x_dim, device
        self.variable_dim_min = variable_dim_min
        self.family_kwargs: dict[str, Any] = dict(
            depth_range=tuple(depth_range),
            width_range=tuple(width_range),
            log_amp_range=tuple(log_amp_range),
            bias_std=bias_std,
            sparseness=sparseness,
        )
        self.weak_relevance, self.p_weak = weak_relevance, p_weak
        self.regime_probs = torch.tensor(regime_probs, dtype=torch.float)
        self.family_probs = torch.tensor(family_probs, dtype=torch.float)
        self.log10_snr_range = tuple(log10_snr_range)
        self.t_df_range, self.contam_rate_range = tuple(t_df_range), tuple(contam_rate_range)
        self.contam_range, self.skew_alpha_range = tuple(contam_range), tuple(skew_alpha_range)
        self.hetero_amp_max, self.hetero_width = hetero_amp_max, hetero_width
        self.coupling_range, self.n_ref = tuple(coupling_range), n_ref
        self.p_bo_context, self.bo_init_frac, self.bo_rounds = p_bo_context, tuple(bo_init_frac), bo_rounds
        self.p_local, self.local_scales = p_local, tuple(local_scales)
        self.p_replicates, self.replicate_frac_max = p_replicates, replicate_frac_max
        self.test_mix = tuple(test_mix)
        self.ecdf_cfg = (ecdf_n_draws, ecdf_samples_per_draw, ecdf_n_samples)

        self.base = BNNPrior(
            batch_size=batch_size,
            x_dim=x_dim,
            device=device,
            seed=seed,
            variable_dim_min=variable_dim_min,
            preactivation_noise_std_range=(0.0, 0.0),
            output_noise_std_range=(0.0, 0.0),
            frac_relevant_features=1.0,
            ecdf_sorted=torch.linspace(0, 1, 8),
            **self.family_kwargs,
        )
        self.generator = self.base.generator  # one stream: the trainer checkpoints its state
        if ecdf_sorted is None:
            ecdf_sorted = self._load_or_fit_ecdf()
        self.ecdf_sorted = ecdf_sorted[:1].to(device)
        self.base.ecdf_sorted = self.ecdf_sorted.expand(self.B, -1).clone()
        self.reset()

    # ------------------------------------------------------------------ ECDF
    def ecdf_key(self) -> str:
        fp = (
            self.d,
            self.variable_dim_min,
            sorted(self.family_kwargs.items()),
            self.weak_relevance,
            self.p_weak,
            self.ecdf_cfg,
        )
        return f"noisy_bnn_ecdf_d{self.d}_{hashlib.sha256(repr(fp).encode()).hexdigest()[:16]}.pt"

    def _load_or_fit_ecdf(self) -> torch.Tensor:
        path = ASSETS / self.ecdf_key()
        if path.exists():
            return torch.load(path, map_location="cpu")
        ecdf = self.fit_ecdf()
        ASSETS.mkdir(parents=True, exist_ok=True)
        torch.save(ecdf, path)
        print(f"fitted and saved the latent ECDF to {path} (commit it: training runs must share it)")
        return ecdf

    @torch.no_grad()
    def fit_ecdf(self, seed: int = 12345) -> torch.Tensor:
        """ECDF of the raw latent network output over the family (with relevance), on CPU with a fixed
        seed, so every machine gets the same file. -> [1, ecdf_n_samples]."""
        n_draws, per_draw, n_samples = self.ecdf_cfg
        fitter = BNNPrior(
            batch_size=self.B,
            x_dim=self.d,
            device="cpu",
            seed=seed,
            variable_dim_min=self.variable_dim_min,
            preactivation_noise_std_range=(0.0, 0.0),
            output_noise_std_range=(0.0, 0.0),
            frac_relevant_features=1.0,
            ecdf_sorted=torch.linspace(0, 1, 8),
            **self.family_kwargs,
        )
        raw = []
        for _ in range(n_draws):
            fitter.reset()
            fitter.W_in = fitter.W_in * self._relevance(fitter.active_dim, fitter.generator, "cpu").unsqueeze(-1)
            x = torch.rand(self.B, per_draw, self.d, generator=fitter.generator) * fitter.active_dim_mask.unsqueeze(1)
            raw.append(fitter._raw_forward(x, noise=False))
        pooled, _ = torch.sort(torch.cat(raw, 1).flatten())
        idx = torch.linspace(0, pooled.numel() - 1, n_samples).long()
        return pooled[idx].unsqueeze(0)

    # ------------------------------------------------------------------ draws
    def _relevance(self, active_dim: torch.Tensor, gen: torch.Generator, dev: str) -> torch.Tensor:
        B, d = active_dim.shape[0], self.d
        u = torch.rand(B, 1, device=dev, generator=gen)
        d_eff = (u * active_dim.view(B, 1).float()).floor().long() + 1  # U{1..active_dim}
        keys = torch.rand(B, d, device=dev, generator=gen)
        keys = torch.where(
            torch.arange(d, device=dev).view(1, d) < active_dim.view(B, 1), keys, torch.full_like(keys, 2.0)
        )
        rank = keys.argsort(1).argsort(1)  # random order of the active dims; inactive ones last
        weak = torch.rand(B, d, device=dev, generator=gen) < self.p_weak
        lam = torch.where(rank < d_eff, 1.0, torch.where(weak, self.weak_relevance, 0.0))
        active = torch.arange(d, device=dev).view(1, d) < active_dim.view(B, 1)
        self._last_d_eff = d_eff.view(B)
        return lam * active.float()

    def _seed_int(self) -> int:
        return int(torch.randint(2**31, (1,), device=self.device, generator=self.generator))

    def _uniform(self, shape, lo: float, hi: float) -> torch.Tensor:
        return torch.empty(shape, device=self.device).uniform_(lo, hi, generator=self.generator)

    def reset(self, regime: str | None = None, family: str | None = None, snr: float | None = None) -> None:
        """New functions. `regime` / `family` / `snr` pin that property for the whole batch (evaluation
        slices); None samples it from the prior."""
        B, d, dev, gen = self.B, self.d, self.device, self.generator
        self.base.reset()
        self.relevance = self._relevance(self.base.active_dim, gen, dev)
        self.d_eff = self._last_d_eff
        self.base.W_in = self.base.W_in * self.relevance.unsqueeze(-1)
        self.active_dim, self.active_dim_mask = self.base.active_dim, self.base.active_dim_mask

        # heteroscedasticity field h: one hidden layer, standardized over the reference points below
        H = self.hetero_width
        self.h_W1 = torch.randn(B, d, H, device=dev, generator=gen) * 2.0 * self.active_dim_mask.unsqueeze(-1)
        self.h_b1 = torch.randn(B, H, device=dev, generator=gen)
        self.h_W2 = torch.randn(B, H, device=dev, generator=gen) / math.sqrt(H)

        cat_gen = torch.Generator().manual_seed(self._seed_int())  # multinomial needs a CPU generator here
        self.regime = (
            torch.multinomial(self.regime_probs, B, replacement=True, generator=cat_gen).to(dev)
            if regime is None
            else torch.full((B,), REGIMES.index(regime), device=dev)
        )
        self.family = (
            torch.multinomial(self.family_probs, B, replacement=True, generator=cat_gen).to(dev)
            if family is None
            else torch.full((B,), FAMILIES.index(family), device=dev)
        )
        lo, hi = self.log10_snr_range
        self.snr = 10 ** self._uniform((B,), lo, hi) if snr is None else torch.full((B,), float(snr), device=dev)
        self.t_df = self._uniform((B,), *self.t_df_range)
        self.contam_rate = torch.where(
            self.family == FAMILIES.index("contaminated"),
            self._uniform((B,), *self.contam_rate_range),
            torch.zeros(B, device=dev),
        )
        self.skew_alpha = self._uniform((B,), *self.skew_alpha_range)
        self.hetero_amp = torch.where(
            self.regime == REGIMES.index("hetero"),
            self._uniform((B,), 0.0, self.hetero_amp_max),
            torch.zeros(B, device=dev),
        )
        self.coupling = (
            self._uniform((B,), *self.coupling_range)
            if self.coupling_range[1] > self.coupling_range[0]
            else torch.zeros(B, device=dev)
        )

        # reference points: the function's own range sets the noise scale
        sobol = torch.quasirandom.SobolEngine(d, scramble=True, seed=self._seed_int())
        self.x_ref = sobol.draw(self.n_ref).to(dev).unsqueeze(0) * self.active_dim_mask.unsqueeze(1)
        with torch.no_grad():
            f_ref = self.latent(self.x_ref.expand(B, -1, -1))
            self.f_star_ref = f_ref.min(1).values
            self.f_med_ref = f_ref.median(1).values
            self.f_q05_ref = f_ref.quantile(0.05, dim=1)
            self.f_ref_mean, self.f_ref_std = f_ref.mean(1), f_ref.std(1).clamp(min=1e-6)
            self._h_mean = torch.zeros(B, device=dev)
            self._h_std = torch.ones(B, device=dev)
            h_ref = self._h_raw(self.x_ref.expand(B, -1, -1), f_ref)
            self._h_mean, self._h_std = h_ref.mean(1), h_ref.std(1).clamp(min=1e-6)
        self.gap = (self.f_med_ref - self.f_star_ref).clamp(min=1e-4)  # Delta
        self.s_bar = torch.where(self.regime == REGIMES.index("free"), torch.zeros(B, device=dev), self.gap / self.snr)
        self.snr_top = torch.where(
            self.s_bar > 0,
            (self.f_q05_ref - self.f_star_ref) / self.s_bar.clamp(min=1e-12),
            torch.full((B,), float("inf"), device=dev),
        )
        self._build_noise_tables()

    def _h_raw(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        xs = (x - 0.5) / math.sqrt(1 / 12)
        h = torch.tanh(torch.einsum("bnd,bdh->bnh", xs, self.h_W1) + self.h_b1.unsqueeze(1))
        h = torch.einsum("bnh,bh->bn", h, self.h_W2)
        h = (h - self._h_mean.unsqueeze(1)) / self._h_std.unsqueeze(1)
        z_f = (f - self.f_ref_mean.unsqueeze(1)) / self.f_ref_std.unsqueeze(1)
        c = self.coupling.unsqueeze(1)
        return c * z_f + (1 - c.abs()) * h

    def _build_noise_tables(self) -> None:
        """Per function: quantile table of the unit noise (mean 0, IQR / 1.349 = 1) and its variance."""
        u = _U_GRID.numpy()
        fam = self.family.cpu().numpy()
        df, alpha = self.t_df.cpu().double().numpy(), self.skew_alpha.cpu().double().numpy()
        tables, var = np.empty((self.B, u.size)), np.empty(self.B)
        for b in range(self.B):
            if FAMILIES[fam[b]] == "student_t":
                dist = stats.t(df[b])
            elif FAMILIES[fam[b]] == "skew":
                dist = stats.skewnorm(alpha[b])  # alpha > 0: heavy right tail = worse values (minimization)
            else:
                dist = stats.norm()
            robust = (dist.ppf(0.75) - dist.ppf(0.25)) / 1.349
            tables[b] = (dist.ppf(u) - dist.mean()) / robust
            var[b] = dist.var() / robust**2
        self.noise_table = torch.as_tensor(tables, dtype=torch.float32, device=self.device)
        self.unit_noise_var = torch.as_tensor(var, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ evaluation
    def latent(self, x: torch.Tensor) -> torch.Tensor:
        """Noise-free f(x) in [0, 1]. x: [B, N, d] -> [B, N]. Differentiable in x."""
        return self.base.evaluate(x, noise=False)

    def noise_scale(self, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        """Robust noise scale s(x) (IQR / 1.349 of y given f). x: [B, N, d] -> [B, N]."""
        if f is None:
            f = self.latent(x)
        h = self._h_raw(x, f)
        return self.s_bar.unsqueeze(1) * torch.exp(self.hetero_amp.unsqueeze(1) * h)

    def noise_var(self, x: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        """Var(y | f, x) for the non-contaminated families (the target of the total-variance check)."""
        return self.noise_scale(x, f).square() * self.unit_noise_var.unsqueeze(1)

    def unit_noise(self, u: torch.Tensor) -> torch.Tensor:
        """Inverse CDF of each function's unit noise. u: [B, N] in (0, 1) -> [B, N]."""
        grid = _U_GRID.to(device=u.device, dtype=torch.float32)
        u = u.clamp(float(grid[0]), float(grid[-1]))
        j = torch.searchsorted(grid, u.contiguous()).clamp(1, grid.numel() - 1)
        w = (u - grid[j - 1]) / (grid[j] - grid[j - 1])
        lo = self.noise_table.gather(1, j - 1)
        hi = self.noise_table.gather(1, j)
        return lo + w * (hi - lo)

    def observe(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
        f: torch.Tensor | None = None,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Noisy y at x [B, N, d] -> [B, N]. `u` [B, N, 3] uniform draws (noise, contamination flag,
        outlier value) make the noise a deterministic function of u, for common random numbers."""
        gen = generator if generator is not None else self.generator
        if f is None:
            f = self.latent(x)
        if u is None:
            u = torch.rand(*f.shape, 3, device=f.device, generator=gen)
        y = f + self.noise_scale(x, f) * self.unit_noise(u[..., 0])
        lo, hi = self.contam_range
        outlier = u[..., 1] < self.contam_rate.unsqueeze(1)
        return torch.where(outlier, lo + (hi - lo) * u[..., 2], y)

    # ------------------------------------------------------------------ episodes
    def _reflect(self, x: torch.Tensor) -> torch.Tensor:
        x = x.abs()
        return (1 - (1 - x).abs()).clamp(0, 1)

    def _perturb(self, centers: torch.Tensor) -> torch.Tensor:
        """Gaussian perturbations with a scale drawn per point from `local_scales`, reflected."""
        gen = self.generator
        scales = torch.tensor(self.local_scales, device=self.device)
        pick = torch.randint(len(self.local_scales), centers.shape[:-1], device=self.device, generator=gen)
        noise = torch.randn(centers.shape, device=self.device, generator=gen) * scales[pick].unsqueeze(-1)
        return self._reflect(centers + noise) * self.active_dim_mask.unsqueeze(1)

    def _gather_points(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    @torch.no_grad()
    def sample_noisy_episode(self, n_train: int, n_test: int) -> NoisyEpisode:
        B, d, dev, gen = self.B, self.d, self.device, self.generator
        mask = self.active_dim_mask.unsqueeze(1)
        bo = torch.rand(B, device=dev, generator=gen) < self.p_bo_context
        rep_frac = torch.where(
            torch.rand(B, device=dev, generator=gen) < self.p_replicates,
            self._uniform((B,), 0.0, self.replicate_frac_max),
            torch.zeros(B, device=dev),
        )
        lo, hi = self.bo_init_frac
        n_init = (self._uniform((B,), lo, hi) * n_train).ceil().long().clamp(1, n_train)
        n_init = torch.where(bo, n_init, torch.full_like(n_init, n_train))
        # rounds: [0, c_1), [c_1, c_2), ...; round 0 is uniform for everyone, later rounds are BO-like
        bounds = sorted(
            {0, *np.linspace(int(n_init.min()), n_train, self.bo_rounds + 1).round().astype(int).tolist(), n_train}
        )
        x = torch.zeros(B, 0, d, device=dev)
        y = torch.zeros(B, 0, device=dev)
        f = torch.zeros(B, 0, device=dev)
        for a, b in itertools.pairwise(bounds):
            m = b - a
            pos = torch.arange(a, b, device=dev).view(1, m)
            x_new = torch.rand(B, m, d, device=dev, generator=gen) * mask
            if a > 0:
                k = min(3, a)
                top = y.topk(k, dim=1, largest=False).indices  # best observed (noisy) points so far
                c = top.gather(1, torch.randint(k, (B, m), device=dev, generator=gen))
                local = self._perturb(self._gather_points(x, c))
                use_local = (pos >= n_init.view(B, 1)) & (torch.rand(B, m, device=dev, generator=gen) < self.p_local)
                x_new = torch.where(use_local.unsqueeze(-1), local, x_new)
                # exact replicates of earlier points (half of them at the current best few)
                r_any = torch.randint(a, (B, m), device=dev, generator=gen)
                r_idx = torch.where(torch.rand(B, m, device=dev, generator=gen) < 0.5, c, r_any)
                use_rep = torch.rand(B, m, device=dev, generator=gen) < rep_frac.view(B, 1)
                x_new = torch.where(use_rep.unsqueeze(-1), self._gather_points(x, r_idx), x_new)
            f_new = self.latent(x_new)
            y_new = self.observe(x_new, f=f_new)
            x, y, f = torch.cat([x, x_new], 1), torch.cat([y, y_new], 1), torch.cat([f, f_new], 1)

        w_u, w_near, _ = self.test_mix
        kind = torch.rand(B, n_test, device=dev, generator=gen)
        x_te = torch.rand(B, n_test, d, device=dev, generator=gen) * mask
        anchor = self._gather_points(x, torch.randint(n_train, (B, n_test), device=dev, generator=gen))
        x_te = torch.where((kind >= w_u).unsqueeze(-1), self._perturb(anchor), x_te)
        x_te = torch.where((kind >= w_u + w_near).unsqueeze(-1), anchor, x_te)
        f_te = self.latent(x_te)
        y_te = self.observe(x_te, f=f_te)
        s_te = self.noise_scale(x_te, f_te)
        self.episode_info = {"bo_context": bo, "replicate_frac": rep_frac, "n_init": n_init}
        return NoisyEpisode(x, y, f, x_te, y_te, f_te, s_te)

    def sample_episode(self, n_train: int, n_test: int):
        """The repo's episode interface (x_tr, y_tr, x_te, y_te) with *noisy* test targets."""
        e = self.sample_noisy_episode(n_train, n_test)
        return e.x_train, e.y_train, e.x_test, e.y_test

    def properties(self) -> dict[str, torch.Tensor]:
        """Per-function ground truth for evaluation and probes (privileged: never a policy input)."""
        return {
            "regime": self.regime,
            "family": self.family,
            "snr": self.snr,
            "snr_top": self.snr_top,
            "s_bar": self.s_bar,
            "gap": self.gap,
            "f_star_ref": self.f_star_ref,
            "d_eff": self.d_eff,
            "hetero_amp": self.hetero_amp,
            "t_df": self.t_df,
            "contam_rate": self.contam_rate,
            "active_dim": self.active_dim,
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    prior = NoisyBNNPrior(batch_size=6, x_dim=1, seed=0)
    for i, regime in enumerate(REGIMES):
        prior.reset(regime=regime)
        x = torch.linspace(0, 1, 400).view(1, -1, 1).expand(prior.B, -1, -1)
        f, y = prior.latent(x), prior.observe(x)
        s = prior.noise_scale(x)
        ax = plt.subplot(1, 3, i + 1)
        ax.plot(x[0, :, 0], f[0], "k-", lw=1, label="latent f")
        ax.plot(x[0, :, 0], y[0], ".", ms=2, label="y")
        ax.fill_between(x[0, :, 0], f[0] - 2 * s[0], f[0] + 2 * s[0], alpha=0.2, label="f +- 2 s(x)")
        ax.set_title(f"{regime}: {FAMILIES[int(prior.family[0])]}, SNR {float(prior.snr[0]):.1f}")
    plt.legend()
    plt.gcf().set_size_inches(13, 3.5)
    out = Path("reports/figures/noisy_bnn_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("saved", out)
    prior = NoisyBNNPrior(batch_size=4, x_dim=2, seed=0)
    e = prior.sample_noisy_episode(30, 64)
    print({k: v.shape for k, v in e.__dict__.items()})
    print("properties:", {k: v.tolist() for k, v in prior.properties().items()})
