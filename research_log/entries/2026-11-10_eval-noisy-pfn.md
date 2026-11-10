---
experiment: eval_noisy_pfn
sweeps: []
---
# Stage-0 checks of the noisy two-headed PFN: calibration, total variance, martingale, identifiability

## Question

See the description at the top of `notebooks/07_2026-11-10_eval_noisy_pfn.ipynb`; not retyped here.

## §1 Sanity (2026-11-10)

- `sanity_20261110-120518` (laptop CPU, commit 3206226; snapshot `results/eval_noisy_pfn/sanity_20261110-120518`)
  on the toy sanity checkpoint (`noisy_pfn` sanity_20261110-115538), reduced sizes (128 functions per regime,
  64 for the martingale test, 1024 for the probes). All tables written; notebook 07's cells execute against it.
- Found and fixed before any pilot evaluation: Var_noisy - Var_latent overestimates the noise variance by
  a factor ~e^5 at every n (median log ratio 4.3-5.5 over regimes and families, `total_variance` table):
  the variance is tail-dominated, and the noisy head must keep tail mass for outliers and heavy tails it
  cannot rule out. Added a robust estimate sqrt(IQR_noisy^2 - IQR_latent^2) / 1.349 against the true robust
  scale; on the same toy checkpoint its median log ratio is -0.17 to +0.10 for Gaussian noise (n = 5, 50).
  Two robust probe features were added. A superseded first sanity eval (120245, commit 8082d7b) was not kept.
- The martingale table's `z_mean` treats the 64 probe points as independent, which overstates z; read
  `bias_abs` against `mc_se` instead.
