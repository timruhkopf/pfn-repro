---
experiment: eval_alpha_large
sweeps: []
---
# alpha-PFN: amortized Entropy Search on the BNN prior

## Question

See the description at the top of `notebooks/04_2026-10-21_eval_alpha_large.ipynb`; not retyped here.

## §1 Sanity (2026-10-21): gates and numbers as originally reported

- Sweep `sanity_20261021-135935` (ulysses, commit 52bf85a); snapshot in `results/eval_alpha_large/`. A
  rerun at another commit (`sanity_20261021-134916`) gave identical gates, panels and agreement tables
  (deterministic on one machine).
- All 24 gate criteria pass, read from the checkpoints' own histories. Stage-2 ρ at n = 20 is 0.60 / 0.47 /
  0.27 / 0.17 (d = 1, 2, 4, 6). Conditioning on x* and f* gains 0.12–0.29 nats at n = 5, 0.05–0.07 at
  n = 20. Agreement: Spearman 0.43–0.70 with EI, −0.10 to 0.39 with std. All of these match pfn-acqpolicy
  notebook 09.
- 1-d: the alpha-PFN behaves like Entropy Search (it ignores uncertain regions that can't hold the
  minimum; function 2 at n = 10 peaks beside x*). A single-draw α̃ is noisy and can be negative; the
  model learns its expectation. Details in `notebooks/04_*`.
