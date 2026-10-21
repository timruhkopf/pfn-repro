---
experiment: eval_pfn_large
sweeps: []
---
# PFN quality: pfn_large vs the smoke PFN

## Question

See the description at the top of `notebooks/03_2026-10-21_eval_pfn_large.ipynb`; not retyped here.

## §1 Sanity (2026-10-21): the port reproduces the original exactly on the original machine

- Sweeps (stage sanity, commit 52bf85a): `sanity_20261021-135936` on the laptop and
  `sanity_20261021-135934` on ulysses; snapshots in `results/eval_pfn_large/`. The original's tables are in
  `results/eval_pfn_large/reference_pfn-acqpolicy/` (pfn-acqpolicy run c091bb58, commit 58b6e2b).
- **Laptop (the original's machine, torch 2.14.1+cpu): bit-identical** per-function values for all 30,720
  rows; summary bounds within 1e-7. The pre-registered criterion (all summary metrics within 1e-5) holds.
- **ulysses (cu126 build, other CPU):** summary means within 5e-5. Per function, NLL differs by up to
  0.013 for 4.6% of the functions; a few coverage, rank and argmin values flip. These are near-ties
  decided by floating point. Cross-machine tolerance: ~1e-4 on summaries.
- An earlier ulysses sweep (`sanity_20261021-134915`, commit 07c2592) gave the same metrics. It was
  superseded only because its density tables were too large to commit (now float16, coarser grid).
- Results as in the original report: NLL gain over smoke at n = 20 is +0.85 / +1.08 / +0.84 / +0.59 /
  +0.46 / +0.28 nats (d = 1…6); 90% coverage 0.88–0.90. New: the 1-d panels show a bimodal PPD away
  from data at small n (`notebooks/03_*`).
