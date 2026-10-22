---
experiment: tabpfn_bnn
sweeps: []
---
# TabPFN-v2-style surrogate on the BNN prior (overnight pilot)

## Question

See the description at the top of `notebooks/05_2026-10-21_tabpfn_bnn.ipynb`; not retyped here.

## §1 Pilot ran; registered as `tabpfn_bnn_pilot` (2026-10-22, 11:00)

- **What ran.** `runs/tabpfn_bnn/pilot_20261021-142330` (6567285, clean), `--start-at 20:00`, deadline
  07:30. Calibration: 372 ms/step → n_steps 100,115 (margin 0.9); DONE at 06:36 after 38,192 s, step
  100,114, lr annealed to 0. One CUDA caching-allocator OOM warning at 20:00:08 (a 3.1 GB request with
  2.8 GB free, during the first steps); training continued without restarts (run.json: restarts 0).
- **Validation** (`final/nll_val_d<d>_n<n>`, run.json summary; the snapshot `results/tabpfn_bnn/`): NLL
  on the fixed validation sets falls monotonically and has flattened by 90k–100k steps, e.g.
  d = 1: −1.88 / −3.39 / −3.83 at n = 5 / 20 / 50; d = 6: −0.54 / −1.08 / −1.54. No NaN or inf in the log.
- **Registered** as `tabpfn_bnn_pilot` (sha256 d4141d53…, 60 MB; provenance in the registry: commit,
  sweep, MLflow run 315b763c6ca040cbb5e911f454fff3dc, step), with golden outputs; v0.2.0.
- **Not yet answered**: the question of notebook 05 (validation against `pfn_large` at equal steps and
  overall). Next: `eval_pfn` with `task.models=[tabpfn_bnn_pilot,pfn_large]`, then the notebook.
