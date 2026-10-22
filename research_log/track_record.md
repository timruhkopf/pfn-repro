# Track record

**Append-only.** New entries go at the bottom; old entries are never edited, a later entry corrects an
earlier one explicitly. Times are local. Each entry names the commit it refers to, what was done, what
was **verified and how**, and numbers with the snapshot or artifact they come from.

Autonomous agent sessions start with their mandate (goal, deadline, which launches are allowed) and end
with deliverables and open questions.

## Entries

### 2026-10-21 14:30 — port done, launching the reproduction checks (012178a)

- Mandate (Tim): a standalone, reusable PFN repo from research-template: pfn_large and both alpha-PFN
  checkpoints, the PFN / alpha-PFN analysis notebooks (with 1-d PPD heatmaps), the BNN and MF priors
  (with 1-d notebooks), architecture, bar distribution, training pipeline; a dependency guide.
- Verified locally: all four checkpoints load and match their sha256; 32 unit tests (golden outputs,
  surrogate invariances, exact resume, every prior x surrogate trains); ruff and pyright clean.
- ulysses: `just remote-setup` (git over ssh, cu126 venv); checkpoints copied from
  pfn-acqpolicy/models and verified there.
- Next: `eval_pfn_large` and `eval_alpha_large` as sanity sweeps on ulysses (detached, CPU), then compare
  with the original eval_pfn_large artifacts.

### 2026-10-21 15:15 — reproduction checks done (52bf85a)

- eval_pfn_large: the laptop run (the original's machine and torch) is **bit-identical** to pfn-acqpolicy's
  run c091bb58 (30,720 per-function rows). ulysses differs at the near-tie level (summary ≤ 5e-5). Log
  entry §1, notebook 03.
- eval_alpha_large: all 24 gate criteria pass with the original values; agreement and gains match notebook 09.
  Log entry §1, notebook 04.
- Notebooks 01 (BNN prior) and 02 (MF prior) are written and executed. I checked every observation
  against the figures and corrected two that the figures contradicted (01: "no constant functions";
  02: "monotone curves").
- Found and fixed along the way:
  - template v0.3.2/v0.3.3: pre-commit 4.6.1 needs git ≥ 2.31; just ran recipes with dash
    (''); remote-ps matched itself; notebook cell ids; default_stages;
  - here: the density tables were too large to commit (float16, coarser grid).

### 2026-10-21 14:25 — correction of the two entries above; TabPFN pilot scheduled

- **Correction:** the two entries above carry wrong times (14:30 and 15:15 were written without reading
  the clock). The reproduction checks were launched at 13:49 (sweep ids `*_134915`, `*_134916`) and
  finished around 14:10. Their content stands.
- Mandate (Tim, 14:15): train TabPFN on the BNN prior tonight on ulysses. Issue #1, branch
  `exp/1-tabpfn-bnn`.
- Timing probe on ulysses (bf16, batch 64, n_train ≤ 100, 256 test points): d_model 128/6 layers
  ≈ 0.20 s/step; 192/8 ≈ 0.39 s/step; 256/8 runs out of memory. Chosen: 192/8, 32 thinking rows. Cost:
  one night, ~11.5 GPU-h, with the cosine schedule fitted to the 07:30 deadline.
- Template v0.4.0 adds `--start-at` (plan and freeze now, start later), used for this launch.

### 2026-10-21 20:12 — TabPFN pilot start check: not verifiable (ulysses unreachable)

Scheduled check of `runs/tabpfn_bnn/pilot_20261021-142330` (planned start 20:00, detached `--start-at`).
`just remote-ps` and ssh both time out during the banner exchange. The jump host
`ssh2.ai.uni-hannover.de:22` has refused connections from the laptop since about 18:40, while other
internet works, so I couldn't see whether the run started. The run doesn't need a connection (it was
planned and frozen at 14:23). Not relaunched. Next: check queue.log and 0/stdout.log once the gateway answers.

### 2026-10-22 09:50 — TabPFN pilot: started and finished (69003e1)

ulysses reachable again at 09:48. `runs/tabpfn_bnn/pilot_20261021-142330/queue.log` (read over ssh):
"waiting until 2026-10-21 20:00:00", then "member 0: DONE (rc=0)" at 2026-10-22 06:36:34, before the
07:30 deadline. `0/stdout.log`: "372 ms/step: n_steps set to 100115"; last step 100114 (38191 s),
"saved checkpoint to .../0/checkpoint.pt". `just remote-ps`: nothing running; GPU idle. This corrects the
20:12 entry (not verifiable then). Not yet pulled or evaluated.
