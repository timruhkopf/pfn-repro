# Checkpoints

The registry is `src/pfn_repro/checkpoints.py`: name → file, sha256, kind, context range, provenance.
A file is loaded only after its sha256 matches.

| Name | Kind | File | Size | Trained | Use within |
|---|---|---|---|---|---|
| `pfn_smoke` | pfn | `pfn_variable_xdim_smoke.pt` (in the package) | 0.6 MB | AnytimeAcquisition d48bf61, 30k steps | n ≤ 20 |
| `pfn_large` | pfn | `pfn_large.pt` | 76 MB | pfn-acqpolicy 0b5683e, 1.5M steps (~24.5 h, RTX 3090) | n ≤ 100 |
| `alpha_conditioned_large` | conditioned | `es_conditioned_large.pt` | 76 MB | pfn-acqpolicy 4d92fd5, 156,963 steps, from pfn_large | n ≤ 100 |
| `alpha_pes_large` | alpha | `es_alpha_pes_large.pt` | 76 MB | pfn-acqpolicy 4d92fd5, 186,299 steps (PES) | n ≤ 100 |
| `bnn_ecdf_d6` (ECDF) | — | `bnn_ecdf_d6_86acf732c579ed0d.pt` (in the package) | 8 KB | AnytimeAcquisition | every BNN evaluation prior |

All are trained on the BNN prior (`configs/prior/bnn.yaml`, x_dim 6, variable d from 1). y is on the ECDF
scale; reuse `load_ecdf()` / `make_prior()`, or y lives on a different scale than the models expect.

## Where the large files are

They are not in git (76 MB each).

| Machine | Location |
|---|---|
| ulysses | `~/PycharmProjects/pfn-acqpolicy/models/` (where they were trained), copied into `~/PycharmProjects/pfn-repro/checkpoints/` |
| laptop | `~/PycharmProjects/pfn-acqpolicy/models/`, symlinked from `pfn-repro/checkpoints/` |

Point `PFN_REPRO_CKPT_ROOT` at the directory (default: `<repo>/checkpoints`, for a source checkout).
`pfn_repro.checkpoints.available()` shows which names are present.

To copy them to a new machine: `rsync -av ulysses:PycharmProjects/pfn-repro/checkpoints/*.pt <dir>/`.
The sha256 check then verifies the copy.

## Registering a new checkpoint

1. Train it as a tracked run (`just sweep … experiment=…`). The file is `<member dir>/checkpoint.pt`.
2. Copy it to the checkpoint directory under a **new** name. Never reuse a registered name.
3. Add a `Checkpoint(...)` entry: file, `sha256sum`, kind, training context range, a one-line
   description, and `source` (repo, commit, sweep id, step).
4. Add its golden outputs to `tests/unit/test_checkpoints.py` (compute them once, paste them in).
5. Evaluate it (`experiment=eval_pfn_large task.models=[<name>,pfn_large]`), snapshot, add to the notebook.
6. Release a minor version with a CHANGELOG entry (docs/dependency_guide.md).
