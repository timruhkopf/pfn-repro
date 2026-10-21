# 0002: Port of the multi-fidelity prior from Meta_FTPFN

**Context.** The MF (learning-curve) prior lives in Meta_FTPFN in several versions: `main` and `dev`
(`src/ppfn/dataset/prior/`), and the latest refactor on branch `claude-init`
(`archive/src/ppfn/prior/multi_fidelity/mf_ftpfn_refactor/`, 2026-08-27). The refactor's `MLP.forward`
raised `NotImplementedError("check if the original mlp also did not use the normalizer!")`. In `dev`,
the same line computed `self.normalizer(x)` and discarded the result.

**Decision.** Port the `claude-init` refactor (the newest; it equals `dev` plus fixes) into
`priors/mf/`, verbatim except for:
1. **No input normalization** in the BNN forward pass, as in the original it reproduces
   (PFNs4HPO `hpo_lc_pfn_bopfn_broken.py::MLP.forward` has none). This also removes the `pfns4hpo`
   dependency.
2. **The latent ECDF is cached per settings** (num_inputs, num_outputs, sample count) in
   `<data root>/priors/`, not once per class in the package directory. Previously the first num_inputs
   used in a process silently served all others.
3. **It is generated under its own fixed seed** and restores the caller's RNG state, and `MFPrior`
   seeds numpy's and torch's global RNGs from its own generator. Together these make the prior
   reproducible from its seed (tested), which it wasn't: the MLP draws its weights from torch's global
   RNG, and only numpy was seeded.
4. `MFPrior` (new, `priors/mf/episodes.py`) wraps the task and the allocation prior behind the BNN
   prior's episode interface: inputs `[config, fidelity]`, `y = 1 - performance` (minimization).

The regeneration is scripted (`scripts/port_mf_prior.py`), so each change is visible as a string
replacement.

**Consequences.** Meta_FTPFN's trained models used the same family, but none of them is registered
here. Training a PFN on `prior=mf` starts from scratch.
