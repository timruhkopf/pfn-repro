"""Reproducible PFN surrogates, priors and amortized acquisitions, with pinned checkpoints, for reuse as a
dependency: priors (BNN, multi-fidelity), surrogates (PFNs4BO-style PFN, TabPFN-v2-style), acquisitions
(standard, alpha-PFN), objectives (bar distribution)."""

from importlib.metadata import version

__version__ = version("pfn-repro")
