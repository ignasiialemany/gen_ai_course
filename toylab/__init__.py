"""toylab — thin shared infrastructure for the 1D generative-AI curriculum.

Promoted from N0. **Hard rule (see CONTEXT.md):** no generative *method* lives here —
no training objective, no model, no sampler-*of-a-model*. Only the un-interesting,
identical-across-notebooks scaffolding:

    targets  — toy densities we secretly know (each exposes sample(key, n) and log_prob(x))
    viz      — standard plots (the eye test)
    metrics  — exact 1D evaluation rulers (KL, Wasserstein), both grid- and sample-based

A method always stays *visible in its notebook*; only the ruler and the playground move here.
"""
from . import targets, viz, metrics
from .targets import Mixture1D, two_bumps
from .viz import plot_samples_vs_truth
from .metrics import to_hist, kl_divergence, kl_grid, wasserstein1, wasserstein1_cdf

__all__ = [
    "targets", "viz", "metrics",
    "Mixture1D", "two_bumps",
    "plot_samples_vs_truth",
    "to_hist", "kl_divergence", "kl_grid", "wasserstein1", "wasserstein1_cdf",
]
