"""toylab — thin shared infrastructure for the 1D generative-AI curriculum.

Promoted from N0. **Hard rule (see CONTEXT.md):** no generative *method* lives here —
no training objective, no model, no sampler-*of-a-model*. Only the un-interesting,
identical-across-notebooks scaffolding:

    targets  — toy densities we secretly know (each exposes sample(key, n) and log_prob(x))
    viz      — standard plots (the eye test)
    metrics  — exact 1D evaluation rulers (KL, Wasserstein) + 2D grid approximations

A method always stays *visible in its notebook*; only the ruler and the playground move here.
"""
from . import targets, viz, metrics
from .targets import Mixture1D, two_bumps, Mixture2D, two_moons
from .viz import (plot_samples_vs_truth, plot_density_2d, plot_samples_2d, density_2d_grid)
from .metrics import (to_hist, kl_divergence, kl_grid, wasserstein1, wasserstein1_cdf,
                      kl_grid_2d)

__all__ = [
    "targets", "viz", "metrics",
    "Mixture1D", "two_bumps", "Mixture2D", "two_moons",
    "plot_samples_vs_truth", "plot_density_2d", "plot_samples_2d", "density_2d_grid",
    "to_hist", "kl_divergence", "kl_grid", "wasserstein1", "wasserstein1_cdf", "kl_grid_2d",
]
