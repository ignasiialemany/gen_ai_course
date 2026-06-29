"""Standard plots — promoted from N0."""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_samples_vs_truth(samples, log_prob_fn, n_bins=60, x_range=(-6, 6), title=None, ax=None):
    """The eye test: density-normalized histogram of samples overlaid on the true curve.

    samples      : (n,) array of model (or real) samples
    log_prob_fn  : callable x -> log p(x), e.g. target.log_prob
    Returns the matplotlib Axes (so callers can keep styling or subplot).
    """
    grid = jnp.linspace(x_range[0], x_range[1], 400)
    true_p = jnp.exp(log_prob_fn(grid))
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.asarray(samples), bins=n_bins, range=x_range, density=True,
            alpha=0.5, color="steelblue", label=f"samples (N={len(samples)})")
    ax.plot(np.asarray(grid), np.asarray(true_p), color="black", lw=2, label="true p(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.legend()
    ax.set_title(title or "eye test: samples vs truth")
    return ax
