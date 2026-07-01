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


# --- 2D (promoted at N3) ---------------------------------------------------
def density_2d_grid(log_prob_fn, lim=7.0, n=300):
    """Evaluate a 2D log_prob on an n×n square grid. Returns (pdf, extent).

    pdf is (n,n) with pdf[i,j] at (x1=axis[j], x2=axis[i]) — imshow(origin='lower')
    convention. extent is [-lim, lim, -lim, lim] for passing straight to imshow.
    """
    axis = jnp.linspace(-lim, lim, n)
    X1, X2 = jnp.meshgrid(axis, axis)
    pts = jnp.stack([X1.ravel(), X2.ravel()], axis=1)
    pdf = jnp.exp(log_prob_fn(pts)).reshape(n, n)
    return pdf, [-lim, lim, -lim, lim]


def plot_density_2d(log_prob_fn, lim=7.0, n=300, ax=None, title=None, cmap="magma"):
    """Heatmap of a 2D density log_prob_fn: (m,2)->(m,). Returns the Axes."""
    pdf, extent = density_2d_grid(log_prob_fn, lim=lim, n=n)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(np.asarray(pdf), origin="lower", extent=extent, cmap=cmap, aspect="equal")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title(title or "p(x1, x2)")
    return ax


def plot_samples_2d(samples, lim=7.0, ax=None, title=None, color="steelblue", s=5):
    """Scatter of 2D samples (n,2). Returns the Axes."""
    samples = np.asarray(samples)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(samples[:, 0], samples[:, 1], s=s, alpha=0.3, color=color)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title(title or f"samples (N={len(samples)})")
    return ax
