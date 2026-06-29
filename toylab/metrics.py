"""Exact 1D evaluation metrics — promoted from N0.

Two regimes for each ruler:

  * grid versions   (kl_grid, wasserstein1_cdf)  take *densities* on a shared grid.
    Since in toy-land we secretly know p, these are the "exact" rulers.
  * sample versions (to_hist + kl_divergence, wasserstein1) take *samples*.
    These are the estimators you'd actually use when you only have a bag of numbers.

The reason this whole module is cheap and exact: we are in 1D. KL is a sum over bins,
W1 is a sort. Both of these break in high dimensions (see N0's exit questions).
"""
import jax.numpy as jnp


# --- binning ---------------------------------------------------------------
def to_hist(samples, edges):
    """Bin samples into a NORMALIZED probability vector over the given bin edges."""
    counts, _ = jnp.histogram(samples, bins=edges)
    return counts / counts.sum()


# --- Ruler #1: KL divergence ----------------------------------------------
def kl_divergence(p, q, eps=1e-12):
    """KL(p || q) for two probability vectors:  sum_i p_i log(p_i / q_i).

    eps nudges off exact zeros so log is finite. Set eps=0 to expose the landmine:
    a single bin with q_i=0 where p_i>0 sends KL to +inf.
    """
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return jnp.sum(p * jnp.log(p / q))


def kl_grid(pdf_a, pdf_b, grid):
    """KL(a || b) from two densities sampled on a shared grid (the 'exact' toy KL).

    Bins with pdf_a == 0 contribute 0 (since lim p log p = 0); a bin with pdf_b == 0
    where pdf_a > 0 yields +inf, on purpose.
    """
    dx = grid[1] - grid[0]
    pa = pdf_a * dx
    pb = pdf_b * dx
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    mask = pa > 0
    return jnp.sum(jnp.where(mask, pa * jnp.log(pa / jnp.where(pb > 0, pb, 1.0)), 0.0))


# --- Ruler #2: Wasserstein-1 ----------------------------------------------
def wasserstein1(a, b):
    """W1 between two EQUAL-size 1D sample sets: sort, pair in order, average the gaps.

    Valid because in 1D the optimal transport plan is monotone (no crossings).
    """
    return jnp.mean(jnp.abs(jnp.sort(a) - jnp.sort(b)))


def wasserstein1_cdf(pdf_a, pdf_b, grid):
    """W1 between two densities on a shared grid = area between their CDFs."""
    dx = grid[1] - grid[0]
    Fa = jnp.cumsum(pdf_a) * dx
    Fb = jnp.cumsum(pdf_b) * dx
    Fa = Fa / Fa[-1]                       # make them proper CDFs (end at 1)
    Fb = Fb / Fb[-1]
    return jnp.sum(jnp.abs(Fa - Fb)) * dx
