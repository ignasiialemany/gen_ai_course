"""Toy target densities — promoted from N0.

Each target is *fully known* (we use it to grade models that never see it) and exposes
the two capabilities the curriculum keeps separate:

    sample(key, n) -> (n,)     the "Alice" power: draw fresh samples
    log_prob(x)    -> (n,)     the "Bob" power: evaluate log p(x) anywhere
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

LOG2PI = jnp.log(2 * jnp.pi)


def _norm_logpdf(x, mu, sd):
    """log N(x; mu, sd), broadcasting over any shapes."""
    return -0.5 * LOG2PI - jnp.log(sd) - 0.5 * ((x - mu) / sd) ** 2


@dataclass(frozen=True)
class Mixture1D:
    """A 1D mixture of Gaussians, fully specified by three vectors of equal length K."""

    weights: jnp.ndarray  # (K,) mixing weights, must sum to 1
    means: jnp.ndarray    # (K,) component centers
    stds: jnp.ndarray     # (K,) component widths

    def log_prob(self, x):
        """log p(x), elementwise.  x: (n,) -> (n,).  p(x) = sum_k w_k N(x; mean_k, std_k)."""
        x = jnp.atleast_1d(x)[:, None]                          # (n, 1) to broadcast over K bumps
        log_components = (                                      # (n, K)
            jnp.log(self.weights)[None, :]
            - 0.5 * jnp.log(2 * jnp.pi)
            - jnp.log(self.stds)[None, :]
            - 0.5 * ((x - self.means[None, :]) / self.stds[None, :]) ** 2
        )
        return jax.scipy.special.logsumexp(log_components, axis=1)   # log-sum the K bumps -> (n,)

    def sample(self, key, n):
        """Draw n samples: pick a component by weight, then sample that Gaussian."""
        key_pick, key_noise = random.split(key)
        comp = random.choice(key_pick, self.weights.shape[0], shape=(n,), p=self.weights)
        z = random.normal(key_noise, shape=(n,))
        return self.means[comp] + self.stds[comp] * z


# The canonical N0 toy target: two equal, well-separated bumps at +/-2.
two_bumps = Mixture1D(
    weights=jnp.array([0.5, 0.5]),
    means=jnp.array([-2.0, 2.0]),
    stds=jnp.array([0.7, 0.7]),
)


# ---------------------------------------------------------------------------
# 2D — promoted at N3 (the first notebook that earns a second coordinate; the
# 1D-default discipline reserves 2D for mechanisms that *need* >=2 coords, and
# autoregressive factorization is one of them).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mixture2D:
    """A 2D mixture of axis-aligned (diagonal-covariance) Gaussians.

        weights : (K,)     mixing weights, sum to 1
        means   : (K, 2)   component centers
        stds    : (K, 2)   per-axis component widths (diagonal covariance)

    Diagonal covariance is the whole point for N3: each component factorizes as
    p_k(x1,x2)=N(x1)N(x2), which hands us **closed-form** marginal p(x1) and
    conditional p(x2|x1). Those let us plot the *true* conditional the
    autoregressive model is trying to learn, and grade it exactly.
    """

    weights: jnp.ndarray
    means: jnp.ndarray
    stds: jnp.ndarray

    def _comp_log_prob(self, X):
        """log of each component density. X: (n,2) -> (n,K)."""
        X = jnp.atleast_2d(X)
        x1 = X[:, 0][:, None]                                   # (n,1)
        x2 = X[:, 1][:, None]
        lp1 = _norm_logpdf(x1, self.means[None, :, 0], self.stds[None, :, 0])
        lp2 = _norm_logpdf(x2, self.means[None, :, 1], self.stds[None, :, 1])
        return jnp.log(self.weights)[None, :] + lp1 + lp2       # (n,K)

    def log_prob(self, X):
        """log p(x1,x2), elementwise.  X: (n,2) -> (n,)."""
        return jax.scipy.special.logsumexp(self._comp_log_prob(X), axis=1)

    def sample(self, key, n):
        """Draw n samples -> (n,2): pick a component by weight, then its Gaussian."""
        key_pick, key_noise = random.split(key)
        comp = random.choice(key_pick, self.weights.shape[0], shape=(n,), p=self.weights)
        z = random.normal(key_noise, shape=(n, 2))
        return self.means[comp] + self.stds[comp] * z

    # --- the closed forms that make this a *teaching* target (N3) ---
    def marginal_x1_log_prob(self, x1):
        """log p(x1) = log sum_k w_k N(x1; m_k1, s_k1)  (x2 integrated out).  (n,)->(n,)."""
        x1 = jnp.atleast_1d(x1)[:, None]                        # (n,1)
        lp = jnp.log(self.weights)[None, :] + _norm_logpdf(
            x1, self.means[None, :, 0], self.stds[None, :, 0])
        return jax.scipy.special.logsumexp(lp, axis=1)

    def cond_x2_given_x1_log_prob(self, x2, x1):
        """log p(x2 | x1) for a SCALAR x1, evaluated at vector x2 -> (n,).

        A mixture over k with responsibilities r_k(x1) proportional to
        w_k N(x1; m_k1, s_k1) and components N(x2; m_k2, s_k2). It is **bimodal**
        exactly when two components with different m_k2 both light up at this x1 —
        which is what a single-Gaussian conditional can never reproduce.
        """
        x2 = jnp.atleast_1d(x2)[:, None]                        # (n,1)
        log_r = jnp.log(self.weights) + _norm_logpdf(
            x1, self.means[:, 0], self.stds[:, 0])              # (K,)
        log_r = log_r - jax.scipy.special.logsumexp(log_r)      # normalize the responsibilities
        lp2 = _norm_logpdf(x2, self.means[None, :, 1], self.stds[None, :, 1])  # (n,K)
        return jax.scipy.special.logsumexp(log_r[None, :] + lp2, axis=1)


def _two_moons_params(M=10, scale=3.0, noise=0.26):
    """Place M diagonal Gaussians along each of two semicircle arcs (-> 2*M components).

    Upper arc  (cosθ,  sinθ),       θ∈[0,π]  spans x∈[-1, 1]
    Lower arc  (1-cosθ, -sinθ+0.5), θ∈[0,π]  spans x∈[ 0, 2]
    The arcs overlap in x∈[0,1], so a vertical slice there crosses BOTH arcs and the
    conditional p(x2|x1) is *bimodal*; outside it (upper-only / lower-only) it is
    *unimodal*. Centered on x and scaled into roughly [-5,5]^2.
    """
    th = np.linspace(0, np.pi, M)
    upper = np.stack([np.cos(th), np.sin(th)], axis=1)
    lower = np.stack([1 - np.cos(th), -np.sin(th) + 0.5], axis=1)
    pts = np.concatenate([upper, lower], axis=0)
    pts[:, 0] -= 0.5                                            # center x around 0
    pts *= scale
    K = pts.shape[0]
    return (jnp.full((K,), 1.0 / K), jnp.asarray(pts, dtype=jnp.float32),
            jnp.full((K, 2), noise * scale, dtype=jnp.float32))


# The canonical 2D toy target (CONTEXT.md): a closed-form "two moons". Bimodal
# conditional p(x2|x1) in the middle, unimodal at the edges — the structure that
# makes autoregressive factorization worth doing.
two_moons = Mixture2D(*_two_moons_params())
