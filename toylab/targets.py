"""Toy target densities — promoted from N0.

Each target is *fully known* (we use it to grade models that never see it) and exposes
the two capabilities the curriculum keeps separate:

    sample(key, n) -> (n,)     the "Alice" power: draw fresh samples
    log_prob(x)    -> (n,)     the "Bob" power: evaluate log p(x) anywhere
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random


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
