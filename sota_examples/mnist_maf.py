"""
MNIST masked autoregressive flow (MAF) — the D=784 version of your N3 build.

You hand-assembled the D=2 masked network in N3: `theta1` was the bias-only
output row, `hyper(phi, x1)` was the degree-1 path. Here you build the real
thing: ONE shared masked MLP (MADE) that emits the flow parameters for all
784 pixel-conditionals in a single forward pass.

The dictionary (N3 §5):
    coordinate x_i            -> pixel i (raster order)
    hypernetwork MLP(x_<i)    -> the masked MLP (one pass, all conditionals)
    conditional flow q(x_i|.) -> per-pixel affine z_i = (x_i - mu_i) * exp(-alpha_i)
    chain-rule sum (C3.1)     -> log q(x) = sum_i [ log N(z_i) - alpha_i ]
    ancestral sampling (§4)   -> D sequential passes through the SAME network

What is provided (boring plumbing): MNIST download/cache, dequantization +
logit transform, the bits-per-dim ruler, a diagonal-Gaussian baseline (your
N1-style ruler), the training harness, and autotests for your masks.

What you implement (every interesting idea) — the TODOs, in order:
    TODO 1  assign_degrees   — tag every unit with "max input index it may see"
    TODO 2  build_masks      — the three connectivity rules, as 0/1 matrices
    TODO 3  made_forward     — one masked pass: x (d,) -> theta (d, 2)
    TODO 4  log_prob         — chain rule + change of variables (C3.1 + C3.2)
    TODO 5  sample           — ancestral: one coordinate lands per pass (§4)
    TODO 6  (stretch) swap the affine head for your N2 K-bump monotone
            transform (+ bisection to invert), and/or stack several MAF
            layers with order reversal between them.

Run stages (in this order):
    python sota_examples/mnist_maf.py masks      # autotest your TODO 1-3
    python sota_examples/mnist_maf.py moons      # your net at D=2 vs N3's numbers
    python sota_examples/mnist_maf.py baseline   # the ruler (no TODOs needed)
    python sota_examples/mnist_maf.py mnist      # the real thing

Success criteria:
    masks    both Jacobian tests pass — z is triangular (C3.2, now a unit test)
             and theta_i is blind to x_{>=i}.
    moons    PREDICT-THEN-RUN: one affine-conditional layer at D=2 — where does
             grid KL land vs N3's flow (0.036) and N3's break-it Gaussian
             conditional (0.070)? Commit before running. Then beat it via TODO 6.
    mnist    val bits/dim clearly below the diagonal-Gaussian baseline, and the
             samples PNG shows digit-ish strokes, not static.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, vmap

import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for toylab (moons stage)

OUT_PER_DIM = 2          # affine head: (mu, alpha) per coordinate
LOG2PI = jnp.log(2 * jnp.pi)
LAM = 1e-6               # logit-transform squeeze (MAF paper's MNIST setting)
DATA_DIR = Path(__file__).resolve().parent / "data"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def log_base(z: jnp.ndarray) -> jnp.ndarray:
    return -0.5 * LOG2PI - 0.5 * z**2


# ======================================================================
# Provided: data — download, dequantize, logit-transform, bits/dim ruler
# ======================================================================

def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Raw MNIST as uint8 arrays: train (60000, 784), test (10000, 784)."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "mnist.npz"
    if not path.exists():
        print(f"downloading MNIST -> {path} ...")
        urllib.request.urlretrieve(MNIST_URL, path)
    with np.load(path) as f:
        x_tr = f["x_train"].reshape(-1, 784)
        x_te = f["x_test"].reshape(-1, 784)
    return x_tr, x_te


def to_logit_space(x_uint8: np.ndarray, key: jax.Array) -> jnp.ndarray:
    """{0..255} -> dequantize -> [0,1) -> logit. A flow needs a *continuous*
    density; discrete pixels have none (point masses). Uniform dequantization
    p=(x+u)/256 makes the density exist; the logit y=log(s/(1-s)) with
    s = LAM + (1-2*LAM)*p unsquashes [0,1] onto R so Gaussian-ish tails fit."""
    u = random.uniform(key, x_uint8.shape)
    p = (jnp.asarray(x_uint8, jnp.float32) + u) / 256.0
    s = LAM + (1 - 2 * LAM) * p
    return jnp.log(s) - jnp.log1p(-s)


def logit_logdet(y: jnp.ndarray) -> jnp.ndarray:
    """Per-image log|dy/dp| — the change-of-variables term of the logit
    preprocessing itself, needed to report densities in PIXEL space.
    dy/dp = (1-2*LAM) / (s*(1-s)) with s = sigmoid(y).  Shape (n, d) -> (n,)."""
    s = jax.nn.sigmoid(y)
    return jnp.sum(jnp.log1p(-2 * LAM) - jnp.log(s) - jnp.log1p(-s), axis=-1)


def bits_per_dim(log_q_y: jnp.ndarray, y: jnp.ndarray) -> float:
    """Model gives log q(y) in logit space; convert to bits per pixel-dim.
    log q_pixel(p) = log q(y) + log|dy/dp|, and the dequantized density on
    [0,1) relates to the discrete pixels via the extra log(256) per dim."""
    d = y.shape[-1]
    log_q_pixel = log_q_y + logit_logdet(y)
    return float(-(jnp.mean(log_q_pixel)) / (d * jnp.log(2.0)) + 8.0)


def from_logit_space(y: jnp.ndarray) -> jnp.ndarray:
    """Inverse preprocessing, for rendering samples: R -> [0,1]."""
    s = jax.nn.sigmoid(y)
    return jnp.clip((s - LAM) / (1 - 2 * LAM), 0.0, 1.0)


# ======================================================================
# Provided: parameter init (unmasked weights; masks are applied by YOU
# in made_forward). Final layer starts tiny => mu~0, alpha~0 => z~x,
# i.e. the flow begins near the identity — N2's FLOW0 lesson.
# ======================================================================

def init_made(key: jax.Array, d: int, hidden: tuple[int, ...]) -> list[dict]:
    sizes = [d, *hidden, d * OUT_PER_DIM]
    params = []
    for i, (f_in, f_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, sub = random.split(key)
        scale = (0.001 if i == len(sizes) - 2 else 1.0) / jnp.sqrt(f_in)
        params.append({"W": scale * random.normal(sub, (f_out, f_in)),
                       "b": jnp.zeros(f_out)})
    return params


# ======================================================================
# =====[ TODO 1 ]=======================================================
# ======================================================================

def assign_degrees(key: jax.Array, d: int, hidden: tuple[int, ...]) -> list[jnp.ndarray]:
    """Tag every unit with a degree = the largest input index it is allowed
    to depend on.

    Return a list of integer arrays: [input_degrees, hidden1_degrees, ...]
      - input degrees are just 1..d (raster order),
      - each hidden unit gets a degree in {1, ..., d-1}   (why is d itself
        useless for a hidden unit? see N3 §5: no output may see input d).
        Random uniform or the cyclic (k % (d-1)) + 1 both work.
    (Output degrees need no array here: block i has degree i, see TODO 2.)
    """
    input_degrees = jnp.arange(1, d+1)
    out = [input_degrees]
    for h in hidden:
        key, sub = random.split(key)
        out.append(random.randint(sub, (h, ), 1, d))
    return out



# ======================================================================
# =====[ TODO 2 ]=======================================================
# ======================================================================

def build_masks(degrees: list[jnp.ndarray], d: int) -> list[jnp.ndarray]:
    """One 0/1 matrix per weight matrix, same shape as params[i]["W"]
    (f_out, f_in). The three rules from N3 §5 (biases are never masked):

      input  -> hidden : allow W[k, j]  iff  j <= m(k)
      hidden -> hidden : allow W[k',k]  iff  m(k') >= m(k)
      hidden -> output : allow          iff  m(k) < i   (STRICT — block i
                          must not see x_i itself; that's what keeps the
                          Jacobian triangular with the (x_i - mu_i) term
                          carrying the diagonal)

    Output layout: unit for (coordinate i, param j) sits at row i*OUT_PER_DIM+j
    — i.e. the d output blocks of size OUT_PER_DIM are tiled in order, so
    tile/repeat the output degrees accordingly.
    """
    out = [] 
    for i in range(1, len(degrees)):
        in_dimensions = len(degrees[i-1])
        out_dimensions = len(degrees[i])
        mask = jnp.zeros((out_dimensions, in_dimensions))
        mask = (degrees[i-1][None, :] <= degrees[i][:, None]).astype(jnp.int32)
        out.append(mask)

    output_mask = jnp.zeros((d*OUT_PER_DIM, len(degrees[-1])))
    block = jnp.repeat(jnp.arange(d), OUT_PER_DIM)
    output_mask = (degrees[-1][None, :] <= block[:, None]).astype(jnp.int32)
    out.append(output_mask)

    return out  

# ======================================================================
# =====[ TODO 3 ]=======================================================
# ======================================================================

def made_forward(params: list[dict], masks: list[jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """ONE masked pass: x (d,) -> theta (d, OUT_PER_DIM).

    Each layer: h = act((W * mask) @ h + b). Use a nonlinearity you like
    on hidden layers (relu/tanh), NO nonlinearity on the final layer.
    Reshape the final (d*OUT_PER_DIM,) vector to (d, OUT_PER_DIM):
    theta[i] = (mu_i, alpha_i), a function of x_{<i} only — IF your masks
    are right. Stage `masks` will put that claim on trial.
    (Batching is done for you: log_prob vmaps this over rows.)
    """
    h = x
    for i in range(len(params)-1):
        h = jax.nn.relu((masks[i] * params[i]["W"]) @ h + params[i]["b"])
    h = (masks[-1] * params[-1]["W"]) @ h + params[-1]["b"]
    return h.reshape(-1, OUT_PER_DIM)

# ======================================================================
# =====[ TODO 4 ]=======================================================
# ======================================================================

def log_prob(params: list[dict], masks: list[jnp.ndarray], X: jnp.ndarray) -> jnp.ndarray:
    """Exact log q for a batch X (n, d) -> (n,).   C3.1 + C3.2, vectorized:

        theta = made_forward(...)            (vmap over the batch)
        z_i   = (x_i - mu_i) * exp(-alpha_i)
        log q = sum_i [ log_base(z_i) - alpha_i ]

    Why "- alpha_i"? That IS the log of the triangular Jacobian's diagonal
    (dz_i/dx_i = e^{-alpha_i}) — your C3.2 determinant, in the log.
    """
    batched_forward = jax.vmap(lambda xi: made_forward(params, masks, xi))
    theta = batched_forward(X)
    z = (X - theta[:, :, 0]) * jnp.exp(-theta[:, :, 1])
    return jnp.sum(log_base(z) - theta[:,:, 1], axis=-1)


# ======================================================================
# =====[ TODO 5 ]=======================================================
# ======================================================================

def sample(params: list[dict], masks: list[jnp.ndarray], key: jax.Array,
           n: int, d: int) -> jnp.ndarray:
    """Ancestral sampling, N3 §4 at scale: draw Z (n, d) ~ N(0,1) up front,
    then a python loop over i = 0..d-1:

        theta = made_forward(x)      # same net, AGAIN — row i is now valid
        x_i   = mu_i + exp(alpha_i) * z_i     # affine inverts in closed form

    The junk in x at columns >= i is ignored (your masks guarantee it).
    Yes: d sequential passes. That is the point — feel GPT's latency.
    Hint: jit one full-batch made_forward pass and loop in python; at d=784
    this is slow-but-fine. (With the TODO 6 K-bump head, the closed-form
    inverse dies and your N2 bisection comes back.)
    """
    z = random.normal(key, (n, d))
    x = jnp.zeros((n, d))
    batched_forward = jax.vmap(lambda xi: made_forward(params, masks, xi))
    for i in range(d):
        theta = batched_forward(x)
        x = x.at[:, i].set(theta[:, i,0] + jnp.exp(theta[:, i,1]) * z[:, i])
    return x


# ======================================================================
# Provided: autotests for the masks — C3.2 as a unit test
# ======================================================================

def stage_masks() -> None:
    d, hidden = 7, (16, 16)
    key = random.PRNGKey(0)
    degrees = assign_degrees(key, d, hidden)
    masks = build_masks(degrees, d)
    params = init_made(random.PRNGKey(1), d, hidden)
    # un-shrink the final layer so dependence, if any, is visible
    params[-1]["W"] = params[-1]["W"] * 1000.0
    x = random.normal(random.PRNGKey(2), (d,))

    def z_of_x(x):
        theta = made_forward(params, masks, x)
        return (x - theta[:, 0]) * jnp.exp(-theta[:, 1])

    J = jax.jacfwd(z_of_x)(x)                                   # (d, d)
    upper = jnp.max(jnp.abs(jnp.triu(J, k=1)))
    diag_min = jnp.min(jnp.diag(J))
    print(f"max |dz_i/dx_j| for j>i : {float(upper):.2e}   (must be 0 — triangular, C3.2)")
    print(f"min  dz_i/dx_i          : {float(diag_min):.2e}   (must be > 0 — monotone, C2.2)")

    Jt = jax.jacfwd(lambda x: made_forward(params, masks, x))(x)  # (d, 2, d)
    leak = jnp.max(jnp.abs(jnp.stack([jnp.triu(Jt[:, j, :], k=0) for j in range(OUT_PER_DIM)])))
    print(f"max |dtheta_i/dx_j| j>=i: {float(leak):.2e}   (must be 0 — theta_i blind to x_i itself)")
    ok = (upper == 0) and (diag_min > 0) and (leak == 0)
    print("PASS ✅" if ok else "FAIL ❌ — a mask rule is leaking future coordinates")


# ======================================================================
# Provided: generic training harness (uses YOUR log_prob)
# ======================================================================

def train(params, masks, data_tr, data_va, steps: int, batch: int,
          lr: float = 1e-3, weight_decay: float = 1e-4, log_every: int = 500):
    opt = optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=weight_decay)
    state = opt.init(params)
    nll = lambda p, X: -jnp.mean(log_prob(p, masks, X))

    @jax.jit
    def step(p, s, X):
        loss, g = jax.value_and_grad(nll)(p, X)
        updates, s = opt.update(g, s, p)
        return optax.apply_updates(p, updates), s, loss

    key = random.PRNGKey(0)
    for t in range(steps):
        key, sub = random.split(key)
        idx = random.randint(sub, (batch,), 0, data_tr.shape[0])
        params, state, loss = step(params, state, data_tr[idx])
        if t % log_every == 0 or t == steps - 1:
            print(f"step {t:6d}   train NLL {float(loss):9.3f}   val NLL {float(nll(params, data_va)):9.3f}")
    return params


# ======================================================================
# Stage: moons — your architecture at D=2, scored on N3's own ruler
# ======================================================================

def stage_moons() -> None:
    from toylab import two_moons, kl_grid_2d

    key = random.PRNGKey(0)
    k_tr, k_va = random.split(key)
    data, vald = two_moons.sample(k_tr, 16000), two_moons.sample(k_va, 4000)

    d, hidden = 2, (64, 64)
    degrees = assign_degrees(random.PRNGKey(3), d, hidden)
    masks = build_masks(degrees, d)
    params = init_made(random.PRNGKey(4), d, hidden)

    print("PREDICT before this prints: N3 flow-conditional KL = 0.036, "
          "N3 break-it Gaussian-conditional KL = 0.070.\n"
          "One affine MAF layer has Gaussian conditionals AND a Gaussian x1-marginal. "
          "Where does it land?\n")
    params = train(params, masks, data, vald, steps=4000, batch=1024)

    lim, G = 7.0, 300
    axis = jnp.linspace(-lim, lim, G)
    GX1, GX2 = jnp.meshgrid(axis, axis)
    gpts = jnp.stack([GX1.ravel(), GX2.ravel()], 1)
    q = jnp.exp(log_prob(params, masks, gpts)).reshape(G, G)
    p = jnp.exp(two_moons.log_prob(gpts)).reshape(G, G)
    print(f"\ngrid KL(p||q) = {float(kl_grid_2d(p, q, axis)):.4f}"
          f"    (N3 flow 0.036 · N3 break-it 0.070)")
    print("Beat it: TODO 6 — K-bump head, or stack MAF layers with order reversal.")

    S = sample(params, masks, random.PRNGKey(42), 4000, d)
    print(f"samples mean ({float(S[:,0].mean()):+.3f}, {float(S[:,1].mean()):+.3f})"
          f"   std ({float(S[:,0].std()):.3f}, {float(S[:,1].std()):.3f})")
    print(f"data    mean ({float(data[:,0].mean()):+.3f}, {float(data[:,1].mean()):+.3f})"
          f"   std ({float(data[:,0].std()):.3f}, {float(data[:,1].std()):.3f})")


# ======================================================================
# Stage: baseline — the N1-style ruler on MNIST (runs with NO TODOs done)
# ======================================================================

def stage_baseline() -> None:
    x_tr, x_te = load_mnist()
    y_tr = to_logit_space(x_tr, random.PRNGKey(0))
    y_te = to_logit_space(x_te, random.PRNGKey(1))
    mu, sig = jnp.mean(y_tr, 0), jnp.std(y_tr, 0) + 1e-6      # diagonal Gaussian, closed-form MLE (N1)
    log_q = jnp.sum(log_base((y_te - mu) / sig) - jnp.log(sig), axis=-1)
    print(f"diagonal-Gaussian baseline: test bits/dim = {bits_per_dim(log_q, y_te):.3f}")
    print("this is the number your MAF must beat — it models every pixel "
          "INDEPENDENTLY (all masks zero, so to speak): no conditioning at all.")


# ======================================================================
# Stage: mnist — the real thing
# ======================================================================

def stage_mnist(n_train: int = 60000, steps: int = 6000, hidden=(1024, 1024)) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_tr, x_te = load_mnist()
    y_tr = to_logit_space(x_tr[:n_train], random.PRNGKey(0))
    y_te = to_logit_space(x_te, random.PRNGKey(1))

    d = 784
    degrees = assign_degrees(random.PRNGKey(3), d, hidden)
    masks = build_masks(degrees, d)
    params = init_made(random.PRNGKey(4), d, hidden)
    n_params = sum(int(np.prod(p["W"].shape)) + p["b"].shape[0] for p in params)
    n_alive = sum(int(m.sum()) for m in masks)
    print(f"weights {n_params:,} of which alive under the masks: {n_alive:,}")

    params = train(params, masks, y_tr, y_te[:2000], steps=steps, batch=256, log_every=250)

    lq = log_prob(params, masks, y_te)
    print(f"\nMAF test bits/dim = {bits_per_dim(lq, y_te):.3f}   "
          f"(run the `baseline` stage for the number to beat)")

    S = sample(params, masks, random.PRNGKey(42), 64, d)
    imgs = np.asarray(from_logit_space(S)).reshape(8, 8, 28, 28)
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for r in range(8):
        for c in range(8):
            axes[r, c].imshow(imgs[r, c], cmap="gray_r"); axes[r, c].axis("off")
    out = Path(__file__).resolve().parent / "mnist_maf_samples.png"
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"64 ancestral samples -> {out}")


# ======================================================================

STAGES = {"masks": stage_masks, "moons": stage_moons,
          "baseline": stage_baseline, "mnist": stage_mnist}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        print(__doc__.split("Run stages")[1])
        sys.exit(1)
    STAGES[sys.argv[1]]()
