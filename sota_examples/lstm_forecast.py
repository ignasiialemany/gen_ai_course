"""
N2 SOTA bridge — predict next-hour household power with a JAX LSTM.

Goal:  p_{t+1}  from  [p_{t-L+1}, ..., p_t]  (+ optional calendar features)

Run:
    python sota_examples/lstm_forecast.py
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------------------
# Config — tweak after the baseline works
# ---------------------------------------------------------------------------

SEQ_LEN = 24          # past 24 hourly readings
HIDDEN = 32
BATCH = 64
LR = 1e-3
STEPS = 500
SEED = 0

TRAIN_END = "2009-12-31"   # time-based split (no shuffle!)
VAL_END = "2010-06-30"


# ---------------------------------------------------------------------------
# Phase 1 — data
# ---------------------------------------------------------------------------

def load_hourly_power() -> "pd.Series":
    """Return a sorted hourly Global_active_power series indexed by datetime."""
    import pandas as pd

    raw = fetch_ucirepo(id=235)
    df = raw.data.features.copy()

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"], errors="coerce"
    )
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce"
    )

    s = (
        df.set_index("datetime")["Global_active_power"]
        .sort_index()
        .resample("1h")
        .mean()
        .dropna()
    )
    return s


def make_windows(
    power: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows: past seq_len hours -> next hour."""
    xs, ys = [], []
    for i in range(seq_len, len(power)):
        xs.append(power[i - seq_len : i])
        ys.append(power[i])
    return np.array(xs)[..., None], np.array(ys)[..., None]


@dataclasses.dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    y_mean: float
    y_std: float


def prepare_data() -> SplitData:
    """Time-based split; standardize X and y using train stats only."""
    import pandas as pd

    hourly = load_hourly_power()
    power = hourly.to_numpy(dtype=np.float64)
    X, y = make_windows(power, SEQ_LEN)

    t_end = pd.DatetimeIndex(hourly.index[SEQ_LEN:])
    train_end = pd.Timestamp(TRAIN_END)
    val_end = pd.Timestamp(VAL_END)
    train_m = t_end <= train_end
    val_m = (t_end > train_end) & (t_end <= val_end)

    X_train, y_train = X[train_m], y[train_m]
    X_val, y_val = X[val_m], y[val_m]

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std == 0:
        raise ValueError("train y_std is 0 — check data")

    # standardize inputs and targets with the same train stats (same units: kW)
    X_train = (X_train - y_mean) / y_std
    X_val = (X_val - y_mean) / y_std
    y_train = (y_train - y_mean) / y_std
    y_val = (y_val - y_mean) / y_std

    return SplitData(X_train, y_train, X_val, y_val, y_mean, y_std)


def naive_baseline_mae(hourly: "pd.Series", seq_len: int) -> float:
    """Seasonal naive: next hour = same hour yesterday (24h lag)."""
    import pandas as pd

    t_end = pd.DatetimeIndex(hourly.index[seq_len:])
    train_end = pd.Timestamp(TRAIN_END)
    val_end = pd.Timestamp(VAL_END)
    val_m = (t_end > train_end) & (t_end <= val_end)
    val_times = t_end[val_m]

    pred_kw = hourly.shift(24).loc[val_times]
    actual_kw = hourly.loc[val_times]
    ok = pred_kw.notna() & actual_kw.notna()
    return float((actual_kw[ok] - pred_kw[ok]).abs().mean())


# ---------------------------------------------------------------------------
# Phase 2 — LSTM in JAX
# ---------------------------------------------------------------------------

class LSTMParams(NamedTuple):
    """One LSTM layer + linear readout. Explicit PyTree — no hidden framework."""

    Wi: jnp.ndarray
    Wf: jnp.ndarray
    Wo: jnp.ndarray
    Wg: jnp.ndarray
    bi: jnp.ndarray
    bf: jnp.ndarray
    bo: jnp.ndarray
    bg: jnp.ndarray
    Wy: jnp.ndarray
    by: jnp.ndarray


def init_lstm_params(key: jax.Array, input_dim: int, hidden: int) -> LSTMParams:
    keys = jax.random.split(key, 10)
    scale = 0.01
    in_out = (hidden + input_dim, hidden)
    Wi = jax.random.normal(keys[0], in_out) * scale
    Wf = jax.random.normal(keys[1], in_out) * scale
    Wo = jax.random.normal(keys[2], in_out) * scale
    Wg = jax.random.normal(keys[3], in_out) * scale
    bi = jax.random.normal(keys[4], (hidden,)) * scale
    bf = jax.random.normal(keys[5], (hidden,)) * scale
    bo = jax.random.normal(keys[6], (hidden,)) * scale
    bg = jax.random.normal(keys[7], (hidden,)) * scale
    Wy = jax.random.normal(keys[8], (hidden, 1)) * scale
    by = jax.random.normal(keys[9], (1,)) * scale
    return LSTMParams(Wi, Wf, Wo, Wg, bi, bf, bo, bg, Wy, by)


def lstm_cell(
    params: LSTMParams,
    carry: tuple[jnp.ndarray, jnp.ndarray],
    x_t: jnp.ndarray,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    h, c = carry
    hx = jnp.concatenate([h, x_t], axis=0)
    i = jax.nn.sigmoid(hx @ params.Wi + params.bi)
    f = jax.nn.sigmoid(hx @ params.Wf + params.bf)
    o = jax.nn.sigmoid(hx @ params.Wo + params.bo)
    g = jax.nn.tanh(hx @ params.Wg + params.bg)
    c_prime = f * c + i * g
    h_prime = o * jax.nn.tanh(c_prime)
    return (h_prime, c_prime), h_prime


def lstm_forward(params: LSTMParams, x_seq: jnp.ndarray) -> jnp.ndarray:
    """x_seq: (seq_len, input_dim) -> scalar prediction."""

    def step(carry: tuple[jnp.ndarray, jnp.ndarray], x_t: jnp.ndarray):
        return lstm_cell(params, carry, x_t)

    init = (jnp.zeros(HIDDEN), jnp.zeros(HIDDEN))
    (h_final, _), _ = jax.lax.scan(step, init, x_seq)
    return (h_final @ params.Wy + params.by).squeeze()


def batch_predict(params: LSTMParams, X: jnp.ndarray) -> jnp.ndarray:
    """X: (batch, seq_len, input_dim) -> (batch, 1)"""
    preds = jax.vmap(lstm_forward, in_axes=(None, 0))(params, X)
    return preds[..., None]


def mse_loss(params: LSTMParams, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((batch_predict(params, X) - y) ** 2)


@jax.jit
def train_step(
    params: LSTMParams,
    opt_state: optax.OptState,
    X: jnp.ndarray,
    y: jnp.ndarray,
) -> tuple[LSTMParams, optax.OptState, jnp.ndarray]:
    loss, grads = jax.value_and_grad(mse_loss)(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(updates, params)
    return params, opt_state, loss


optimizer = optax.adam(LR)


def train(data: SplitData, steps: int = STEPS) -> LSTMParams:
    key = jax.random.key(SEED)
    key, k_init = jax.random.split(key)

    params = init_lstm_params(k_init, input_dim=1, hidden=HIDDEN)
    opt_state = optimizer.init(params)

    X = jnp.asarray(data.X_train)
    y = jnp.asarray(data.y_train)
    n = X.shape[0]

    for step in range(steps):
        key, k_batch = jax.random.split(key)
        idx = jax.random.randint(k_batch, (BATCH,), 0, n)
        params, opt_state, loss = train_step(
            params, opt_state, X[idx], y[idx]
        )
        if step % 50 == 0:
            print(f"step {step:4d}  train_mse {float(loss):.4f}")

    return params


def eval_mae(params: LSTMParams, data: SplitData) -> float:
    pred_std = batch_predict(params, jnp.asarray(data.X_val))
    pred_kw = pred_std * data.y_std + data.y_mean
    actual_kw = data.y_val * data.y_std + data.y_mean
    return float(jnp.mean(jnp.abs(pred_kw - actual_kw)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Phase 1: data ===")
    data = prepare_data()
    print(f"train windows: {data.X_train.shape}, val windows: {data.X_val.shape}")

    print("\n=== Phase 1: naive baseline (beat this!) ===")
    hourly = load_hourly_power()
    naive_mae = naive_baseline_mae(hourly, SEQ_LEN)
    print(f"naive MAE (kW): {naive_mae:.3f}")

    print("\n=== Phase 2: LSTM ===")
    params = train(data)
    lstm_mae = eval_mae(params, data)
    print(f"LSTM val MAE (kW): {lstm_mae:.3f}")
    delta = naive_mae - lstm_mae
    print(f"{'beat naive by' if delta > 0 else 'lost to naive by'} {abs(delta):.3f} kW MAE")


if __name__ == "__main__":
    main()
