# Generative AI from 1D Fundamentals

A learning curriculum that builds every generative-model family (VAE, normalizing flows,
GANs, diffusion, flow matching, …) from scratch on the simplest possible playground: a 1D
probability density learned from unlabeled samples. The bet: master the *components* on a
toy you can fully visualize, then extrapolate to any dimensionality.

## Language

**Target density** `p(x)`:
The unknown 1D probability distribution we are trying to learn. We only ever observe samples
from it, never its formula. In toy experiments we secretly *know* `p(x)` (so we can grade
ourselves), but the model is not allowed to see it.
_Avoid_: "the function", "the curve" (those suggest a mapping `y=f(x)` — see Flagged ambiguities)

**Sample**:
A single unlabeled scalar `xᵢ` drawn from the target density. The training set is a bag of
these — `{x₁, …, x_N}` — with **no `y`, no labels**.
_Avoid_: "data point" when it risks implying an `(x, y)` pair

**Generative model**:
A model that learns to *produce new samples* resembling those from the target density (and,
for some families, to evaluate the density). Contrast with regression, which learns a mapping
`y=f(x)` from labeled pairs.
_Avoid_: using "model" to mean a regressor

**Toy target**:
A deliberately simple, fully-known target density used as the canonical playground. The
"hello world" is a **multimodal mixture of Gaussians** on the real line (chosen because it is
multimodal — so it breaks naive models instructively — has a closed-form density, and is
trivially plottable).

**1D-default discipline**:
The project convention that every concept is introduced and dissected in 1D, where the math
is a plottable scalar curve. We promote to 2D *only* when a mechanism is defined by
interactions between ≥2 coordinates (coupling flows, autoregressive factorization,
manifold/latent geometry), and we flag in that notebook why 1D can't show it.

**Two-moons**:
The canonical 2D toy target, used only when the 1D-default discipline forces an escalation.

## Pedagogy vocabulary

**Socratic worksheet**:
What a notebook *is* — not a read-along, but exposition interleaved with questions the learner
must answer in chat before progressing.

**Checkpoint**:
An inline `🤔 CHECKPOINT` cell (1–3 questions) at a pivotal moment in a notebook, with a blank
"Your answer:" cell beneath. The learner answers in chat; the guide grades Socratically.
Each notebook also has opening *objective-questions* and a harder closing *exit-questions* set.

**Archetype**:
One of the five question kinds a checkpoint draws from — **Predict-then-run**, **Derive**,
**Hand-compute** (do it by hand on ~3 numbers), **Break-it** (degenerate a knob, predict the
failure), **Connect** (relate the key component back to an earlier notebook).

**Strict gating**:
The convention that the guide holds a private rubric per checkpoint and does not advance to the
next concept until the learner's answer holds up — unless the learner explicitly taps out.

**Capstone**:
The final challenge. A guide-*sealed* 2D target engineered with pathologies (separated
multimodality, a sharp support boundary / thin manifold, an imbalanced *rare* mode, and a small
sample budget) that pull the evaluation axes apart. The learner implements **every** covered
method on it, commits a full method×axis ranking *before* grading, and the deliverable is a
trade-off table plus a verdict that **names its criterion**. Embodies the thesis: "best" is a
multi-objective frontier, not a scalar.

**Evaluation axes**:
The six criteria every method is scored on at the capstone — **Fidelity**, **Mode coverage**,
**Exact likelihood**, **Sampling speed**, **Training stability**, **Data efficiency**.

## Stack vocabulary

**No-magic detour**:
A one-time early exercise (right after N1) that hand-codes backprop through a 1-hidden-layer net
in pure NumPy and hand-rolls Adam — to show, once, what `jax.grad` and Optax do underneath.
After it, autodiff/Optax carry the bookkeeping and attention goes to the generative ideas.

**Explicit param PyTree**:
The project's model representation — network parameters are a plain JAX PyTree (e.g. a list of
`(W, b)` arrays) threaded by hand into forward functions, with `jax.grad`/`jit`/`vmap` for the
calculus and Optax for the optimizer. No object/layer abstraction hides the wiring (the reason
JAX was chosen over PyTorch). Objectives and samplers are always hand-rolled from the math.

**toylab**:
The thin shared module holding *only* identical, un-interesting scaffolding —
`targets.py` (toy densities, each with both `sample(key, n)` and `log_prob(x)`), `viz.py`
(standard plots), `metrics.py` (exact 1D KL/Wasserstein + 2D approximations). **Hard rule:** a
method (objective, model, sampler) *never* lives here — it stays visible in its notebook. N0
hand-builds these tools, then "promotes" them into `toylab`.

## Flagged ambiguities

**"1D function"** — Resolved to mean a 1D probability *density* `p(x)` (option b), learned
from unlabeled samples. It does **not** mean a mapping `y=f(x)` (regression). The "function"
we secretly learn *is the density itself*.

## Example dialogue

> **Learner:** I have samples from a 1D function.
> **Guide:** Careful — do you have `(xᵢ, yᵢ)` pairs, or just scalars `xᵢ`?
> **Learner:** Just the scalars: `[-2.1, 1.9, 2.0, -2.0, …]`.
> **Guide:** Then you don't have a function `f(x)`, you have **samples** from a **target
> density** `p(x)` — here, two bumps near ±2. A **generative model** learns to spit out 1000
> more numbers from that same `p`. No `y` anywhere.
