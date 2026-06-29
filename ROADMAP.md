# Roadmap — Generative AI from 1D Fundamentals

The plan-of-record for *what we're going to build*. Domain terms live in [CONTEXT.md](./CONTEXT.md);
decisions with trade-offs live in [docs/adr/](./docs/adr/). This file is the syllabus + conventions.

## The through-line

> Every generative model is just a different trick to **represent and sample from `p(x)`** given
> only a bag of samples. They differ in (1) how they represent `p`, (2) how they're trained,
> (3) how they draw a sample.

We learn each trick on the simplest playground that still shows it — a 1D density we can fully
plot — then extrapolate.

## Conventions

- **1D-default discipline** — every concept introduced in 1D (a plottable scalar curve); promote to
  2D (two-moons) *only* where a mechanism needs ≥2 coordinates (coupling flows, autoregressive,
  manifold/latent geometry).
- **Stack** — pure **JAX** with **explicit param PyTrees** + **Optax**; objectives & samplers
  hand-rolled from the math; tiny MLPs; explicit `PRNGKey` threading. One **no-magic detour** after
  N1 (hand-coded NumPy backprop + hand-rolled Adam, once). See ADR-0001.
- **`toylab/`** — thin infra-only shared module (`targets.py`, `viz.py`, `metrics.py`); the *method*
  never leaves the notebook. N0 hand-builds these, then promotes them.
- **Socratic worksheets** — each notebook interleaves `🤔 CHECKPOINT` questions drawn from five
  archetypes (Predict-then-run, Derive, Hand-compute, Break-it, Connect), answered in chat under
  **strict gating** (guide holds a rubric, won't advance until the answer holds).

## Syllabus

| # | Notebook | Arc | Key component (the one new idea) | Dim |
|---|---|---|---|---|
| N0  | What is generative modeling? | Foundations | the evaluation ruler (histogram-vs-true `p`; exact 1D KL & Wasserstein) | 1D |
| N1  | Maximum likelihood = minimizing KL | Foundations | the MLE/KL objective (the bedrock) | 1D |
| N1b | Histogram & KDE | Foundations | the nonparametric baseline | 1D |
| N2  | Normalizing flows | Explicit densities | exact likelihood via invertible transport | 1D (→2D coupling) |
| N3  | Autoregressive models | Explicit densities | factorize the joint (the GPT connection) | 2D |
| N4  | VAE | Latent-variable | the variational lower bound (ELBO) | 1D |
| N5  | GANs | Implicit | density-ratio / adversarial training (watch mode collapse) | 1D |
| N6  | Energy-based models | Energy & score | unnormalized densities + Langevin MCMC | 1D |
| N7  | Score matching | Energy & score | learn `∇log p`, not `p` | 1D |
| N8  | DDPM / diffusion | Diffusion | multi-scale denoising | 1D |
| N9  | Score SDEs & probability-flow ODE | Diffusion | SDE/ODE duality | 1D |
| N10 | Flow matching / rectified flow | Flow matching | simulation-free transport (SD3/Flux) | 1D |
| N11 | Conditional generation & guidance | Conditioning | steering the score/velocity (→ text-to-image) | 1D/2D |
| N12 | Unification & frontier | Frontier | consistency/distillation; "everything is transport" | 1D |

## Capstone (mini-arc)

A guide-**sealed 2D target** engineered with pathologies (separated multimodality, sharp support
boundary / thin manifold, an imbalanced *rare* mode, small sample budget) chosen to pull the
**six evaluation axes** apart: Fidelity · Mode coverage · Exact likelihood · Sampling speed ·
Training stability · Data efficiency.

Rules: learner **implements every** covered method on it and **commits a full method×axis ranking
before grading**; the guide reveals ground-truth `p(x)` + a defensible ranking only after.
Deliverable = a trade-off table + a verdict that **names its criterion**. Thesis: *"best" is a
multi-objective frontier, not a scalar.*

## Repo layout (target)

```
ml_fp/
├── CONTEXT.md          # glossary
├── ROADMAP.md          # this file
├── docs/adr/           # decision records
├── toylab/             # targets.py · viz.py · metrics.py
└── notebooks/          # N0 … N12, then capstone/
```

## Resuming in a new context window

A fresh session has *none* of the design conversation — but it has these files. They are the
shared memory. To reload the guide:

**Session opener (paste this):**

> Open `ml_fp`. Read `CONTEXT.md` and `ROADMAP.md` first — we're doing the 1D generative-AI
> curriculum. I'm on **N0** *(or wherever)*. Start it / resume. Use **strict gating**.

**Refer to things by their canonical handles** so the guide knows exactly what you mean:

- **Notebooks by ID** — "N0", "N4", "the capstone" → the syllabus table above.
- **Concepts by their CONTEXT.md term** — e.g. *target density*, *1D-default discipline*,
  *checkpoint*, *hand-compute*, *toylab*, *explicit param PyTree*, *evaluation axes*. Avoid the
  flagged aliases ("the function", "the curve") — they mean something different here.
- **Mid-session signals:** "give me the answer" = tap out of a checkpoint · "connect this to N2"
  = Connect archetype · "break-it on X" = failure-mode probe · "hand-compute check" = tiny-numbers
  exercise.

**Where am I?** Notebooks present under `notebooks/` = started. Tell the guide the ID and you
resume there; the guide keeps a one-line status banner at the top of each notebook.
