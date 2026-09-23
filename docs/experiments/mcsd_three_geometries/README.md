# Three-geometry MCSD study

This branch implements a fixed-width dense decoder experiment based on the
handoff in this directory. The original four files are retained verbatim as
the initial research contract. `PROTOCOL_DELTA.md` records later decisions.

## Method IDs

| Geometry | Optimizer | Main direction | Return |
| --- | --- | --- | --- |
| Frobenius sphere | `manifold_mcsd` | ambient EMA, tangent-projected momentum, BF16 Polar Express LMO | FP32 Frobenius normalization |
| Frobenius sphere | `manifold_mcsd_tp` | same, plus tangent projection after LMO | FP32 Frobenius normalization |
| scaled Stiefel | same two IDs | same respective directions | BF16 Polar Express full `msign`, then three FP32 polar-polish steps |
| spectral sphere, exact reference | same two IDs with `--manifold-spectral-solver exact` | exact SVD tangent normal or practical LMO | MCSD: full-SVD clipping; MCSD-TP: full-SVD norm radial return |
| spectral sphere, practical pilot | same IDs with `--manifold-spectral-solver pi_topk` | BF16 PI tangent normal | MCSD: top-8 deflation; MCSD-TP: PI radial scaling |

The practical spectral pilot is **not** a certified projection, and the
Stiefel finite-step return is **not** an exact polar projection. The exact
reference modes are controls, not the default performance configuration.
The sub-`1e-4` spectral gap is logged but does not stop a run. At a repeated
top singular value the chosen rank-one normal has no unique smooth-stratum
tangent interpretation, so such a run is empirical rather than a direct
validation of the smooth-region theorem.
For the practical spectral mode, exact gap and constraint checks are audits
every 200 component updates by default (every 20 in the H20 smoke script),
not per-step decisions; events between audits may be missed.
The same interval staggers LMO feasibility and return-defect audits across
logical components. The reported LMO ball excess and model error are measured
before MCSD-TP projects the direction for the second time. Exact spectral
returns cache the chosen top normal for the next step, avoiding a repeated
SVD of the current master matrix after the first step.

## Local checks

```bash
PYTHONPATH=Megatron-LM python -m pytest tests/test_manifold_mcsd.py -q
python scripts/manifold/launch.py --geometry frobenius --method manifold_mcsd_tp
```

The launcher prints its resolved Slurm command by default. `--submit` must be
given explicitly. The first-phase launcher limits runs to 100 optimizer steps.
The job script expects indexed OLMo-Mix data and an existing H20 environment;
its paths can be overridden through environment variables without editing the
tracked script. It writes a new run directory keyed by the Slurm job ID.

## Reproducibility boundaries

The model is a 28-layer, width-384, FFN-1152 decoder with 6 query heads,
3 KV groups, head dimension 64, sequence length 2048, SwiGLU, QK norm,
RMSNorm and RoPE. Each transformer block is split into logical Q/K/V/O and
gate/up/down matrices. The optimizer stores a fixed radius and shape scale
per logical component, with FP32 master weights and momentum. The forward
model remains BF16. Auxiliary parameters are routed to AdamW with their own
learning rate; constrained matrices receive no weight decay.

The initial phase is implementation and smoke only. Its losses are not a
paper comparison. Formal comparisons still need matched baselines, multiple
seeds, validation-only selection, final held-out evaluation, full timing,
checkpoint-resume checks on H20, and per-step constraint audits at a stated
frequency. Do not compare a practical variant with an exact reference as if
only the algorithmic direction changed.

`MAIN1_ALIGNMENT.md` compares this implementation with the current paper and
lists the evidence still needed before using the new study as a publication
result. The paper's current experiment section contains Stiefel PCA/Brockett
tests, not this LLM study.

Primary implementation references:
[Polar Express msign](https://github.com/thinking-machines-lab/manifolds/blob/main/src/msign.py),
[hyperspherical descent](https://github.com/thinking-machines-lab/manifolds/blob/main/src/hyperspherical_descent.py).
The latter normalizes its tangent direction; MCSD-TP intentionally does not,
because that would change the algorithm specified in `IMPLEMENTATION.md`.
