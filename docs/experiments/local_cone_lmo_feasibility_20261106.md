# Local Feasibility: MCSD with P2GD-Type and Selected-Face LMO Directions

**2026-11-06. CPU only. No H20 connection, submission, or production change.**

[Executable study](../../scripts/local_cone_lmo_feasibility.py) |
[Regression tests](../../scripts/test_local_cone_lmo_feasibility.py) |
[Full local records](../results/local_cone_lmo_20261106/feasibility.json) |
[LLM results, a separate experiment](mcsd_p2gd_results_summary.md)

**Follow-up:** [exact-projection study and conditional stochastic convergence
proof](local_projected_lmo_study_20261106.md). It separates original MCSD from
MCSD-TP, extends 288 trajectories to 10,000 steps, and gives a counterexample
for exact projection as well as a separately analyzed model safeguard.

## Main Findings

1. The production small-matrix correction is a valid tangent-cone construction
   **when supplied the complete exact top singular subspace at a feasible point**.
   We derive the identity below and test 432 exact-arithmetic-model cases in FP64.
2. Its output is not generally an LMO on the original norm unit ball. Spectral
   norm reached 1.213056 and entrywise l1 norm reached 5.641621 despite unit-ball
   input oracles. This is not automatically an MCSD implementation bug: bounded
   corrected directions can still be analyzed, but cannot be called unit LMOs.
3. A near cluster or truncated top subspace need not produce a direction in the
   true tangent cone. Explicit 2x2/3x3 counterexamples demonstrate both failures.
4. Exact cone membership does not guarantee a finite radial step decreases the
   objective near a change of multiplicity. We give an exact-LMO counterexample.
5. All 768 toy trajectories remained feasible to FP64 accuracy, but the LMO
   hybrid did not consistently improve optimization. Equal-length controls change
   rankings. Persistent near calls can be appropriate when the optimum is singular.

These are geometry and implementation tests, not an LLM performance claim or a
proof of stochastic Clarke convergence.

## 1. Why the Small Correction Works

Let $C_R=\{X:\|X\|_2=R\}$, $R>0$. Suppose the complete top singular subspace
has multiplicity $q$, with paired orthonormal bases $U,V$. Write
$H(Z)=\operatorname{sym}(U^\top ZV)$. The directional derivative is
$\mathrm d\|X\|_2[D]=\lambda_{\max}(H(D))$, and

$$
T_{C_R}(X)=\{D:\lambda_{\max}(H(D))=0\}.
$$

Necessity follows by differentiating feasible secants. Sufficiency follows by
radially returning $X+tD$: a zero directional derivative implies the correction
is $o(t)$. This argument uses the **Bouligand tangent cone**, not a linear tangent
space at multiple top singular values. Spectral directional-derivative background:
[Lewis and Sendov, 2005](https://doi.org/10.1007/s11228-004-7198-6).

For descent input $Z=-M$, set $A=H(Z)$. With descending eigenvalues of $A$, set
$B$ by clipping positive eigenvalues to zero and forcing the largest eigenvalue
to zero even if all are negative. Then

$$
Q=Z+U(B-A)V^\top\in P^F_{T_{C_R}(X)}(Z).
$$

Let $P$ be the orthogonal projector onto the strictly negative eigenspace of
$B$. In particular $\operatorname{rank}(P)\le q-1$. Obtain one ordinary norm
oracle $S\in\arg\max_{N(S)\le1}\langle Q,S\rangle$. Define

$$
C=PH(S)P,\qquad D=S+U(C_- -H(S))V^\top,
$$

where $C_-$ retains the negative eigenvalues. The implementation also sandwiches
the reconstructed negative part by $P$ to reduce eigensolver roundoff.

**Cone membership:** $H(D)=C_-\preceq0$, supported on $\operatorname{range}P$.
Because $\ker P$ is nonempty, $H(D)$ has a zero eigenvalue. Therefore
$\lambda_{\max}(H(D))=0$. Negative semidefiniteness alone would only establish
an inward-or-tangent direction; the retained zero eigenvalue is essential for
the **sphere**, as opposed to the ball.

**Descent identity:** $A-B$ is supported on $\ker P$, whereas $H(D)$ is supported
on $\operatorname{range}P$. Consequently

$$
\langle Z,D\rangle=\langle Q,S\rangle-\langle B,C_+\rangle
\ge\langle Q,S\rangle=N_*(Q).
$$

Here $B\preceq0$ and $C_+\succeq0$. Thus with an exact gradient ($Z=-\nabla f$),
nonzero $Q$ yields a strict first-order descent direction. For an approximate
oracle the right-hand side is $\langle Q,S\rangle$, not automatically $N_*(Q)$.
BF16 NS8 must be assessed separately; the algebra study uses exact norm oracles.

This is a **selected-face, corrected LMO direction**, not the solution of the
norm-constrained tangent-cone LMO in general. For the Frobenius unit-ball oracle
it reduces to $Q/\|Q\|_F$ when $Q\ne0$.

### Norm Control Without Another Full SVD

The correction is the Frobenius projection onto the convex cone
$K_P=\{D:H(D)=PCP,\ C\preceq0\}$, hence $\|D\|_F\le\|S\|_F$.
For any fixed norm $N$ in finite dimension, choose $a,b$ such that
$\|Z\|_F\le aN(Z)$ and $N(Z)\le b\|Z\|_F$. Then $N(D)\le ab$.
This supplies a uniform bound, even if $N(D)>1$.

If unit-ball membership is required, positive scaling by $\max(1,N(D))$ preserves
cone membership and descent. That exact norm may be costly. For a spectral-norm
oracle, $a=\sqrt{\min(m,n)}$, $b=1$, so dividing by this fixed dimension factor is
a conservative SVD-free bound. It changes effective step length and should be a
separate ablation, not silently added to an existing run. No such production
change was made here.

## 2. What Does Not Follow

### Missing or Near Top Directions

- **Omitted true direction:** $X=I_3$, $Z=\operatorname{diag}(-1,1,1)$, but supply
  only the first two paired vectors. The corrected polar-LMO output is
  $D=\operatorname{diag}(-1,0,1)$. The model-cone residual is zero, but the true
  cone has $\lambda_{\max}(H(D))=1$. The missing direction moves outward.
- **Positive-gap surrogate:** $X=\operatorname{diag}(1,0.9999)$ and
  $Z=\operatorname{diag}(-1,1)$. Treating both directions as a repeated top gives
  $D=\operatorname{diag}(-1,0)$. The model cone is satisfied, but the true top is
  simple and the tangent condition is $D_{11}=0$, which fails by -1.

The latter direction points inward to first order. It may still work after
radial return, but the return then changes its first-order direction. Approximate
Ritz vectors introduce an additional error. A zero **model** residual cannot
certify the full true cone, and increasing top-4 to top-6 alone is not a proof.

### No Uniform Radial Model Across Multiplicity Changes

Let $X_\epsilon=\operatorname{diag}(1,1-\epsilon)$ and
$f(X)=-2X_{11}-X_{22}$. The exact top-1 MCSD/selected-face LMO direction is
$D=\operatorname{diag}(0,1)$, with $\langle\nabla f,D\rangle=-1$.
At step $\alpha=2\epsilon$:

$$
Y=\operatorname{diag}(1,1+\epsilon),\qquad
\operatorname{Rad}(Y)=\operatorname{diag}((1+\epsilon)^{-1},1).
$$

Thus $[f(\operatorname{Rad}(Y))-f(X_\epsilon)]/\alpha\to+1/2$, an **ascent**
step, while the linear model predicts -1. Moreover
$\|\operatorname{Rad}(Y)-Y\|_F/\alpha\to1/\sqrt2$, not zero.
Even the metric projection's displacement from the trial has ratio 1/2 here;
projection-based convergence uses a different inequality.

For a *fixed* point with a fixed positive separation from inactive singular
values, the local property is valid. A test at $X=\operatorname{diag}(1,1,.7,.4)$
gave radial error divided by step of 0.004611, 0.0004610, 0.00004610, 0.000004610
at steps $10^{-2},10^{-3},10^{-4},10^{-5}$. This does not make that rate uniform
over the whole sphere.

## 3. Conditional Stochastic Analysis

The current manuscript `main1.tex` uses a uniform return-map model on the safe
region and a separate Euclidean-projection branch elsewhere. Replacing that
branch by this construction is a change to the theorem, not a direct corollary.
The original projected line-search framework also does not establish convergence
for this stochastic, fixed-schedule radial variant:
[Schneider and Uschmajew, 2015](https://doi.org/10.1137/140957822).

A useful partial result can nevertheless be derived. For the **true** tangent
cone define

$$
\chi(x,g)=\max_{v\in T_C(x),\ \|v\|_F\le1}\langle-g,v\rangle
         =\|P^F_{T_C(x)}(-g)\|_F.
$$

At fixed $x$, $\chi$ is 1-Lipschitz in $g$, even though a nonconvex-cone
projection need not be Lipschitz. Using the identity above, norm equivalence,
and $e=\|m-\nabla f(x)\|_F$, the unscaled corrected LMO satisfies

$$
\|D\|_F\le a,\qquad
\langle\nabla f(x),D\rangle
\le-\chi(x,\nabla f(x))/b+(a+1/b)e.
$$

If the chosen feasible return **additionally** satisfies the uniform objective
model $f(R_x(\alpha D))\le f(x)+\alpha\langle\nabla f(x),D\rangle+
L\alpha^2\|D\|_F^2/2$, telescoping yields

$$
\frac{\sum_t\alpha_t\mathbb E\chi_t}{\sum_t\alpha_t}
\le \frac{b(f(x_0)-f_{\inf})}{\sum_t\alpha_t}
 +(ab+1)\frac{\sum_t\alpha_t\mathbb E e_t}{\sum_t\alpha_t}
 +\frac{bLa^2}{2}\frac{\sum_t\alpha_t^2}{\sum_t\alpha_t}.
$$

This is a **conditional descent calculation**, not a theorem that the current
LLM implementation satisfies these hypotheses. It identifies the missing work:

- Establish an appropriate uniform return inequality, or replace it with a
  separately justified mechanism that handles cross-stratum transitions.
  The counterexample above rules out the naive global radial assumption.
- Control momentum tracking error. Fixed-noise, fixed-momentum operation does
  not automatically make the weighted error term vanish. Unbiased samples do
  not imply unbiased normalized directions: values +1/-9 with probabilities
  .9/.1 have mean zero but mean sign .8.
- Bound errors from omitted directions, near-cluster substitution, finite
  NS/Ritz iterations and radial norm estimation at the scale of the steps.
- Handle stationarity-measure behavior across strata before asserting all
  cluster points are Clarke or Bouligand stationary. Small weighted averages
  by themselves do not prove such a cluster-point statement.

The toy trajectories use the implementation's Nesterov combination; the paper's
EMA-only tracking theorem cannot be cited as its analysis without adaptation.
There is no backtracking in any local test.

## 4. Exact Geometry Tests

432 configurations: shapes 8x6, 6x8 and 16x12; multiplicities 1, 2, 4 and full;
12 random draws; spectral, Frobenius and entrywise-l1 LMOs. Full bases and FP64
isolate algebra from subspace error. Max identity error: **3.553e-14**.

| Oracle unit ball | Cases | Max cone residual | Max output norm | Norm > 1 cases |
|---|---:|---:|---:|---:|
| Spectral | 144 | 4.459e-16 | 1.213056 | 144 |
| Frobenius | 144 | 1.465e-16 | 1.000000 | 0 |
| Entrywise l1 | 144 | 2.422e-16 | 5.641621 | 144 |

All exact-LMO descent inequalities hold to floating-point tolerance. Regression
tests also verify positive norm capping preserves cone membership, the momentum
tracking-error bound, the ascent counterexample, and sphere rather than ball
projection for interior trials.

## 5. Paired Stochastic Toy Optimization

Objective: $f(X)=\tfrac12\|X-A\|_F^2$ on $\|X\|_2=1$; $f^\star$ is computed
from the exact sphere projection of $A$. All methods share initial matrices,
noise arrays and the preset cosine schedule .015 to .0015 over 160 steps.
Shapes are 16x12, initial multiplicity is 2 or 4, and seed is 20261106 plus
case index. Zero noise and Gaussian noise with expected squared Frobenius norm
.01 are both tested. Noise is an unbiased additive gradient sample.

Two separate target families, each with 12 cases:

- **Random targets:** target spectral norm .7 (4 cases) or 1.4 (8 cases).
  Most hybrids enter the near branch only at initialization. This family alone
  is inadequate to study persistent nonsmooth behavior.
- **Plateau attractors:** target shares the initial singular bases, with its top
  2/4 values equal to 1.3. The constrained optimum genuinely has a repeated top.
  This deliberately stresses persistent near-region updates.

Eight methods x 12 cases x two noise levels x two length rules x two target
families = **768 trajectories**. True spectra and exact polar LMOs are used.
Safe directions are MCSD-TP. Near threshold is 5e-5 and cluster width is 2e-3,
with top<=4. All near multipliers are **1**, not the LLM c8 setting. `PGD raw`
uses unnormalized momentum; `PGD unit` and P2GD-type branches normalize
momentum first. P2GD labels below mean **near-cluster P2GD-type**, not an
invocation of a proven standard P2GD solver.

`Literal` preserves each direction's length. `Matched` normalizes **every**
nonzero direction to unit Frobenius norm, including safe steps. The two rules
are separate protocols; their absolute objective levels are not comparable.
Full projection caps all singular values above 1 and raises the top to 1 when
the trial is inside. Radial returns scale the entire trial. No best-LR selection.

Mean final $f-f^\star$, noise scale .1 (all cases retained):

| Method | Random, literal | Random, matched | Plateau, literal | Plateau, matched |
|---|---:|---:|---:|---:|
| MCSD-TP | 0.08288197 | 2.45598324 | **0.00037465** | **0.00014115** |
| MCSD + PGD raw | 0.08729536 | 2.44802268 | 0.00795336 | 0.00039052 |
| MCSD + PGD unit | 0.08631566 | 2.44802268 | 0.00039052 | 0.00039052 |
| MCSD + P2GD-type + project | 0.08623994 | 2.44724520 | 0.00039055 | 0.00015272 |
| MCSD + P2GD-type + radial | 0.08629226 | **2.44721926** | 0.00039079 | 0.00020207 |
| MCSD + selected-face LMO + project | **0.07681016** | 2.45322684 | 0.00038046 | 0.00014519 |
| MCSD + selected-face LMO + radial | 0.08285752 | 2.45318252 | 0.00044586 | 0.00016286 |
| MCSD + ambient unit + radial | 0.08718448 | 2.45580154 | 0.00051631 | 0.00054684 |

Maximum final sphere residual across all runs is below 3e-15. This confirms
exact-return feasibility, not successful convergence in 160 steps. The JSON
also retains final/best sampled true-cone stationarity, uphill-step counts and
branch use; near-stationarity has not been asserted based on loss alone.

In noisy plateau/literal runs, PGD and P2GD-type projection/radial branches are
used at every step on average. Selected-face LMO+radial uses its near branch
32.97% of the time, but has worse loss than MCSD-TP and LMO+projection. Lower
near frequency is not evidence of better optimization. With matched lengths,
LMO+radial near use becomes 100%. Step magnitude and return geometry interact.
The PGD raw/unit matched results coincide, as expected after length matching.

## 6. Width-256 Warm-Ritz Sanity Check

24 local checks: shapes 256x256, 768x256, 256x768; true multiplicities 2/6;
trial steps .0015/.015; K10/K20. The previous basis is initialized from the
**exact** preceding spectrum, and inactive singular values are at most .95.
Directions use the full true top subspace; they are unit-Frobenius scaled.
This is a favorable one-step warm-start test, not an actual LLM matrix sequence
or an empirical model of LLM update lengths, clustered tails or cache drift.

| Iterations | Checks | Mean gap error | Max gap error | Max returned-radius error | Route disagreements |
|---|---:|---:|---:|---:|---:|
| K10 | 12 | 6.400e-8 | 1.417e-7 | 2.300e-7 | 0 |
| K20 | 12 | 1.911e-7 | 4.596e-7 | 2.282e-7 | 0 |

Both work in this separated, oracle-initialized test. More FP32 iterations are
not necessarily more accurate after reaching a roundoff floor. These numbers
do **not** supersede the 1e-4-scale selected errors seen in real training audits.

## Next Local Work

1. Prioritize exact-top-subspace and matched-length comparisons before choosing
   a new gap or coefficient. Keep PGD projection and P2GD-type projection as
   separate references; a shared radial return is an explicit ablation.
2. Extend the plateau family with a shrinking inactive gap and incomplete q=4/6
   subspaces, then study multi-step FP32 cache drift. The one-step warm audit
   is too favorable to choose a production method.
3. For theory, keep the conditional corrected-LMO descent lemma separate from
   the unresolved return/stratum argument. The existing MCSD-PGD theorem remains
   the reliable reference; no replacement theorem has been established here.

## Reproduce Locally

Requires Python with CPU PyTorch; no training dataset or GPU.

```bash
OMP_NUM_THREADS=1 python -B scripts/local_cone_lmo_feasibility.py \
  --output docs/results/local_cone_lmo_20261106/feasibility.json
python -B -m unittest discover -s scripts -p 'test_local_cone_lmo_feasibility.py' -v
```

Production helpers are reused without modification. The JSON contains all
cases, seeds, protocol settings and SHA256 hashes of the script and helpers.
These CPU tests are not added to the count of completed H20 training runs.
