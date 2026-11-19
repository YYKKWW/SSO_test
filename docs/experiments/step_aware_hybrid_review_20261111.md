# Step-Aware Hybrid Theory Audit

Audit date: 2026-11-11. Scope: Sections 3-6 of the
research handoff `MCSD_Hybrid_ICML_Theory_and_Experiments_20261111.md`
(author-local document, not included in this repository).
The handoff and production optimizers are unchanged. This review concerns real
matrices, exact spectra, feasible iterates and Frobenius geometric corrections.

## Verdict

The model projection, momentum-descent inequality and conditional radial-return
bound are consistent. The proposed stochastic Clarke result has a viable proof
under its strong exact-algorithm assumptions. It is **not** a convergence theorem
for the fixed-K10, fixed-window, top-4 Nesterov implementation.

Three qualifications matter for the experiments:

1. The A4 inner-product test protects a momentum prediction, not the returned
   objective. A larger accepted direction can increase the return remainder.
2. The external-gap condition certifies a remainder bound, not monotonic
   objective descent. The gradient and smoothness terms must also be included.
3. Complete-window capacity and the capacity needed for the step-size bound
   are different. Extending a cluster changes the direction and must be audited
   by reconstructing it, not by holding its norm fixed and claiming a certificate.

## Section 3: Geometry

For the full true top space of multiplicity k, let H(D)=sym(U^T D V).
The Bouligand tangent is {D: lambda_max(H(D))=0}. Necessity follows from the
directional derivative of the spectral norm. Sufficiency follows by applying
the radial map to X+tD: its correction is o(t) when this derivative is zero.

At k=1 the regular normal is the entire line spanned by uv^T. At k>=2 the
polar of the Bouligand tangent is U S_+^k V^T. Limiting normals include both
signs of each paired rank-one top normal, obtained by approaching through
simple-top points. All limiting normals remain in the symmetric paired top
block; their closed convex hull is that whole block. Thus

$$
N^C_C(X)=\{UAV^T:A=A^T\},\qquad T^C_C(X)=\ker H.
$$

Do not substitute the spectral **ball's** PSD normal cone for this sphere's
Clarke normal. The claim concerns Clarke stationarity, not Bouligand
stationarity. For f(X)=X_11 at I_2, the Clarke residual is zero whereas
D=diag(-1,0) is a feasible strict descent direction.

The selected near cluster may contain more than the true multiplicity. Its
kernel L is then a surrogate, not the true Clarke tangent. For rectangular
matrices with q=r, Gamma=R is conservative: the unselected dilation block
contains zero and/or nonpositive eigenvalues. For a top-1 control at a multiple
root Gamma=0; no finite external-gap bound is available.

## Section 4: Model and Complete Return

The adjoint H*(A)=UAV^T is an isometry on symmetric matrices and HH*=I.
Write Z=(I-H*H)Z+H*H(Z). Only the second orthogonal component needs changing.
With E=RI-S, the shifted constraint is

$$
H(D)-E/\alpha\in\{B:B\preceq0,\ \lambda_{\max}(B)=0\}.
$$

The nearest symmetric matrix keeps the eigenspaces, clips positive
eigenvalues, and raises a largest eigenvalue to zero when all are negative.
This proves the stated global Frobenius projection formula, including its
possibly nonunique choices. It is a model projection, not a projection onto C.

Because P_L Z is feasible for this model, nearest-point comparison gives

$$
2\langle Z,D_F\rangle\ge\|D_F\|_F^2+\|P_LZ\|_F^2.
$$

Consequently the normalized-momentum bound in Section 4.2 is valid. The bound
||D_F||<=2 follows by comparison with zero. Similarly, projecting a bounded
LMO output gives ||D_L||<=2 a_N. The A4 test preserves the reference momentum
decrease, but does not make D_L an exact N-unit constrained LMO.

For the dilation proof, the selected block has top eigenvalue R, the
complement has top eigenvalue at most R-Gamma+delta, and the coupling norm
is at most delta=alpha||D||_2. A two-block Rayleigh bound gives

$$
0\le\|X+\alpha D\|_2-R
\le\frac{\delta^2}{\Gamma-\delta}
\le\frac{2\delta^2}{\Gamma},\qquad \delta\le\Gamma/2.
$$

The Frobenius radial remainder is at most sqrt(r) times this spectral
overshoot. The displacement bound follows independently from
| ||X+alpha D||_2-R |<=alpha||D||_2. These proofs do not need separation
inside the selected cluster.

**The additional full-descent test** is the analytic upper bound, with
e=Rad(X+alpha D)-(X+alpha D), d=Rad(X+alpha D)-X:

$$
f(X+d)-f(X)\le
\alpha\langle\nabla f(X),D\rangle+
\|\nabla f(X)\|_F\frac{2\sqrt r\alpha^2\|D\|_2^2}{\Gamma}
+\frac{L_f}{2}\|d\|_F^2.
$$

Only a negative right-hand side certifies actual decrease. A4 checks neither
this expression nor the function value. The local experiments log it for
diagnosis and do not use it for acceptance or backtracking.

## Section 5: Conditions for the Stochastic Claim

Use the filtration **before** the current sample: X_t and the stored previous
momentum are F_t-measurable; G_t, updated M_t, the selected direction and
X_(t+1) are F_(t+1)-measurable. Require
E[G_t|F_t]=grad f(X_t), bounded conditional variance and uniformly bounded
sample norms for this particular sufficient-condition theorem. No unbiasedness
of the nonlinear LMO output is asserted. This matches the current-sample
convention in the author's local manuscript `main1.tex`, around line 600
(not a public repository dependency).

The following argument completes the main logical links in the draft:

1. Compactness bounds grad f. Bounded M and norm equivalence convert the
   safe-branch dual-norm decrease and near-branch quadratic decrease into
   a common bound -c||P_L M||^2. If M=0, the tracking error term covers the
   gradient; choose uniform positive bounding constants, not 1/||M||.
2. Replace M by grad f using bounded directions and
   | ||P_L M||^2-||P_L grad f||^2 |<=C||M-grad f||. Smoothness and the
   radial bound then yield the draft's pathwise inequality (*).
3. Tonelli gives sum alpha_t e_t<infinity almost surely from its expectation
   bound. Add the tail of this summable error to f(X_t): the resulting
   sequence is nonincreasing and bounded below. Hence f converges and
   sum alpha_t r_t^2<infinity almost surely.
4. For any non-Clarke-stationary cluster point, the shrinking-window
   detection argument gives a positive local lower bound on r(X,h) for
   all small h. The bounded displacement and alpha_t->0 imply that each
   inner-to-outer neighborhood excursion spends a fixed positive sum of
   alpha_t inside this neighborhood. Infinitely many excursions contradict
   summability; staying there forever contradicts sum alpha_t=infinity.

The detection step uses a subsequence of fixed q and convergent **paired**
bases. A vanishing surrogate residual places the limit gradient in a
symmetric sub-block of the limit's true top space, hence in its Clarke normal.
Completeness of the window is needed for Gamma>=h and the return proof;
it is not, by itself, needed for this one-way detection implication.

For deterministic gamma_t proportional to t^(-b), the EMA error expands
as weighted martingale differences plus gradient drift. Bounded steps give
the drift O(alpha_t/gamma_t); the martingale variance is O(gamma_t).
Thus E e_t=O(t^(b-a)+t^(-b/2)), plus the decaying initialization term.
The inequalities 2a-b>1, a+b/2>1 and 2a-d>1, with 0<b,d<a<1,
make the required errors summable. The example (.9,.5,.3) is admissible.

This supports **all cluster points Clarke stationary, almost surely, and
convergent objective values** for the exact reference algorithm. It does not
give convergence of the entire point sequence, B-stationarity, global
optimality, or a fixed-beta cosine-schedule theorem. The finite-time bound
controls the surrogate r(X,h), not the current true Clarke residual.

The external papers provide context, not this new theorem:
[MCSD v2](https://arxiv.org/abs/2601.21487v2) develops the existing
MCSD/MCSD-TP and projection hybrid; [Levin, Kileel and Boumal](https://arxiv.org/abs/2107.03877)
study cross-stratum stationarity failures; [Davis et al.](https://arxiv.org/abs/1804.07795)
analyze stochastic methods on tame objectives. The proof above is a separate
audit of the handoff's assumptions, not a claimed direct application of them.

## Section 6: What Still Does Not Transfer

Fixed h can create surrogate stationary points. A1 may stop because it removes
every symmetric direction in a wide top cluster. A shrinking window can avoid
that asymptotic obstruction while still being very conservative at finite
budgets. A4 must therefore be compared with both A1 and the top-1 control.

For approximate iterates, writing an extra summable r_num is not by itself a
proof: an infeasible X need not have E>=0 or the stated sphere geometry.
A formal extension needs a feasible shadow sequence or a robust one-step
inequality and a robust detection property at the approximate states. Keep
Section 6.2 as a route to an extension, not as an established approximate
algorithm theorem.

For a fixed candidate direction, increasing q changes Gamma; rebuilding that
direction at larger q also changes its spectral norm. The offline capacity
audit searches extensions q>=q_window and reconstructs the direction each
time. It is not used to change the actual trajectory. MCSD-TP's top-1 direction
is never silently replaced by a cluster direction. A bound may fail even at
full capacity if the step is too large. The bound is sufficient, not necessary.

The small matrix correction and caching do not remove the cost of acquiring
the top subspace. Here exact full spectra and exact polar LMOs deliberately
isolate geometry. Their CPU timing, including audits, cannot be interpreted
as a speedup over the H20 approximate implementation.

## Existing H20 Confirmation Batch

Two read-only SSH attempts on 2026-11-11 timed out. There is **no newly verified
server result** in this report. Last verified status is 2026-11-10 23:42 HKT,
all five pending resources; this is not asserted to be their current status.

| Job | Existing method | Last verified state |
|---|---|---|
| `4419145_0` | MCSD-TP K10 | Pending, 2026-11-10 |
| `4419145_1` | Single-LMO c0.5 | Pending, 2026-11-10 |
| `4419145_2` | 25% shape blend c0.5 | Pending, 2026-11-10 |
| `4419145_3` | MuonSphere | Pending, 2026-11-10 |
| `4419145_4` | SSO | Pending, 2026-11-10 |

[Submission/protocol](cluster_lmo_confirmation_1b_20261110.md).
No job was submitted, canceled or modified during this review. Retrieve the
five endpoints and matching curves when connectivity returns; do not replace
them with guessed results or submit duplicate runs.

## Local Implementation

[Reference code](../../scripts/local_step_aware_hybrid.py) and
[regression tests](../../scripts/test_local_step_aware_hybrid.py) implement all
four requested methods. Full-window q=1 uses the same MCSD-TP path. A2 uses
normalized momentum before model projection; A4 compares the corrected LMO
candidate with A2. The effective alpha is shared by model construction and
trial. There is no post-projection direction normalization, online error gate,
full-projection fallback, weight decay or function-value line search.

Tests include the 972 single-step cases, radius scaling 0.1/1/10, the original
2x2 radial ascent example, cross-face movement, the Clarke/B distinction,
zero momentum, q=1 equality, plateau zero-Gamma handling, paired reproducibility
and correct trial-length matching. The 2x2 example is repaired by the complete
cluster A2/A4 model; A1 gives a zero direction there. This is not evidence of
unconditional descent for arbitrary steps.

[Trajectory results and decision](step_aware_hybrid_results_20261111.md).
