# Direction Plus Radial Return: A Conditional Stochastic Theory

Research note, 2026-11-18. This is a spectral-sphere-specific candidate and
proof framework, not a theorem already established for the production
optimizer. No H20 job or production algorithm was changed for this study.

[Earlier finite-step framework](kernel_lmo_theory_20261111.md) |
[Corner audit](corner_momentum_audit_20261117.md) |
[Local implementation](../../scripts/local_anchored_cone_radial.py) |
[Numerical checks](../results/anchored_cone_radial_20261118/local_checks.json)

## Main Conclusion

Radial return can support a stochastic convergence argument without a
function-value line search or a nearest-point projection onto the spectral
sphere. However, a descent direction must control the **complete returned
update**, not just its pre-return inner product with momentum.

A concrete sufficient construction is an **anchored cone-face correction**:
keep one genuinely active top singular direction unchanged to first order,
allow the other selected directions to contract, and preserve all free
directions. It has a small symmetric eigensolve and a uniform return estimate
controlled by the separation from the *unselected* spectrum. It does not
require a positive gap between the first two singular values.

Under the conditions below, the target is almost-sure **Clarke stationarity
of accumulation points**. This is weaker than Bouligand stationarity and
does not certify the fixed-K10, fixed-momentum, capped-top-4 LLM implementation.

## 1. Why Tangency Alone Is Not Enough

Let

$$
C_R=\{X\in\mathbb R^{m\times n}:\|X\|_2=R\},\qquad
\operatorname{Rad}_R(Y)=RY/\|Y\|_2,\quad R>0.
$$

If the true top singular value has multiplicity k, let U_k,V_k be paired
orthonormal bases of its complete singular subspaces. Put

$$
H_k(D)=\operatorname{sym}(U_k^\top D V_k),\qquad
\ell_X(D)=\lambda_{\max}(H_k(D)).
$$

The one-sided norm derivative is ell_X(D), and the Bouligand tangent cone is
T_C(X)={D:ell_X(D)=0}. Necessity follows by differentiating feasible curves;
sufficiency follows by radially returning X+alpha D and using the expansion

$$
\operatorname{Rad}_R(X+\alpha D)
=X+\alpha\left(D-\frac{\ell_X(D)}R X\right)+o(\alpha).
\tag{1}
$$

Thus genuine tangent-cone directions have no first-order radial correction.
But (1) at a fixed X,D is not a uniform quadratic estimate across strata.
For X_e=diag(1,1-e) and D=[[0,1],[1,0]], D is tangent for every e>0, while
the quadratic remainder coefficient grows like 1/e. Numerically, at
alpha=0.1e, ||Rad(X_e+alpha D)-(X_e+alpha D)||_F/alpha^2 increases from
13.31 to 1399.63 as e decreases from 0.1 to 0.001.

### A near-cluster correction can introduce first-order ascent

Take X_e=diag(1,1-e), gradient G=diag(2,1), and Z=-G/||G||_F. Treating both
singular values as an exact plateau produces the usual two-dimensional
cone correction Q=diag(-2/sqrt(5),0). This is tangent at I, but not at X_e.
For alpha much smaller than e, radial return has effective direction

$$
Q_{\rm eff}=\operatorname{diag}(0,2(1-e)/\sqrt5),\qquad
\langle G,Q_{\rm eff}\rangle>0.
$$

The local script verifies ascent for the smooth quadratic with current
gradient G at alpha=e^2 and e=1e-2,1e-3,1e-4. This failure is present with
exact FP64 spectra and exact gradients. Improving PI accuracy alone cannot
fix this mismatch between the chosen cluster geometry and the actual point.

## 2. An Anchored Cone Face

Select **every** singular value in [R-h,R], where 0<h<R. Write the paired
bases as U,V and S=diag(s_1,...,s_q), with s_1=R. Define

$$
H(D)=\operatorname{sym}(U^\top D V),\qquad
H^*(A)=UAV^\top,\qquad
L=\ker H,\quad P_L=I-H^*H.
$$

Choose an anchor a of unit length satisfying Sa=Ra. In a singular-vector
basis we can take a=e_1. The candidate direction set is the closed convex cone

$$
\boxed{\mathcal A=\{D:H(D)\preceq0,\ H(D)a=0\}.}
\tag{2}
$$

Its interpretation is precise: none of the selected top directions expands
to first order, and the genuinely active anchor does not contract. It
contains L and is a subset of the **true** Bouligand cone at X, even when
the selected cluster includes unequal singular values. Indeed, the true
active block is a compression of H(D), is negative semidefinite, and has
the active anchor in its kernel.

This is a selected convex face of the tangent-cone model, not the entire
nonconvex cone and not standard P2GD. For q=1 it reduces to the ordinary
top-1 tangent hyperplane. For q>1 it permits contraction directions that
the stronger constraint H(D)=0 would forbid.

### Explicit Frobenius projection

For Z, write A=H(Z) in the anchor-first basis. Let A_22 be the block excluding
the first row and column, and let (A_22)_- denote its negative semidefinite
part, obtained by replacing positive eigenvalues by zero. Then

$$
B=\begin{bmatrix}0&0\\0&(A_{22})_-\end{bmatrix},\qquad
\boxed{P_{\mathcal A}^F(Z)=Z+U(B-A)V^\top.}
\tag{3}
$$

Proof: H H*=I on symmetric matrices, and Z=P_LZ+H*H(Z) is an orthogonal
decomposition. Only H(Z) needs modification. The closest symmetric block
with a zero anchor row/column and a negative semidefinite remainder is B.
The remaining eigensolve has size (q-1)-by-(q-1).

As a projection onto a closed convex cone, this map is nonexpansive,
positively homogeneous, and satisfies

$$
\langle Z,P_{\mathcal A}Z\rangle=\|P_{\mathcal A}Z\|_F^2,
\quad \|P_{\mathcal A}Z\|_F\le\|Z\|_F,
\quad P_LP_{\mathcal A}Z=P_LZ.
\tag{4}
$$

Positive post-scaling preserves (2). This is an advantage over the earlier
finite-step set K, whose correction generally cannot be freely rescaled.

## 3. The Complete Radial Return Bound

For every D in A and every alpha>=0,

$$
S+\alpha H(D)\preceq RI,\qquad
(S+\alpha H(D))a=Ra,
\quad\Longrightarrow\quad
\lambda_{\max}(S+\alpha H(D))=R.
\tag{5}
$$

Thus A is contained in the earlier finite-step model K, without computing
(RI-S)/alpha and without an extra LMO.

Let r=min(m,n) and Gamma=R-s_(q+1) if q<r, or conservatively Gamma=R if q=r.
The complete-window rule implies Gamma>=h. If alpha||D||_2<=Gamma/2, then

$$
0\le\|X+\alpha D\|_2-R
\le\frac{2\alpha^2\|D\|_2^2}{\Gamma},
\tag{6}
$$

$$
\boxed{\|\operatorname{Rad}_R(X+\alpha D)-(X+\alpha D)\|_F
\le\frac{2\sqrt r\,\alpha^2\|D\|_2^2}{\Gamma}.}
\tag{7}
$$

Proof: use the symmetric dilation of X and the selected positive-eigenvalue
subspace W=(U;V)/sqrt(2). Its trial compression has largest eigenvalue R by
(5). The complementary compression is at most R-Gamma+alpha||D||_2, and
the cross-block norm is at most alpha||D||_2. The two-block Rayleigh bound
gives an excess of at most alpha^2||D||_2^2/(Gamma-alpha||D||_2). This proves
(6). Finally, radial correction has Frobenius norm
(||Y||_2-R)||Y||_F/||Y||_2 <= sqrt(r)(||Y||_2-R).

The potentially vanishing internal gap s_1-s_2 does not appear. What is
needed is coverage of the entire window and separation from the omitted
directions. At q=r the estimate remains valid, but the selected problem
may no longer be computationally small.

## 4. Normalized Momentum and Descent

Use the user's Frobenius normalization, with F(0)=0:

$$
F(M)=P_{\mathcal A}(-M/\|M\|_F).
$$

Fix X and its cone A; its construction must not change with M in the
following comparison. Write G=grad f(X), e=||M-G||_F, and r_h=||P_LG||_F.
Choose Gbar>=max(1,sup_C ||grad f||_F). Equations (3)-(4) imply

$$
\langle G,F(G)\rangle
=-\frac{\|P_{\mathcal A}(-G)\|_F^2}{\|G\|_F}
\le-\frac{r_h^2}{\mathrm{Gbar}}.
$$

Nonexpansiveness and the normalization inequality
||M/||M||-G/||G|||| <= 2||M-G||/||G|| give

$$
\boxed{\langle G,F(M)\rangle
\le-\frac{r_h^2}{\mathrm{Gbar}}+2e,\qquad \|F(M)\|_F\le1.}
\tag{8}
$$

The zero cases follow directly. This argument needs bounded true gradients
on the compact feasible set, not uniformly bounded stochastic samples.
A positive scale c_t in a fixed interval [c_min,c_max], c_min>0, is allowed:
replace the descent constant by c_min/Gbar and the error constant by
2c_max. A scale that vanishes, such as unqualified length matching to a
vanishing D0, does not automatically retain this conclusion.

### Keeping the paper's arbitrary-norm MCSD-TP branch

When q=1, use D=P_L LMO_N(-P_LM), with the LMO maximizing its displayed
argument over the N-unit ball. The two projections are the MCSD-TP input
and output projections. If a_N bounds the Frobenius norm of that ball and
N_*(z)>=b_N||z||_F, then a feasible inexact LMO with error zeta gives

$$
\langle G,D\rangle
\le-\frac{b_N}{\mathrm{Gbar}}r_h^2+2a_Ne+\zeta,
\quad \|D\|_F\le a_N.
$$

This direction has H(D)=0 and also satisfies (5)-(7). The near branch is
Frobenius cone projection of normalized momentum, **not** an arbitrary-norm
constrained LMO. The resulting hybrid retains the arbitrary-norm *safe*
branch but should not be renamed a single arbitrary-norm P2GD-LMO.

## 5. A Conditional Stochastic Convergence Proposition

Consider a reference algorithm with exact paired spectral information:

```text
sample stochastic gradient Gsample_t at feasible X_t
M_t = (1-gamma_t) M_(t-1) + gamma_t Gsample_t
select the complete singular-value window [R-h_t, R]
if q_t == 1:
    D_t = P_L LMO_N(-P_L M_t)
else:
    D_t = c_t P_A(-M_t / ||M_t||_F)  # zero when M_t == 0
Y_t = X_t + alpha_t D_t
X_(t+1) = R Y_t / ||Y_t||_2
```

There is no function-value backtracking and no metric projection onto C_R.
The spectrum of the returned point can be cached by scaling the trial
spectrum. A current-point initial decomposition is still required unless
available from initialization. Algebraic elimination of the full-set
projection is not yet a guarantee that a fixed partial eigensolver is accurate.

### Conditions

1. f is C1 with Lipschitz gradient on a neighborhood of the spectral ball;
   X_0 belongs to C_R. All returns in this proposition are exact radial returns.
2. The selected window is complete; the anchor is a genuine active direction;
   all choices are measurable. LMO outputs are feasible for the chosen norm
   ball. The near scales satisfy 0<c_min<=c_t<=c_max<infinity.
3. Deterministic positive alpha_t,h_t satisfy h_t<R, h_t->0,
   alpha_t B<=h_t/2, sum alpha_t=infinity, and sum alpha_t^2/h_t<infinity,
   where B=max(a_N,c_max).
4. The momentum error and any inexact LMO error obey
   sum alpha_t E||M_t-grad f(X_t)||_F<infinity and
   sum alpha_t E zeta_t<infinity.

**Conditional conclusion:** f(X_t) converges and every accumulation point is
Clarke stationary almost surely. The claim is about cluster points, not
convergence of the whole sequence, a global optimum, or B-stationarity.

### Proof framework

From (7), (8), the safe-branch inequality and smoothness, uniform constants
c,C>0 give the pathwise inequality

$$
f(X_{t+1})\le f(X_t)-c\alpha_t r_{h_t}(X_t)^2
+C\alpha_t e_t+C\alpha_t^2/h_t+\alpha_t\zeta_t.
\tag{9}
$$

All iterates are feasible and the trial norm is at least R. In particular,
||X_(t+1)-X_t||_F <= (1+sqrt(r))B alpha_t; this follows from the Lipschitz
bound | ||Y||_2-R | <= alpha_t||D||_2 and the exact radial correction.
By Tonelli's theorem the nonnegative error series in (9) is almost surely
summable. Boundedness below of f yields convergence of f and
sum alpha_t r_(h_t)^2<infinity.

It remains to exclude nonstationary cluster points rather than just claim
liminf r_h=0. At a point with true top multiplicity k,

$$
N^C_{C_R}(X)=\{U_k A V_k^\top:A=A^\top\}.
\tag{10}
$$

To see (10), limiting normals are supported in this paired active symmetric
block. Approaching X through simple-top points gives both signs of every
rank-one symmetric block aa^T. Their closed convex conic hull is all of
Sym(k), which is the Clarke normal. At k=1 this is the usual normal line.

Suppose X_j->X_* and h_j->0. Every limit of selected singular pairs lies
in the true active subspace of X_*. Taking a subsequence with fixed selected
dimension, r_(h_j)(X_j)->0 forces grad f(X_*) into the right-hand side of
(10). Consequently a non-Clarke-stationary X_* has a neighborhood where
r_(h_t) stays bounded below for all sufficiently large t. Repeated entries
into a smaller neighborhood either stay there forever, or require infinitely
many excursions of a fixed distance. The step bound makes either case
contradict sum alpha_t r_(h_t)^2<infinity. This proves the stated cluster-point
conclusion under Conditions 1-4.

### One implementable stochastic schedule

For conditionally unbiased samples of uniformly bounded conditional variance
and finite-second-moment initialization, current-sample EMA with
gamma_t asymptotic to t^(-b) and alpha_t asymptotic to t^(-a), 0<b<a<1,
has the sufficient tracking estimate

$$
\mathbb E e_t=O(t^{b-a}+t^{-b/2}).
$$

This follows by unrolling EMA: the gradient drift is O(alpha_t/gamma_t)
using the bounded step, and the martingale noise contribution is O(sqrt(gamma_t)).
Take h_t asymptotic to t^(-d). For example,

$$
(a,b,d)=(0.9,0.5,0.3)
$$

satisfies 2a-b>1, a+b/2>1, 2a-d>1 and a>d. Constants can enforce
alpha_t B<=h_t/2 from the start. These are sufficient asymptotic schedules,
not recommended LLM hyperparameters. Fixed beta=0.9, Nesterov momentum and
a finite cosine schedule are not covered by this tracking argument.

## 6. What Approximation Would Need

There are three separate numerical errors; a small estimated gap error does
not by itself control all three.

| Error | What the argument needs | What is not justified |
|---|---|---|
| Direction/subspace error | At a feasible base, distance eps_t from a valid bounded direction with sum alpha_t eps_t finite is sufficient with exact radial return | Fixed K10 automatically makes this series finite |
| Missing cluster directions | Complete coverage of the selected window, or a new bound for the omitted directions | A fixed top-4 cap uniformly covers all corners |
| Radial norm error | Feasibility or a separately analyzed feasible-shadow construction, plus controlled accumulated error | Approximate normalization is exact feasibility |

For the first row, radial return is locally Lipschitz away from zero, so
changing a bounded trial direction by eps_t changes the returned point by
O(alpha_t eps_t). But a small error in the *value* of the gap need not imply
a small error in the direction, particularly at a branch or anchor change.

The anchor is another important computational limitation. A good approximation
of a whole top subspace does not automatically identify a genuinely active
individual direction. Near-coincident unequal singular values can make the
anchor hard to resolve. One can instead impose H(D)=0 for the entire selected
block, which needs no distinguished anchor, but that is the more conservative
cluster-tangent construction already considered in the earlier study.

This note does **not** solve the cost of accurate spectral information. It
isolates a geometry that makes a radial-only proof possible. Exact spectral
calculations are used in the local checks to avoid confusing geometric failure
with spectral-estimation error.

The condition alpha_t B<=h_t/2 is sufficient and potentially very conservative.
A numerical relative-gap threshold of 2e-4 and a learning-rate label of 0.015
must not be presumed to satisfy it: h_t=R times the relative threshold, while
alpha_t||D_t||_2 must use the *actual* per-matrix update, including all learning
rate/radius scaling. Small cross-subspace coupling may yield a tighter bound,
as in the earlier finite-step note, but that must be quantified rather than
assumed. If the complete selected window is large, this theory does not promise
a cheap fixed-top-4 algorithm.

The proposition concerns a spectral-sphere-constrained problem. It extends
to a finite product of these spheres with suitable uniform constants and
bounded feasible directions. An LLM also updates unconstrained embeddings,
heads and other parameters with a separate optimizer. Their boundedness,
descent/tracking properties, gradient clipping and mixed-precision errors need
additional analysis before claiming a theorem for the entire training system.

## 7. Local Evidence and a Necessary Limitation

The FP64 script checks 324 random algebra cases, with rectangular and square
matrices, radii 0.1/1/10, and both equal and unequal selected spectra. Tests
cover cone membership, idempotence, positive homogeneity, projection identity,
nonexpansiveness, preservation of free directions, finite-step compression,
the radial bound, and the normalized-momentum inequality.

For the same 360 small quadratic states as the earlier audit, the exact-corner
subset contains 72 cases per momentum alignment. All directions in the table
use the earlier matched half-D0 trial length unless marked otherwise. This
finite-length experiment is not the asymptotic normalized-direction theorem.

| Operation | Loss increases, M=G | Loss increases, cos(M,G)=0.5 |
|---|---:|---:|
| Half MCSD-TP + radial | 10/72 | 24/72 |
| Normalized ambient + radial | 8/72 | 24/72 |
| Complete true-cone correction + radial | 0/72 | 4/72 |
| New anchored-face correction + radial | 0/72 | 4/72 |
| Normalized ambient + metric projection | 0/72 | 4/72 |

For M=G, mean delta f is -0.00922382 for the anchored candidate and
-0.00934252 for the full-cone candidate. The anchored candidate is not better
on this test; its benefit is the controlled near-cluster model. None of the
72 anchored cases has a zero trial direction. These are constructed
quadratics, not LLM training-loss evaluations or a statistical convergence test.

**Why the conclusion is only Clarke stationarity:** at X=I_2, G=diag(1,0),
and anchor e_1, (3) returns zero, while the full Bouligand-cone projection
of -G equals diag(-1,0) and admits descent. The anchored method can stop at
a Clarke-stationary but B-nonstationary point. The script explicitly records
anchored norm 0 versus full-cone norm 1; this is not hidden by averaging.

Reproduce with Python and PyTorch, from the repository root:

```bash
python scripts/local_anchored_cone_radial.py \
  --output docs/results/anchored_cone_radial_20261118/local_checks.json
```

The script runs on CPU, uses float64, and imports the existing local-study
helpers. Assertions check the claimed finite-dimensional inequalities; these
checks complement, but do not replace, the proofs and asymptotic assumptions.

## 8. Relation to the Existing Experiments

The Nov. 18 matched width-256/3B results favor the simple near-only half-D0
control (3.423857) over the matched MCSD-TP control (3.429645) and the earlier
warm ambient-radial hybrid (3.432013). See the
[completed-run report](warm_radial_controls_3b_20261117.md). Therefore the new
candidate should not skip a matched half-D0 comparator or be presented as an
established improvement. The earlier A2/K-LMO candidates already tested
finite-step corrections; the present construction is not evidence that an
extra LMO suddenly becomes useful.

The next justified local comparison is actual snapshots at a common base:
half-D0, ambient radial, unanchored cluster correction, anchored correction,
and A2; record full-return objective on a replayable model/batch, actual step
length and required cluster capacity. Existing matrix/momentum snapshots do
not contain the model and batches needed to reconstruct genuine LLM loss.
Only after that comparison should one consider a small paired H20 pilot.

## References and Scope

- [Yang and Lai, MCSD](https://arxiv.org/abs/2601.21487v2): source of the
  user-chosen-norm smooth branch and the distinction between MCSD and MCSD-TP.
  The public MCSD--PGD result does not establish this new radial-only hybrid.
- [Olikier and Waldspurger, PGD](https://arxiv.org/abs/2403.02530): stationarity
  results for the metric-projection algorithm under its prescribed rules;
  they must not be transferred by renaming radial normalization a projection.
- [Levin, Kileel and Boumal](https://arxiv.org/abs/2107.03877): a primary source
  on cross-stratum stationarity pitfalls, demonstrated for bounded-rank
  matrices. It motivates checking the limiting stationarity measure, rather
  than supplying a theorem for the present spectral-sphere construction.

Equations (2)-(9) and the conditional proposition are the derivation in this
research note. They should receive a separate mathematical review before
being promoted to a submitted-paper theorem.
