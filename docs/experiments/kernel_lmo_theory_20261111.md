# A Step-Aware MCSD Hybrid: Theory and Deployable Candidates

Research note, 2026-11-11. This is a proposed extension, not a revision of the
paper's proved MCSD--PGD theorem. No production optimizer or H20 job is changed.

[Previous audit](step_aware_hybrid_review_20261111.md) |
[Previous local results](step_aware_hybrid_results_20261111.md) |
[New reference implementation](../../scripts/local_kernel_lmo_study.py) |
[Tests](../../scripts/test_local_kernel_lmo_study.py)

## 1. What Transfers From the Paper

The local manuscript `main1.tex` uses the current stochastic sample immediately:
update ambient momentum, project its input, call an arbitrary-norm LMO, and
return a feasible point. MCSD-TP also projects the LMO output. The effective
direction set is the image of the norm ball under the tangent projector, not
the intersection of that ball with the tangent space.

We retain these three ideas: **current-sample tracking, an ordinary ambient
LMO, and a first-order return bound**. We do not assume that the paper's smooth
uniform return bound holds across the spectral sphere's corners. The paper's
off-safe-region Euclidean projection is also not silently replaced by radial
normalization. A new near-region model and a new return estimate are needed.

The aim here is conditional **Clarke stationarity** for a stochastic,
no-line-search reference algorithm. This is weaker than the existing
deterministic MCSD--PGD B-stationarity result. At a multiple top singular value,
Clarke stationarity can hold despite an available feasible descent direction.

## 2. Keep the Finite-Step Normal Correction, Change Only Free Directions

Let C_R={X: ||X||_2=R}, R>0. At an exactly feasible real matrix X, select every
singular value in [R-h,R], with 0<h<R. Write their paired bases as U,V and their
values as S=diag(s_1,...,s_q). Define

$$
H(D)=\operatorname{sym}(U^\top D V),\quad H^*(A)=UAV^\top,
\quad L=\ker H,\quad P_L=I-H^*H.
$$

When q=1 use the existing MCSD-TP direction
T=P_L LMO_N(-P_L M), with the LMO maximizing its displayed input. This is the
same as minimizing the inner product with P_L M in the paper's convention.
When q>1 use the following construction, with the **actual** positive step alpha:

$$
K=\{D:\lambda_{\max}(S+\alpha H(D))=R\},\qquad
Z=-M/\|M\|_F,\qquad F\in P_K^F(Z).
$$

Set F=0 when M=0. This is A2's normalized-momentum reference. It has an explicit
small-matrix implementation. If

$$
A=H(Z)-(RI-S)/\alpha=O\operatorname{diag}(a_1,\ldots,a_q)O^\top,
\quad a_1\ge\cdots\ge a_q,
$$

then

$$
B=O\operatorname{diag}(0,\min(a_2,0),\ldots,\min(a_q,0))O^\top,
\qquad F=Z+H^*(B-A).
$$

The kernel is exactly the part we can change without undoing this correction:

$$
\boxed{K+L=K.}
$$

This is more useful than merely knowing F belongs to a tangent cone. It
preserves the complete selected finite-step spectral model, including the
nonzero gaps R-s_i, at the actual trial step.

### Candidate K-LMO

Compute the ordinary MCSD direction in the free space:

$$
T=P_L\operatorname{LMO}_N(-P_LM),\qquad
E=\begin{cases}
\theta\|F\|_F\,T/\|T\|_F,&T\ne0,\ \langle M,T\rangle\le0,\\
0,&\text{otherwise},
\end{cases}\qquad D=F+E.
$$

Here theta is a fixed bounded nonnegative constant; the local study tests
0.5 and 1. The small scalar check uses momentum, not a function evaluation.
For an exact LMO it is redundant. For an approximate matrix sign it prevents
adding an ascent component, provided the kernel projection is accurate.

Both branches use Y=X+alpha D and X_+=RY/||Y||_2. Cache the spectrum scaled
by the same return factor. There is no projection onto the full spectral
sphere in this near branch, and no second projection of the LMO onto K.

**What this is not:** it is not standard P2GD, not an exact constrained
N-unit-ball LMO, and not a proof that the existing fixed-top-4 approximate
training code converges. The auxiliary norm N chooses the LMO shape; the
explicit Frobenius budget controls its additional amplitude. A bounded scalar
multiple does not change that LMO shape, but does change its unit-step meaning.

### Algebraic guarantee

Orthogonality of H*H and P_L, and comparison with P_LZ in the nearest-point
problem, give

$$
2\langle Z,F\rangle\ge\|F\|_F^2+\|P_LZ\|_F^2,
\quad \|F\|_F\le2,
\quad P_LF=P_LZ.
$$

Thus the new direction satisfies

$$
H(D)=H(F),\qquad D\in K,\qquad
\langle M,D\rangle\le\langle M,F\rangle
\le-\frac{\|P_LM\|_F^2}{2\|M\|_F},
\quad \|D\|_F\le2(1+\theta).
$$

For an exact LMO the extra decrease equals
-theta ||F||_F N_*(P_LM)/||T||_F whenever T is nonzero. These claims require
no internal gap s_1-s_2 and hold for any ambient norm with a measurable LMO
selection. At M=0 all directions are zero and the divided expression is not
evaluated.

This construction avoids two pitfalls of A4: an LMO of F can modify its
already-corrected top block, and accepting a better inner product can also
accept a much larger direction. K-LMO preserves that block and bounds the
additional length relative to F. It still **does not** promise a better
objective after the complete radial return.

Do not scale D after correction. Also do not freely interpolate two different
members of K: K is generally nonconvex. Adding a vector in L is safe because
it leaves H(D) unchanged, not because arbitrary convex combinations are safe.

### A fundamental amplitude obstruction

There is an important limit on any proposal protected against A2. Its exact
nearest-point optimality gives, for every D in the same K,

$$
\langle M,D-F\rangle
\ge-\frac{\|M\|_F}{2}\big(\|D\|_F^2-\|F\|_F^2\big).
$$

Therefore a **strictly better momentum inner product requires a strictly
larger Frobenius norm**. This applies to A4 and K-LMO alike. It is impossible
to prove that either beats A2's momentum model at the same or a smaller
Frobenius budget. Different radial coupling can still change the actual
returned objective, but that must be checked separately.

The inexpensive competing mechanism is to increase the normalized input
before correction: F_a=P_K(-a M/||M||_F), with a=1.5 or 2 in the study.
This retains K and requires only one small eigensolve, with no near-branch
LMO. It is not post-scaling F. Comparing it with K-LMO is essential before
attributing an improvement to an arbitrary-norm direction rather than extra
amplitude. Equal a and 1+theta are equal worst-case norm budgets, not exactly
equal realized step lengths.

## 3. A Tighter Complete-Return Bound

Let Gamma=R-s_(q+1), or conservatively Gamma=R if q=min(m,n). Introduce
the symmetric dilation A(D)=[[0,D],[D^T,0]] and W=(U;V)/sqrt(2). Define

$$
C=(I-WW^\top)\mathcal A(D)W,\qquad c=\|C\|_2,\qquad
\delta=\alpha\|D\|_2,\qquad g=\Gamma-\delta.
$$

The selected block of A(X+alpha D) has largest eigenvalue exactly R. The
complement has largest eigenvalue at most R-g. A two-block Rayleigh quotient
therefore proves, even when g is nonpositive,

$$
0\le\|X+\alpha D\|_2-R
\le \omega:=\frac{\sqrt{g^2+4\alpha^2c^2}-g}{2}.
$$

For g>0 this is at most alpha^2 c^2/g. If delta<=Gamma/2, it is no larger
than the earlier 2 alpha^2 ||D||_2^2/Gamma bound. A small cross-subspace
coupling can substantially reduce the remainder without increasing q.
However, this does not automatically reduce the q needed for a chosen
sufficient certificate: complement separation is still a distinct issue.

The coupling can be computed without constructing the dilation or its
complement basis:

$$
C^\top C=\tfrac12\big[(DV)^\top(DV)+(D^\top U)^\top(D^\top U)\big]-H(D)^2.
$$

This is a q-by-q Gram matrix. Small negative eigenvalues from roundoff must
be distinguished from a mathematical PSD statement. The local audit uses
exact paired bases and exact operator norms. In production, ||D||_F is a
cheap valid upper bound for ||D||_2, whereas an uncertified PI estimate is
not necessarily an upper bound. **This audit is not an online gate.**

For Y=X+alpha D and X_+=RY/||Y||_2, the exact radial correction is
e=-(1-R/||Y||_2)Y. If grad f is L_f-Lipschitz on the ball, then the tighter
signed objective estimate is

$$
f(X_+)-f(X)\le
\alpha\langle\nabla f(X),D\rangle
+\frac{\omega}{R}\max\{0,-\langle\nabla f(X),Y\rangle\}
+\frac{L_f}{2}\|X_+-X\|_F^2.
$$

The plus sign in the second term is intentional: radial shrinkage helps when
the gradient has nonnegative inner product with Y and can hurt otherwise.
A negative upper bound certifies complete descent. A positive bound is
inconclusive. Stochastic normalized updates need not decrease f every step.

## 4. Removing the Bounded-Sample Assumption

The previous sufficient-condition note assumed uniformly bounded stochastic
samples. That is stronger than the paper's oracle assumption and is not
necessary for the bounded directions above.

Let g_X=grad f(X), choose G>=max(1,sup_{C_R}||g_X||_F), put
e=||M-g_X||_F, r_h=||P_L g_X||_F, and let ||D||_F<=B_0. The near-branch
momentum bound implies

$$
\boxed{\langle g_X,D\rangle
\le-\frac{r_h^2}{8G}+(B_0+1/4)e.}
$$

Proof: write v=||P_LM||_F. If e<=G, then ||M||<=2G and
v^2>=r_h^2/2-e^2, so
v^2/(2||M||)>=r_h^2/(8G)-e/4. If e>G the same right-hand side is
nonpositive, and the left-hand side is nonnegative. Finally replace M by
g_X in the inner product using B_0 e. The case M=0 follows directly and
needs no division. This argument permits unbounded samples with finite
conditional variance.

For q=1 the original MCSD-TP inequality applies. If a_N bounds ||s||_F for
N(s)<=1 and b_N>0 satisfies N_*(z)>=b_N||z||_F, then

$$
\langle g_X,D\rangle\le
-\frac{b_N}{G}r_h^2+2a_N e+\zeta,
$$

where zeta is the paper's feasible inexact-LMO error. Combine the two bounds
with uniform constants. The near-branch LMO can instead use the scalar
nonascent check, so its baseline decrease does not require an exact LMO.
An approximate top subspace is a different error and is not covered by this
observation.

## 5. Conditional Convergence Framework

### Assumptions

1. Exact feasible iterates on C_R; f is C1 with Lipschitz gradient on its
   convex hull. The sphere is compact.
2. Complete selected window h_t, no artificial top-q truncation; paired exact
   spectral data and exact finite-step model correction. All choices measurable.
3. Current-sample EMA with conditional unbiasedness and bounded conditional
   variance, as in `main1.tex`. No unbiasedness of the direction is assumed.
4. Uniform bounded directions; alpha_t B_0<=h_t/2; h_t decreases to zero;
   sum alpha_t=infinity; sum alpha_t^2/h_t<infinity.
5. sum alpha_t E||M_t-grad f(X_t)||<infinity and, for the safe inexact LMO,
   sum alpha_t E zeta_t<infinity.

Then, for constants independent of t,

$$
f(X_{t+1})\le f(X_t)-c\alpha_t r_{h_t}(X_t)^2
+C\alpha_t e_t+C'\alpha_t^2/h_t+\alpha_t\zeta_t.
$$

The radial estimate follows from Gamma>=h_t and bounded D. The displacement
is at most (1+sqrt(min(m,n))) alpha_t B_0. Consequently the same tracking
decomposition used in the manuscript is available.

Tonelli makes the nonnegative noise-error series summable almost surely.
Adding its remaining tail to f produces a bounded-below decreasing sequence.
Thus f(X_t) converges and sum alpha_t r_h_t(X_t)^2 is finite almost surely.

For a subsequence X_j -> X*, h_j -> 0 with r_h_j(X_j) -> 0, select convergent
paired bases at fixed q. The selected block lies in the true top space at
X*, and the limiting gradient belongs to its symmetric paired normal block.
It therefore belongs to the Clarke normal. At any non-Clarke-stationary
limit candidate, this compactness argument provides a positive local lower
bound on r_h for all sufficiently small h. Bounded displacement and
sum alpha=infinity then rule out both staying in that neighborhood forever
and infinitely many fixed-radius excursions through it. **All cluster points
are Clarke stationary almost surely.**

This is a sufficient-condition proof for the newly specified exact algorithm,
not a citation-based transfer of the original PGD theorem. It does not show
convergence of the entire point sequence, B-stationarity, global optimality,
or an improvement over MCSD-TP in a finite training budget.

### Schedules consistent with the paper's tracking argument

Let the new-sample EMA weight be gamma_t proportional to t^(-b), alpha_t
proportional to t^(-a), and h_t proportional to t^(-d), with suitable initial
constants. Bounded displacements and martingale-difference variance give

$$
\mathbb E e_t=O(t^{b-a}+t^{-b/2})+\text{initialization decay}.
$$

It is enough that 0<b,d<a<1, 2a-b>1, a+b/2>1 and 2a-d>1.
The previous example (a,b,d)=(0.9,0.5,0.3) still works, now without a
uniform bound on every gradient sample.

For a fixed finite horizon, constant gamma and alpha give instead the
weighted surrogate bound

$$
\min_{t<T}\mathbb E r_h(X_t)^2
\le C\left[\frac{1}{T\alpha}+\frac{\alpha}{h}
+\frac{\alpha}{\gamma}+\sigma\sqrt\gamma
+\frac{e_{\rm init}}{T\gamma}+\bar\zeta\right].
$$

For example, alpha~T^(-3/4), gamma~T^(-1/2), h~T^(-1/4) yields
O(T^(-1/4)) for the **squared surrogate residual**, when other constants and
inexactness are controlled. This is not a rate for the current true Clarke
residual, and not the original paper's smooth-gradient rate. It also is not
an all-cluster-points theorem for fixed-beta=0.9 Nesterov and cosine decay.

### What an arbitrary closed-set extension would require

The algebra transfers to a model K_x,alpha and a linear free space L_x if
0 is in K, K+L=K, and P_K is computable. The projection comparison still
provides baseline descent, and adding an LMO direction in L preserves the
model. But the geometric part additionally needs a feasible-return bound,
bounded displacement and a shrinking-model stationarity detection property.
Stratification alone does not supply these. The spectral-sphere construction
above verifies the required model and return geometry explicitly.

## 6. Practical Costs and Unresolved Approximation

| Operation | A4 | K-LMO near branch |
|---|---|---|
| Small model correction | Two q-by-q eigensolves | One q-by-q eigensolve |
| Ordinary LMO | On the corrected F | On -P_L M, as in MCSD-TP |
| Extra amplitude | Can grow with LMO shape | At most theta times ||F||_F |
| Preserve corrected top block | Not automatically | Exactly, with exact bases |
| Full trial return | One spectral solve | One spectral solve |

Both still need top spectral information and a radial norm estimate. A small
q correction does **not** remove that dominant cost. The present CPU study
uses full SVD and exact polar LMOs to isolate geometry; its time is not a
CUDA performance forecast.

The prospective implementation is cached warm block iteration, FP32 small
Ritz data, the small correction, and one approximate radial norm computation.
The LMO may reuse the existing Newton--Schulz code. Before claiming a theorem
for this implementation, one must control (i) missed selected directions,
(ii) errors in H and E, (iii) model-boundary error and (iv) radial feasibility.
Simply declaring a small fixed error or adding a summable scalar to the proof
does not establish these conditions. A feasible shadow sequence or a robust
inexact-model theorem is still needed. No costly online certificate is added
by this research note.

The tighter coupling bound may explain why a small block works empirically.
It is not permission to count a truncated cluster as a complete true cluster.
The two-stage decision remains: test direction geometry locally, then inspect
real width-256 snapshot capacity, before changing the production solver.

## 7. References and Attribution

The conditional lemmas and K-LMO construction above are new derivations in
this note, not results attributed to the following papers.

- [Yang and Lai, MCSD v2](https://arxiv.org/abs/2601.21487v2): the original
  ambient-LMO/smooth-return framework and projection-based closed-set hybrid.
- [Levin, Kileel and Boumal](https://arxiv.org/abs/2107.03877): why stationarity
  and limits across nonsmooth strata need care. Their bounded-rank results
  do not prove the spectral-sphere hybrid here.
- The user's local `main1.tex`, shared-update, return-model and stochastic
  oracle assumptions: the current-sample EMA and finite-time tracking setup
  to which this extension is compared.

Local evidence is recorded separately in
[the paired-study report](kernel_lmo_results_20261111.md). Numerical checks
can refute a proposed inequality; they cannot establish an asymptotic theorem.
