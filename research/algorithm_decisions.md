# Algorithm decisions

## 2026-09-23, first implementation

- Keep the handoff's current-sample ambient EMA, rank-aware spectral-ball LMO,
  fixed initialization scale, and distinct MCSD/MCSD-TP return contracts.
- Use exact SVD geometry as a reference and BF16 Polar Express as the practical
  direction. Stiefel practical returns are explicitly approximate.
- Continue spectral runs through small top gaps at the user's request, record
  the gap, and restrict theory claims to the smooth subset.
- Do not promote PI10/top-8 spectral clipping to the formal main method on
  current local evidence. It remains a labelled pilot until its constraint
  error, cost and paired validation loss are measured on H20.

Next minimal control: hold initialization, data order, optimizer direction
and LR fixed; vary only the Stiefel return (`exact` versus `ns`) for at most
100 steps, measuring constraint residual and synchronized training time.
