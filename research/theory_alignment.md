# Theory alignment

- On the Frobenius sphere, FP32 norm scaling is the metric projection.
- On full-rank scaled Stiefel, `a U V^T` is the local metric projection and a
  polar retraction. BF16 finite-step `msign` with FP32 polishing approximates
  this map; the exact-return theorem does not automatically cover its error.
- On the spectral sphere with a simple leading singular value, `u1 v1^T`
  defines the smooth tangent normal. At a tie there is no unique such normal.
  Continued training with one numerical SVD/PI vector is an empirical
  extension, not a proof of the smooth theorem.
- Exact spectral clipping and radial return are different maps. PI/top-k
  clipping is not a certified metric projection; a practical result must use
  that label and report constraint residuals.
