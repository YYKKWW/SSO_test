"""Single-LMO extension of MCSD-TP to a selected top spectral face.

For a complete, exactly multiple top subspace this constructs a tangent-cone
direction. A near cluster is an experimental surrogate, not an exact cone.
The caller supplies the norm LMO; no original-matrix SVD is performed here.
"""

from collections.abc import Callable

import torch


@torch.no_grad()
def spectral_cone_lmo_direction(
    z: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    lmo: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Select a cone face from descent input z, use one LMO, and return to it.

    Let A=sym(U.T z V), B=P_{lambda_max=0}(A), Q=z+U(B-A)V.T.
    P projects onto the negative eigenspace of B. For S=LMO(Q), return
    D=S-U[H(S)-(P H(S) P)_-]V.T. With an exact LMO,
    <z,D> = N_*(Q)-<B,(P H(S) P)_+> >= N_*(Q).
    This does not imply N(D)<=1 for an arbitrary norm. With approximate
    matrix sign, the right-hand dual norm is replaced by <Q,S>.
    """
    if z.ndim != 2 or u.ndim != 2 or v.ndim != 2:
        raise ValueError("expected a matrix and two column bases")
    q = u.shape[1]
    if q < 1 or v.shape[1] != q or u.shape[0] != z.shape[0] or v.shape[0] != z.shape[1]:
        raise ValueError("expected nonempty paired top bases matching the matrix")
    dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    z = z.to(dtype=dtype)
    u, v = u.to(z), v.to(z)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if z.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
        a = u.T @ z @ v
        a = .5 * (a + a.T)
        if q == 1:
            b, p = torch.zeros_like(a), torch.zeros_like(a)
        else:
            values, vectors = torch.linalg.eigh(a)  # Ascending.
            clipped = values.clamp(max=0)
            clipped[-1] = 0
            b = (vectors * clipped) @ vectors.T
            p = (vectors * (clipped < 0).to(dtype)) @ vectors.T
        projected = z + u @ (b - a) @ v.T
        s = lmo(projected).to(z)
        if s.shape != z.shape:
            raise ValueError("LMO output must match the matrix shape")
        h = u.T @ s @ v
        h = .5 * (h + h.T)
        if q == 1:
            negative = torch.zeros_like(h)
        else:
            c = p @ h @ p
            values, vectors = torch.linalg.eigh(.5 * (c + c.T))
            negative = (vectors * values.clamp(max=0)) @ vectors.T
            # Retain the selected face despite eigensolver roundoff.
            negative = p @ negative @ p
        return s + u @ (negative - h) @ v.T
    finally:
        if z.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def matrix_sign_cone_direction(z, u, v, *, msign, steps=8):
    """Production adapter: the same normalized BF16 NS LMO as MCSD-TP."""
    def lmo(q):
        normalized = q / torch.linalg.vector_norm(q).clamp_min(1e-7)
        return msign(normalized, steps=steps).float()

    return spectral_cone_lmo_direction(z, u, v, lmo)
