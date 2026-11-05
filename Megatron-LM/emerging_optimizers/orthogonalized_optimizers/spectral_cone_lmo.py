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
    *,
    blend: float = 0.0,
    match_reference_fro: bool = False,
) -> torch.Tensor:
    """Select a cone face from descent input z, use one LMO, and return to it.

    Let A=sym(U.T z V), B=P_{lambda_max=0}(A), Q=z+U(B-A)V.T.
    P projects onto the negative eigenspace of B. For S=LMO(Q), return
    D=S-U[H(S)-(P H(S) P)_-]V.T. With an exact LMO,
    <z,D> = N_*(Q)-<B,(P H(S) P)_+> >= N_*(Q).
    This does not imply N(D)<=1 for an arbitrary norm. With approximate
    matrix sign, the right-hand dual norm is replaced by <Q,S>.

    Optional blending happens BEFORE the final face correction. Positive
    rescaling then matches the unblended corrected direction's Frobenius
    length, not the MCSD branch's length. This is not an exact norm LMO.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be between zero and one")
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

        def correct_to_face(candidate):
            h = u.T @ candidate @ v
            h = .5 * (h + h.T)
            if q == 1:
                negative = torch.zeros_like(h)
            else:
                c = p @ h @ p
                values, vectors = torch.linalg.eigh(.5 * (c + c.T))
                negative = (vectors * values.clamp(max=0)) @ vectors.T
                # Retain the selected face despite eigensolver roundoff.
                negative = p @ negative @ p
            return candidate + u @ (negative - h) @ v.T

        reference = correct_to_face(s)
        if blend == 0.0:
            return reference
        # Reuse the SAME oracle evaluation; no second NS or matrix SVD.
        tiny = torch.finfo(dtype).tiny
        gradient = projected * (
            torch.linalg.vector_norm(s) / torch.linalg.vector_norm(projected).clamp_min(tiny)
        )
        direction = correct_to_face((1.0 - blend) * s + blend * gradient)
        if match_reference_fro:
            direction = direction * (
                torch.linalg.vector_norm(reference) / torch.linalg.vector_norm(direction).clamp_min(tiny)
            )
        return direction
    finally:
        if z.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def matrix_sign_cone_direction(z, u, v, *, msign, steps=8, blend=0.0,
                               match_reference_fro=False):
    """Production adapter: the same normalized BF16 NS LMO as MCSD-TP."""
    def lmo(q):
        normalized = q / torch.linalg.vector_norm(q).clamp_min(1e-7)
        return msign(normalized, steps=steps).float()

    return spectral_cone_lmo_direction(
        z, u, v, lmo, blend=blend, match_reference_fro=match_reference_fro,
    )
