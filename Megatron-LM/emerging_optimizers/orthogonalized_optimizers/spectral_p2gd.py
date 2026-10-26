"""Euclidean tangent-cone projection on a strict spectral-norm sphere.

The caller supplies the COMPLETE paired top singular subspace of the current
returned matrix. Near-top singular values must not be promoted by the PGD
branch threshold. This module does not normalize directions or return points.
"""

import torch


@torch.no_grad()
def project_spectral_sphere_cone(
    direction: torch.Tensor, u_top: torch.Tensor, v_top: torch.Tensor
) -> torch.Tensor:
    """Project a direction onto {D: lambda_max(sym(U.T D V)) = 0}.

    Float64 is retained for oracle tests; lower-precision inputs use float32.
    U and V must be orthonormal and paired with identical rotations.
    """
    if direction.ndim != 2 or u_top.ndim != 2 or v_top.ndim != 2:
        raise ValueError("expected one matrix and two column bases")
    q = u_top.shape[1]
    if q < 1 or q != v_top.shape[1]:
        raise ValueError("a nonempty complete paired top subspace is required")
    if u_top.shape[0] != direction.shape[0] or v_top.shape[0] != direction.shape[1]:
        raise ValueError("top subspace dimensions do not match the matrix")
    dtype = torch.float64 if direction.dtype == torch.float64 else torch.float32
    z = direction.to(dtype=dtype)
    u = u_top.to(device=z.device, dtype=dtype)
    v = v_top.to(device=z.device, dtype=dtype)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if z.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
        a = u.T @ z @ v
        a = (a + a.T) * 0.5
        if q == 1:
            b = torch.zeros_like(a)
        else:
            values, vectors = torch.linalg.eigh(a)  # Ascending.
            clipped = values.clamp(max=0)
            clipped[-1] = 0  # Boundary of the sphere, including all-negative A.
            b = (vectors * clipped) @ vectors.T
        return z + u @ (b - a) @ v.T
    finally:
        if z.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
