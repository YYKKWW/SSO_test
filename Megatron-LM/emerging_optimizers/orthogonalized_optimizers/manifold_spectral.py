"""SVD-free practical spectral-sphere directions and returns.

These finite-iteration estimates are not certified projections.  Exact SVD
checks belong to an independent, low-frequency diagnostic run.
"""

import torch


@torch.no_grad()
def leading_triplet(
    matrix: torch.Tensor, *, steps: int = 10, matmul_dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cold two-sided power iteration with an FP32 Rayleigh quotient."""
    if matrix.ndim != 2 or matrix.dtype != torch.float32:
        raise ValueError("expected one FP32 matrix")
    if steps < 1:
        raise ValueError("steps must be positive")
    if matmul_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("matmul dtype must be BF16 or FP32")
    work = matrix.to(matmul_dtype)
    vector = torch.arange(1, matrix.shape[1] + 1, device=matrix.device, dtype=torch.float32)
    vector = torch.sin(vector) + 1.5
    v = torch.nn.functional.normalize(vector.to(matmul_dtype), dim=0)
    for _ in range(steps):
        u = torch.nn.functional.normalize(work @ v, dim=0, eps=1e-12)
        v = torch.nn.functional.normalize(work.T @ u, dim=0, eps=1e-12)
    u = torch.nn.functional.normalize((matrix @ v.float()), dim=0, eps=1e-12)
    v = v.float()
    sigma = u @ matrix @ v
    return sigma, u, v


@torch.no_grad()
def spectral_tangent_project_pi(
    matrix: torch.Tensor,
    vector: torch.Tensor,
    *,
    steps: int = 10,
    matmul_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    _, u, v = leading_triplet(matrix, steps=steps, matmul_dtype=matmul_dtype)
    normal = torch.outer(u, v)
    return vector - torch.sum(normal * vector) * normal


@torch.no_grad()
def spectral_return_pi(
    trial: torch.Tensor,
    radius: float,
    *,
    mode: str,
    steps: int = 10,
    rank: int = 8,
    matmul_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Approximate radial return or rank-k singular-value clipping."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if mode == "radial":
        sigma, _, _ = leading_triplet(trial, steps=steps, matmul_dtype=matmul_dtype)
        if sigma.item() <= 0:
            raise ValueError("zero or invalid spectral trial")
        return trial * (radius / sigma)
    if mode != "projection" or rank < 1:
        raise ValueError("invalid spectral approximation")
    updated = trial.clone()
    residual = trial.clone()
    for index in range(min(rank, min(trial.shape))):
        sigma, u, v = leading_triplet(residual, steps=steps, matmul_dtype=matmul_dtype)
        if sigma.item() <= 0:
            break
        outer = torch.outer(u, v)
        desired = radius if index == 0 else min(sigma.item(), radius)
        updated.add_(outer, alpha=desired - sigma.item())
        residual.add_(outer, alpha=-sigma.item())
    return updated
