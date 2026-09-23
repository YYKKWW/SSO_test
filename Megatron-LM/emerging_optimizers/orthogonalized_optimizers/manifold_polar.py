"""Practical BF16 Polar Express matrix sign for the manifold study.

The iteration and stabilized coefficients follow the authors' implementation
at https://github.com/thinking-machines-lab/manifolds/blob/main/src/msign.py.
Unlike the exact SVD reference, finite steps give an approximate polar factor.
"""

import torch

_COEFFICIENTS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
)


@torch.no_grad()
def polar_express_msign(
    matrix: torch.Tensor, *, steps: int = 10, polish_steps: int = 0
) -> torch.Tensor:
    """Approximate full matrix sign; optional FP32 polar polishing for returns.

    The polynomial iteration uses BF16 matrix products.  Polishing uses the
    cubic Newton-Schulz map around singular value one, so it should only be
    applied after Polar Express has brought a full-rank matrix close to polar.
    """
    if matrix.ndim != 2 or matrix.dtype != torch.float32:
        raise ValueError("Polar Express expects one FP32 matrix")
    if steps < 1 or polish_steps < 0:
        raise ValueError("invalid iteration count")
    if not torch.isfinite(matrix).all():
        raise FloatingPointError("nonfinite Polar Express input")
    magnitude = torch.linalg.vector_norm(matrix)
    if magnitude.item() == 0:
        return torch.zeros_like(matrix)
    transpose = matrix.shape[0] > matrix.shape[1]
    x = (matrix.T if transpose else matrix) / (magnitude * 1.01)
    x = x.to(torch.bfloat16)
    for i in range(steps):
        a, b, c = _COEFFICIENTS[min(i, len(_COEFFICIENTS) - 1)]
        if i < len(_COEFFICIENTS) - 1:
            a, b, c = a / 1.01, b / 1.01**3, c / 1.01**5
        gram = x @ x.T
        polynomial = c * gram
        polynomial.diagonal().add_(b)
        polynomial = polynomial @ gram
        polynomial.diagonal().add_(a)
        x = polynomial @ x
    x = x.float()
    for _ in range(polish_steps):
        gram = x @ x.T
        x = 0.5 * (3.0 * x - gram @ x)
    result = x.T if transpose else x
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite Polar Express output")
    return result
