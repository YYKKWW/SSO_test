"""Fixed-budget FP32 warm block iteration; no residual gate or full-SVD fallback."""

import torch


@torch.no_grad()
def warm_block8_ritz(matrix, *, iterations, initial_v=None, seed=4321):
    """Return paired approximate U, s, Vh using only an at-most-8x8 SVD.

    Ritz values need not identify the true leading spectrum. In particular,
    a warm subspace can miss a direction; no accuracy guarantee is asserted.
    The local generator leaves the model/data RNG streams unchanged.
    """
    if matrix.ndim != 2 or min(matrix.shape) < 2 or iterations < 1:
        raise ValueError("expected a nontrivial matrix and positive iteration count")
    x = matrix.detach().float()
    b = min(8, *x.shape)
    if initial_v is None:
        generator = torch.Generator(device=x.device).manual_seed(seed)
        q = torch.randn(x.shape[1], b, device=x.device, dtype=torch.float32,
                        generator=generator)
    else:
        if initial_v.shape != (x.shape[1], b):
            raise ValueError("warm basis has incompatible shape")
        q = initial_v.detach().to(device=x.device, dtype=torch.float32)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_precision = torch.get_float32_matmul_precision()
    try:
        if x.is_cuda:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=x.device.type, enabled=False):
            scale = torch.linalg.vector_norm(x).clamp_min(torch.finfo(torch.float32).tiny)
            normalized = x / scale
            q, _ = torch.linalg.qr(q, mode="reduced")
            for _ in range(iterations):
                p, _ = torch.linalg.qr(normalized @ q, mode="reduced")
                q, _ = torch.linalg.qr(normalized.T @ p, mode="reduced")
            p, h = torch.linalg.qr(normalized @ q, mode="reduced")
            kwargs = {"driver": "gesvd"} if x.is_cuda else {}
            uh, s, vh = torch.linalg.svd(h, full_matrices=False, **kwargs)
            return p @ uh, s * scale, vh @ q.T
    finally:
        if x.is_cuda:
            torch.set_float32_matmul_precision(previous_precision)
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
