# Runtime source and environment

The training entry remains `Megatron-LM/pretrain_gpt.py`. Algorithm code is
under `Megatron-LM/emerging_optimizers/orthogonalized_optimizers/`; registration,
CLI options, initialization metadata and parameter ownership are small Megatron
integration changes. `configs/manifold`, `scripts/manifold` and the Slurm script
are the outer experiment layer, analogous to upstream's `megatron_scripts`.

Upstream references:

- https://github.com/Unakar/Megatron-LM/tree/SSO_main/emerging_optimizers
- https://github.com/Unakar/Spectral-Sphere-Optimizer

The old broad `models/` ignore rule accidentally excluded framework source,
not just model weights. On 2026-09-24 the Python-only `megatron/core/models`,
`megatron/training/models` and tokenizer `models` snapshots were copied
from the existing H20 exploration runtime into this
isolated branch. Original SPDX/license headers are retained. No old H20
file was changed, and no checkpoints or tokenizer assets were copied to Git.
The precise upstream commit of this older source is not known, so it is
recorded as a vendored snapshot, not falsely attributed to today's SSO_main.
The current upstream branch is not silently substituted for this runtime.

Snapshot check: `megatron/core/models/gpt/gpt_model.py` SHA256 is
`fcb8f5a0e6661f20d2c31bec02037784a025b980cacad7424234e5f6d7f51eb2`.
The Git commit and per-file hashes in each resolved run config identify the
complete effective source. A mutable symlink to the old project is not used
for submitted runs. Preserve the NVIDIA and other third-party licenses.

H20 uses its existing `sso_h20` environment (PyTorch 2.6.0+cu124, CUDA module
12.4, Python module 3.12.1). The Slurm wrapper activates it explicitly; run.py
records actual versions and loaded modules. This is the tested-environment
target, not a claim that local CPU unit tests establish CUDA compatibility.
Do not replace the shared environment while historical jobs are running.
