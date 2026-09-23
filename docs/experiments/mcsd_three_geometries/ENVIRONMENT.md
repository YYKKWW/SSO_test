# Environment and loading the project

## Existing H20 installation

Reuse the working environment. Do not reinstall shared dependencies while
the historical jobs use it:

```bash
module load python/3.12.1
module load cuda/12.4
source "$HOME/envs/sso_h20/bin/activate"
cd "$HOME/projects/SSO_test-three-geometries"
export PYTHONPATH="$PWD/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
python scripts/manifold/launch.py --geometry stiefel --method manifold_mcsd_tp --stage main
```

The last command previews the full 3B configuration; add `--submit` only to
submit that one run. The Slurm wrapper performs the same environment loading.

Observed versions in the environment that passed the 2026-09-24 H20 checks:

| Component | Version |
| --- | --- |
| Python module | 3.12.1 |
| CUDA module | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Triton | 3.2.0 |
| NumPy | 2.4.4 |
| Transformers | 5.12.1 |
| Tokenizers | 0.22.2 |
| Datasets | 5.0.0 |
| Pybind11 | 3.0.4 |
| Packaging | 26.2 |
| Psutil | 7.2.2 |
| SentencePiece | 0.2.1 |

Apex and Transformer Engine are absent: these runs use Megatron's local
transformer implementation and Torch AdamW fallback. TensorBoard is absent
too; Slurm text logs are the verified source of training/validation metrics.
This is an observed environment record, not a claim that a fresh installation
using just this table has been tested. Do not silently upgrade these packages
or install another `megatron-core` over the vendored source.

## Another checkout or cluster

```bash
git clone --branch exp/mcsd-three-geometries --single-branch https://github.com/YYKKWW/SSO_test.git
cd SSO_test
export PYTHONPATH="$PWD/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
python scripts/manifold/launch.py \
  --geometry spectral --method manifold_muonsphere --stage main \
  --train-prefix /path/to/train_text_document \
  --valid-prefix /path/to/valid_text_document \
  --tokenizer /path/to/OLMo-2-tokenizer \
  --env-dir /path/to/venv
```

Prefixes refer to matching `.bin` and `.idx` files, without their suffixes.
Download/preprocess data separately; no training data, model weights, or
virtual environment is included in Git. Adapt the site's partition/QoS and
Slurm module-loading commands before submitting on a different cluster.

Both old and new optimizers live under `Megatron-LM/emerging_optimizers`.
The new entry points are registered in Megatron and invoked through
`pretrain_gpt.py`; there is no second training framework. A separate clone
of upstream Megatron is not needed on each launch. Record the selected branch
commit and preserve licenses. Do not copy mutable symlinks to the old project.
