import os
from pathlib import Path
import runpy

import torch

from checkpoint_compat import install as install_checkpoint_compat
from p2gd_budget_replay import BudgetReplay

torch.set_num_threads(1)
torch.set_float32_matmul_precision('highest')
install_checkpoint_compat()
owner = BudgetReplay(os.environ['ELR_OUTPUT'],253,268)
owner.install()
owner.install_training_hooks()
runpy.run_path(str(Path(os.environ['MEGATRON_PATH'])/'pretrain_gpt.py'),run_name='__main__')
