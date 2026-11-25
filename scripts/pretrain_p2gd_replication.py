import os
from pathlib import Path
import runpy

import torch

from checkpoint_compat import install as install_checkpoint_compat
from p2gd_replication import ReplicationTrajectory

torch.set_num_threads(1)
torch.set_float32_matmul_precision('highest')
install_checkpoint_compat()
owner=ReplicationTrajectory(os.environ['ELR_OUTPUT'],os.environ['P2GD_POLICY'],
    int(os.environ['ELR_FIRST']),int(os.environ['ELR_LAST']))
owner.install()
owner.install_training_hooks()
runpy.run_path(str(Path(os.environ['MEGATRON_PATH'])/'pretrain_gpt.py'),run_name='__main__')
