# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from emerging_optimizers.orthogonalized_optimizers.muon_ball import *
from emerging_optimizers.orthogonalized_optimizers.orthogonalized_optimizer import *
from emerging_optimizers.orthogonalized_optimizers.spel import *
from emerging_optimizers.orthogonalized_optimizers.spel_pgd_same_projection import *
from emerging_optimizers.orthogonalized_optimizers.spectral_ball import *
from emerging_optimizers.orthogonalized_optimizers.spectral_ball_utils import *


def get_muon_scale_factor(size_out: int, size_in: int, mode: str = "spectral") -> float:
    """Lightweight Muon scale helper exported without importing Triton-backed Muon."""
    if mode == "shape_scaling":
        return max(1, size_out / size_in) ** 0.5
    if mode == "align_adamw_rms":
        return 0.2 * max(size_out, size_in) ** 0.5
    if mode == "spectral_mup":
        return (size_out / size_in) ** 0.5
    raise ValueError(f"Invalid mode for Muon update scale factor: {mode}")


# Muon and spectral_clipping_utils import Triton-backed utilities at module
# import time. Some cluster Python environments do not provide Python.h for
# Triton driver compilation; keep SpEL/SpectralBall importable by not exporting
# those modules from the package root.
