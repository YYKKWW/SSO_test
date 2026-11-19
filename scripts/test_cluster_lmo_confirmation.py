"""The confirmation batch changes seed, not the selected candidate mechanisms."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/cluster_lmo_confirmation_1b_20261110.sbatch"
SHARED = ROOT / "slurm/cluster_cone_lmo_1b_20261031.sbatch"
ARMS = (2, 8, 11, 0, 1)


def describe(path, index, **overrides):
    env = dict(os.environ, PROJECT_DIR=str(ROOT), SLURM_ARRAY_TASK_ID=str(index))
    env.pop("CLUSTER_ARM_INDEX", None)
    env.pop("CLUSTER_EXPERIMENT_SEED", None)
    env.update(overrides)
    return subprocess.check_output(
        ["bash", str(path), "--describe"], env=env, text=True
    ).strip().split("\t")


class ConfirmationLaunchTests(unittest.TestCase):
    def test_five_arms_use_original_definitions(self):
        for index, original in enumerate(ARMS):
            expected = describe(SHARED, original, CLUSTER_EXPERIMENT_SEED="1234")
            self.assertEqual(describe(LAUNCHER, index), expected)
            self.assertEqual(expected[6:], ["1234", "1908", "20"])
        self.assertEqual(describe(LAUNCHER, 0)[2], "control")
        self.assertEqual(describe(LAUNCHER, 1)[3:6], ["cone_lmo", "2e-4", "0.5"])
        self.assertEqual(describe(LAUNCHER, 2)[3:6], ["cone_lmo_mix25", "2e-4", "0.5"])
        self.assertEqual(describe(LAUNCHER, 3)[1], "muon_ball_dist")
        self.assertEqual(describe(LAUNCHER, 4)[1], "spectral_ball_dist")

    def test_spooled_wrapper_overrides_inherited_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = Path(directory) / "slurm_script"
            shutil.copyfile(LAUNCHER, spool)
            for index in range(len(ARMS)):
                self.assertEqual(
                    describe(spool, index, CLUSTER_ARM_INDEX="9", CLUSTER_EXPERIMENT_SEED="999"),
                    describe(LAUNCHER, index),
                )

    def test_invalid_index_fails_before_training(self):
        for index in ("-1", "5", "bad"):
            result = subprocess.run(
                ["bash", str(LAUNCHER), "--describe"], capture_output=True, text=True,
                env=dict(os.environ, PROJECT_DIR=str(ROOT), SLURM_ARRAY_TASK_ID=index),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Unknown confirmation index", result.stderr)

    def test_resource_limit_and_independent_smoke(self):
        text = LAUNCHER.read_text()
        self.assertIn("#SBATCH --array=0-4%2", text)
        self.assertIn("#SBATCH --gres=gpu:nvidia_h20:1", text)
        self.assertIn('CLUSTER_SMOKE=1 CLUSTER_RUN_SUFFIX=_smoke bash "$launcher"', text)
        self.assertIn('export CLUSTER_SMOKE=0 CLUSTER_RUN_SUFFIX=""', text)
        self.assertIn('export SAVE_CHECKPOINT=0', SHARED.read_text())


if __name__ == "__main__":
    unittest.main()
