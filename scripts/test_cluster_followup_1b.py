"""Follow-ups preserve the six-arm 1B protocol, including spooled launchers."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/cluster_mechanism_scales_1b_20261102.sbatch"


def describe(launcher, index, **overrides):
    env = {**os.environ, "PROJECT_DIR": str(ROOT), "SLURM_ARRAY_TASK_ID": str(index)}
    env.pop("CLUSTER_ARM_INDEX", None)
    env.pop("CLUSTER_EXPERIMENT_SEED", None)
    env.update(overrides)
    return subprocess.check_output(
        ["bash", str(launcher), "--describe"], env=env, text=True
    ).strip().split("\t")


class FollowupLaunchTests(unittest.TestCase):
    def test_four_arms_match_existing_references(self):
        shared = ROOT / "slurm/cluster_cone_lmo_1b_20261031.sbatch"
        wide, lmo = describe(shared, 3), describe(shared, 5)
        for i, projection, coefficient in (
            (0, "ambient", "8.0"), (1, "tangent1", "8.0"),
            (2, "cone_lmo", "0.5"), (3, "cone_lmo", "2.0"),
        ):
            row = describe(LAUNCHER, i)
            expected = (wide if i < 2 else lmo).copy()
            expected[0], expected[3], expected[5] = str(i + 6), projection, coefficient
            self.assertEqual(row, expected)
            self.assertEqual(row[6:], ["2345", "1908", "20"])

    def test_spooled_wrapper_and_ambient_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            spool = Path(directory) / "slurm_script"
            shutil.copyfile(LAUNCHER, spool)
            for i in range(4):
                self.assertEqual(
                    describe(spool, i, CLUSTER_ARM_INDEX="0", CLUSTER_EXPERIMENT_SEED="999"),
                    describe(LAUNCHER, i),
                )

    def test_invalid_index_fails(self):
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--describe"],
            env={**os.environ, "PROJECT_DIR": str(ROOT), "SLURM_ARRAY_TASK_ID": "4"},
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unknown follow-up index", result.stderr)


if __name__ == "__main__":
    unittest.main()
