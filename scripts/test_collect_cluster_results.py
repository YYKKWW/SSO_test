"""Keep intermediate and final validation formats in the published curves."""

import json
import tempfile
import unittest
from pathlib import Path

from collect_cluster_results import collect


class ResultCollectorTests(unittest.TestCase):
    def test_intermediate_and_final_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.log").write_text(
                " validation loss at iteration 250 | lm loss value: 4.711915E+00 |\n"
                " iteration 1900/ 1908 | cumulative pgd branches: 10894/478800 | "
                "cumulative pgd bursts/repeat rate/mean length: 3163/0.710/3.44 |\n"
                " validation loss at iteration 1908 on validation set | lm loss value: 3.572540E+00 |\n")
            (root / "launch_config.json").write_text(json.dumps({"TRAIN_ITER": "1908"}))
            (root / "source.sha256").write_text("source fixture\n")
            (root / "training_wall_seconds.txt").write_text("21324\n")
            (root / "TRAINING_COMPLETED").touch()
            (root / "COMPLETED").touch()
            row = collect(root)
        self.assertEqual(row["vals"], [dict(step=250, loss=4.711915), dict(step=1908, loss=3.572540)])
        self.assertEqual(row["last_regular_statistics"]["step"], 1900)
        self.assertEqual(row["final"], 3.572540)
        self.assertAlmostEqual(row["last_regular_statistics"]["repeat_rate"], 1-3163/10894)


if __name__ == "__main__":
    unittest.main()
