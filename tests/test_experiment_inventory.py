"""Check archived coverage and provenance without querying or changing H20."""

import importlib.util
import json
import re
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_experiment_inventory", ROOT / "scripts/manifold/build_experiment_inventory.py"
)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class ExperimentInventoryTests(unittest.TestCase):
    def setUp(self):
        self.current, self.supplemental, self.runs = inventory.load_inventory()

    def test_no_historical_or_current_jobs_dropped(self):
        self.assertEqual(len(self.runs), 80)
        self.assertEqual(len({r["job_id"] for r in self.runs}), 80)
        self.assertEqual(Counter(r["protocol"] for r in self.runs), {
            "w256_shape_r1": 64, "w384_init_radius_100m": 10, "w384_init_radius_3b": 6,
        })
        indexed = {r["job_id"]: r for r in self.runs}
        for original in self.current["runs"]:
            for key, value in original.items():
                self.assertEqual(indexed[original["job_id"]][key], value)

    def test_all_current_runs_have_coverage_cells(self):
        groups = {(r["tokens"], r["seed"], r["direction"]) for r in self.current["runs"]}
        rendered = "\n".join(inventory.coverage(self.current["runs"], *group) for group in groups)
        for run in self.current["runs"]:
            self.assertEqual(rendered.count(f"#job-{run['job_id']})"), 1)
        self.assertIn("0.015", rendered)
        self.assertIn("0.020", rendered)

    def test_historical_transcription_matches_original_tables(self):
        for batch in self.supplemental["completed_batches"]:
            text = (inventory.DOCS / batch["source"]).read_text()
            for run in batch["runs"]:
                self.assertIn(run["job_id"], text)
                matching_rows = [line for line in text.splitlines() if run["elapsed"] in line]
                self.assertTrue(any(f"{run['final_val']:.6f}" in row for row in matching_rows))

    def test_cancelled_jobs_do_not_become_final_losses(self):
        ids = {r["job_id"] for r in self.runs}
        for run in self.supplemental["non_final_records"]:
            self.assertTrue((inventory.DOCS / run["source"]).is_file())
            if run["kind"] == "Cancelled by user":
                self.assertNotIn(run["jobs"], ids)

    def test_elapsed_time_display_preserves_days(self):
        self.assertEqual(inventory.elapsed_hours("1-12:52:20"), "36:52:20")
        self.assertEqual(inventory.elapsed_hours("25:28:44"), "25:28:44")

    def test_generated_files_current_and_local_links_resolve(self):
        outputs = inventory.build()
        for path, expected in outputs.items():
            self.assertEqual(path.read_text(), expected, str(path))
            if path.suffix != ".md":
                json.loads(expected)
                continue
            for target in re.findall(r"\]\(([^)]+)\)", expected):
                if "://" in target:
                    continue
                filename, _, anchor = target.partition("#")
                linked = (path.parent / filename).resolve() if filename else path
                self.assertTrue(linked.is_file(), f"Broken link in {path}: {target}")
                if anchor.startswith("job-"):
                    self.assertIn(f'id="{anchor}"', linked.read_text())

    def test_toy_grid_preserves_all_settings(self):
        for filename in inventory.TOYS:
            toy = inventory.read_json(inventory.ARCHIVE / filename)
            self.assertEqual(len(toy["summary"]), 18)
            self.assertEqual(toy["config"]["seeds"], 12)
            self.assertEqual(toy["config"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
