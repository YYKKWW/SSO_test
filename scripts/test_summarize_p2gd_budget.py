"""Regression checks for relative wins versus complete-update descent."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from summarize_p2gd_budget import read, summarize


class BudgetSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = (Path(__file__).resolve().parents[1] /
                    'docs/results/p2gd_budget_20261120')
        cls.metrics = cls.root / 'h20'
        cls.result = summarize(cls.metrics)
        cls.rows = {row['name']: row for row in cls.result['results']}

    def test_relative_win_is_not_complete_update_descent(self):
        row = self.rows['spectral_100_finite_momentum']
        self.assertEqual(row['better_steps'], 7)
        self.assertEqual(row['heldout_decrease_steps'], 4)
        self.assertEqual(row['heldout_increase_steps'], 3)
        self.assertEqual(row['heldout_tied_steps'], 0)
        self.assertAlmostEqual(row['mean_heldout_change_from_pre_update'],
                               -0.0020051480020797685, places=8)

    def test_training_and_heldout_signs_remain_separate(self):
        row = self.rows['output_100_finite_momentum']
        self.assertLess(row['mean_difference_vs_matched_control'], 0)
        self.assertGreater(row['mean_training_difference_vs_matched_control'], 0)

    def test_all_steps_include_nine_zero_difference_controls(self):
        self.assertEqual(self.result['total_steps'], 16)
        self.assertEqual(self.result['informative_steps'], 7)
        for row in self.rows.values():
            self.assertAlmostEqual(row['mean_all_step_difference_vs_matched_control'],
                                   row['mean_difference_vs_matched_control'] * 7 / 16)

    def test_published_historical_fields_are_unchanged(self):
        historical = json.loads((self.metrics / 'BUDGET_COMPLETED.json').read_text())
        old_rows = {row['name']: row for row in historical['results']}
        for name, row in self.rows.items():
            for key, value in old_rows[name].items():
                self.assertEqual(row[key], value, (name, key))
        self.assertEqual(self.result['source_sha256'], historical['source_sha256'])

    def test_duplicate_and_misassigned_geometry_are_rejected(self):
        loss = read(self.metrics / 'repair_loss_replay.jsonl')
        geometry = read(self.metrics / 'repair_geometry.jsonl')
        with patch('summarize_p2gd_budget.read', side_effect=[loss + [loss[0]], geometry]):
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                summarize(self.metrics)
        misplaced = copy.deepcopy(geometry)
        empty_step = next(row['iteration'] for row in loss if not row['near_components'])
        misplaced[0]['iteration'] = empty_step
        with patch('summarize_p2gd_budget.read', side_effect=[loss, misplaced]):
            with self.assertRaisesRegex(ValueError, 'per-step'):
                summarize(self.metrics)


if __name__ == '__main__':
    unittest.main()
