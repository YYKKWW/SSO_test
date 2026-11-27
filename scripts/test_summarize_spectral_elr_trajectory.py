"""Small regression checks; the CLI also audits the full published evidence."""
import unittest

from summarize_spectral_elr_trajectory import STEPS, indexed, paired_losses


class SummaryTests(unittest.TestCase):
    def test_grid_rejects_duplicates_and_missing_steps(self):
        rs = [dict(iteration=s,component_index=0) for s in STEPS]
        self.assertEqual(len(indexed(rs,[0],"test")),64)
        with self.assertRaises(ValueError): indexed(rs+rs[:1],[0],"test")
        with self.assertRaises(ValueError): indexed(rs[:-1],[0],"test")

    def test_tail_and_endpoint_are_separate(self):
        reference = [dict(loss_after=4.,after=[dict(loss=4.),dict(loss=4.)]) for _ in STEPS]
        candidate = [dict(loss_after=3.,after=[dict(loss=3.),dict(loss=3.)]) for _ in STEPS]
        candidate[-1] = dict(loss_after=5.,after=[dict(loss=5.),dict(loss=5.)])
        r = paired_losses(candidate,reference)
        self.assertEqual(r["final"],1.)
        self.assertEqual(r["last16_mean"],-.875)
        self.assertEqual(r["lower_steps"],63)
        self.assertEqual(r["lower_final_batches"],0)


if __name__ == "__main__":
    unittest.main()
