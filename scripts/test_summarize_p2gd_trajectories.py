import unittest

from summarize_p2gd_trajectories import geometry_statistics


class GeometrySummaryTests(unittest.TestCase):
    def row(self, step, component, q=2):
        return dict(iteration=step, component=component, cluster_dimension=q,
                    relative_error=0., correction_rank=None)

    def test_streaks_are_component_specific_and_include_last_run(self):
        rows = [self.row(513,0), self.row(514,0), self.row(516,0),
                self.row(514,1), self.row(515,1), self.row(516,1)]
        result = geometry_statistics(list(reversed(rows)))
        self.assertEqual(result['near_components'],6)
        self.assertEqual(result['repeated_near_events'],3)
        self.assertEqual(result['max_observed_streak'],3)
        self.assertEqual(result['mean_observed_streak'],2)

    def test_capacity_and_budget_disclosures(self):
        rows = [self.row(513,0,6), self.row(513,1,2)]
        rows[0].update(relative_error=.02, correction_rank=3,
                       reference_cluster_match={'matching_satisfied':False})
        result = geometry_statistics(rows)
        self.assertEqual(result['exceeds_top4'],1)
        self.assertEqual(result['over_one_percent_budget'],1)
        self.assertEqual(result['reference_cluster_unmatched'],1)
        self.assertEqual(result['correction_ranks'],{3:1})

    def test_duplicate_rows_rejected(self):
        row = self.row(513,0)
        with self.assertRaises(ValueError):
            geometry_statistics([row,row])

    def test_empty_geometry(self):
        result = geometry_statistics([])
        self.assertEqual(result['near_components'],0)
        self.assertEqual(result['repeat_fraction'],0)

    def test_later_window_label(self):
        result=geometry_statistics([self.row(769,0),self.row(770,0)],769,1024)
        self.assertEqual(result['streak_scope'],
            'Truncated to steps 769--1024; not complete lifetime residence times')


if __name__=='__main__':
    unittest.main()
