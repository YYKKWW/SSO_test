import unittest
from unittest.mock import patch

import torch

from p2gd_elr_geometry import METHODS, aus, match_aus, repair
from returned_step_repair_study import State


class RepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(127)
        self.state = State(torch.diag(torch.tensor([1., .9999, .6, .2], dtype=torch.float64)),
                           torch.randn(4, 4, dtype=torch.float64), 1., .015)

    def test_same_formulas_as_local_study(self):
        st = self.state
        for method in METHODS:
            for alpha in (.001, .015, .06):
                actual = repair(method, st.d0, st.m, st.u[:, :st.q], st.s[:st.q],
                                st.v[:, :st.q], st.radius, alpha)
                torch.testing.assert_close(actual, st.direction(method, alpha))

    def test_angular_search_without_full_svd(self):
        st = self.state
        target = aus(st.x, st.x+.5*st.alpha*st.d0)
        with patch('torch.linalg.svd', side_effect=AssertionError('SVD in search')), \
             patch('torch.linalg.svdvals', side_effect=AssertionError('SVD in search')):
            for method in METHODS:
                fn = lambda alpha: repair(method, st.d0, st.m, st.u[:, :st.q],
                    st.s[:st.q], st.v[:, :st.q], 1., alpha)
                _, trial, record = match_aus(st.x, fn, st.alpha, target, rtol=1e-5, atol=0.)
                self.assertLessEqual(abs(aus(st.x, trial)-target), target*1e-5)
                self.assertEqual(record['search_full_svd_calls'], 0)
                self.assertAlmostEqual(aus(st.x, trial), aus(st.x, 3.5*trial), places=12)

    def test_finite_model_uses_final_coefficient(self):
        st = self.state
        target = aus(st.x, st.x+.5*st.alpha*st.d0)
        fn = lambda alpha: repair('finite_momentum', st.d0, st.m, st.u[:, :st.q],
            st.s[:st.q], st.v[:, :st.q], 1., alpha)
        d, _, info = match_aus(st.x, fn, st.alpha, target)
        h = st.u[:, :st.q].T@d@st.v[:, :st.q]
        model = torch.diag(st.s[:st.q])+info['effective_alpha']*(h+h.T)/2
        self.assertAlmostEqual(float(torch.linalg.eigvalsh(model)[-1]), 1., places=12)

    def test_impossible_or_zero_target_is_not_a_success(self):
        with self.assertRaises(RuntimeError):
            match_aus(self.state.x, lambda a: self.state.x, .01, .01)
        with self.assertRaises(ValueError):
            match_aus(self.state.x, lambda a: self.state.x, .01, 0.)


if __name__ == '__main__':
    unittest.main()
