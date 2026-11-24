"""Six counterfactuals after an unchanged sixteen-step common prefix."""
from p2gd_budget_replay import BudgetReplay
from p2gd_six_geometry import ARMS, candidates


class SixReplay(BudgetReplay):
    candidate_factory = staticmethod(candidates)

    def __init__(self, output, first=253, last=300):
        super().__init__(output, first, last)
        self.methods = tuple(a['name'] for a in ARMS)
        self.measure_first = 269
