"""Run against the frozen ELR host; no Megatron initialization is required."""
import os
import unittest
from unittest.mock import patch

import torch

from adapter import Intervention
from confirm_adapter import ConfirmationIntervention
from p2gd_elr_geometry import METHODS, aus, match_aus, repair
from p2gd_elr_replay import RepairReplay


class ReplayTests(unittest.TestCase):
    def test_fp32_geometry_on_requested_device(self):
        device = os.environ.get('ELR_TEST_DEVICE', 'cpu')
        torch.manual_seed(139)
        x = torch.diag(torch.tensor([1., .9999, .6, .2], device=device))
        m = torch.randn_like(x)
        d0 = torch.randn_like(x)
        d0[0, 0] = 0.
        u = torch.eye(4, device=device)[:, :2]
        s = torch.tensor([1., .9999], device=device)
        target = aus(x, x+.01*d0)
        for method in METHODS:
            fn = lambda a: repair(method, d0, m, u, s, u, 1., a)
            with patch('torch.linalg.svd', side_effect=AssertionError('search SVD')), \
                 patch('torch.linalg.svdvals', side_effect=AssertionError('search SVD')):
                _, y, info = match_aus(x, fn, .02, target)
            norm = torch.linalg.svdvals(y, **({'driver':'gesvd'} if y.is_cuda else {}))[0]
            returned = y/norm
            self.assertLessEqual(abs(aus(x, returned)-target), info['tolerance']+2e-6)

    def check_restore(self, failure=False, methods=METHODS):
        owner = RepairReplay.__new__(RepairReplay)
        owner.iteration = 253
        owner.last_heldout = None
        owner.methods = methods
        batch = (torch.ones(2,dtype=torch.long), torch.ones(2,dtype=torch.long), torch.ones(2))
        owner.batch = batch
        owner.fixed_batches = [batch]
        module = torch.nn.Linear(2,2,bias=False).to(torch.bfloat16)
        master = module.weight.detach().float().clone()
        candidates = {method:master.clone()+i*.1 for i,method in enumerate(methods)}
        owner.replay_records = [dict(W=master,module=module,row_start=0,row_stop=2,candidates=candidates)]
        rows = []
        owner.emit = lambda name, record: rows.append((name,record))
        original_master, original_model = master.clone(), module.weight.clone()

        def baseline(instance, model):
            instance.last_heldout = dict(iteration=253, loss_before=2., loss_after=1.,
                after=[dict(loss=1., tokens='t', labels='l', valid_tokens=2.)])
            return 1., torch.zeros(1)

        def alternate(instance, model):
            if failure:
                raise RuntimeError('injected evaluation failure')
            return float(module.weight.float().sum()), torch.zeros(1)

        with patch.object(ConfirmationIntervention, 'evaluate_probe', new=baseline), \
             patch.object(Intervention, 'evaluate_probe', new=alternate):
            if failure:
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    owner.evaluate_probe(None)
            else:
                owner.evaluate_probe(None)
                self.assertEqual(len(rows[-1][1]['results']), len(methods))
                self.assertTrue(rows[-1][1]['restore_asserted'])
        torch.testing.assert_close(master, original_master, rtol=0, atol=0)
        torch.testing.assert_close(module.weight, original_model, rtol=0, atol=0)
        self.assertIs(owner.batch, batch)

    def test_restore_after_all_counterfactuals(self):
        self.check_restore()

    def test_restore_when_evaluation_fails(self):
        self.check_restore(failure=True)


if __name__ == '__main__':
    unittest.main()
