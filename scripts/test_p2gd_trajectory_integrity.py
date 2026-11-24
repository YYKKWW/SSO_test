import copy
import json
from pathlib import Path
import tempfile
import unittest

from run_p2gd_trajectories import verify_branch, verify_initial


class IntegrityTests(unittest.TestCase):
    def fixture(self,path):
        batches=[dict(tokens=str(i),loss=4.,valid_tokens=10.) for i in range(20)]
        values={
            'trajectory_update':[dict(iteration=i,components=252,exact_commit=True,
                max_radius_error=1e-7,near_components=1,unmatched_components=0,max_budget_error=0.) for i in (513,514)],
            'training_stream':[dict(iteration=i,batches=[['a','b']]*32,seconds=1.) for i in (513,514)],
            'trajectory_validation':[dict(iteration=i,policy='half_d0',loss=4.,batches=batches) for i in (512,514)],
            'evaluation_data':[dict(calibration_tokens='cal',heldout_tokens=[str(i) for i in range(20)])],
            'trajectory_initial':[dict(model=['weight'],optimizer=['momentum'],iteration=513)],
            'state':[dict(iteration=513,weight_sha256='a',gradient_sha256='b',nesterov_sha256='c')],
        }
        self.save(path,values)
        return values

    def save(self,path,values):
        for k,v in values.items():
            (path/(k+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in v))

    def test_complete_fixture_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);self.fixture(p)
            self.assertEqual(verify_branch(p,514)['updates'],2)

    def test_missing_update_or_validation_fails(self):
        for key in ('trajectory_update','trajectory_validation'):
            with tempfile.TemporaryDirectory() as folder:
                p=Path(folder);values=self.fixture(p)
                values[key].pop();self.save(p,values)
                with self.assertRaises(RuntimeError):
                    verify_branch(p,514)

    def test_pairing_rejects_optimizer_and_data_change(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a';b=Path(folder)/'b';a.mkdir();b.mkdir()
            values=self.fixture(a);self.fixture(b)
            verify_initial(a,b)
            for key in ('trajectory_initial','training_stream'):
                changed=copy.deepcopy(values)
                if key=='trajectory_initial':
                    changed[key][0]['optimizer']=['changed']
                else:
                    changed[key][0]['batches'][0]=['different','tokens']
                self.save(a,changed)
                with self.assertRaises(RuntimeError):
                    verify_initial(a,b)
                self.save(a,values)


if __name__=='__main__':
    unittest.main()
