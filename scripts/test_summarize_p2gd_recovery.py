import copy
import json
from pathlib import Path
import tempfile
import unittest

from summarize_p2gd_recovery import snapshot_summary, describe
from summarize_p2gd_trajectories import summarize


class RecoverySummaryTests(unittest.TestCase):
    def fixture(self,root):
        policies=('half_d0','top1','cluster_cone','half_output_of_cluster')
        tokens=[str(i) for i in range(20)]
        published=[]
        for name in policies:
            folder=root/('branch_'+name)/'metrics';folder.mkdir(parents=True)
            curve=[dict(iteration=i,loss=loss,batches=[dict(tokens=t,loss=loss,valid_tokens=4) for t in tokens])
                for i,loss in ((768,1.),(770,.8))]
            data=dict(trajectory_validation=curve,
                trajectory_initial=[dict(model='same',optimizer='same')],
                evaluation_data=[dict(heldout_tokens=tokens,calibration_tokens='separate')],
                training_stream=[dict(iteration=i,batches=list(range(32))) for i in (769,770)],
                trajectory_update=[dict(iteration=i,components=252,exact_commit=True,max_radius_error=0.,near_components=1) for i in (769,770)],
                trajectory_geometry=[dict(iteration=i,component=0,cluster_dimension=2,relative_error=0.) for i in (769,770)])
            for key,rows in data.items():
                (folder/(key+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            published.append(dict(policy=name,final_loss=.8,last_four_mean_loss=.9,
                unmatched_components=0,max_budget_error=0,max_radius_error=0,
                wall_seconds={'seconds':10},mean_step_seconds=5,first32_mean_step_seconds=5))
        receipt=dict(paired_data=True,paired_initial_state=True,first=769,last=770,results=published)
        (root/'REPLICATION_COMPLETED.json').write_text(json.dumps(receipt))
        return policies

    def test_four_arm_window_is_not_hardcoded_to_earlier_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);policies=self.fixture(root)
            result=summarize(root,first=769,last=770,policies=policies,receipt='REPLICATION_COMPLETED.json')
            self.assertEqual(result['updates_per_arm'],2)
            self.assertEqual(len(result['results']),4)
            self.assertEqual(result['results'][0]['near_rate'],1/252)
            self.assertIn('769--770',result['results'][0]['geometry']['streak_scope'])

    def test_invalid_pairing_and_window_rejected(self):
        for case in ('initial','stream','weighted','window','duplicate'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);policies=self.fixture(root);folder=root/'branch_top1/metrics'
                if case=='window':
                    path=root/'REPLICATION_COMPLETED.json';r=json.loads(path.read_text());r['last']=768
                elif case=='duplicate':
                    path=root/'REPLICATION_COMPLETED.json';r=json.loads(path.read_text());r['results'].append(r['results'][0])
                else:
                    name={'initial':'trajectory_initial','stream':'training_stream','weighted':'trajectory_validation'}[case]
                    path=folder/(name+'.jsonl');rows=[json.loads(s) for s in path.read_text().splitlines()]
                    if case=='initial': rows[0]['model']='different'
                    elif case=='stream': rows[-1]['batches'].pop()
                    else: rows[-1]['batches'][0]['loss']=.5
                    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                if case in ('window','duplicate'): path.write_text(json.dumps(r))
                with self.assertRaises(ValueError):
                    summarize(root,first=769,last=770,policies=policies,receipt='REPLICATION_COMPLETED.json')

    def snapshot_fixture(self):
        cap=lambda name,k:dict(name=name,capacity=k,used=min(k,5),omitted=max(5-k,0),
            point_relative_error=1e-5 if k==4 else 0.,radius_error=0.)
        row=dict(snapshot='one',q=5,category='large_cluster',svdvals_seconds=.01,
            capacities=[cap('top4',4),cap('top6',6),cap('full',5)],
            approximate_returns=[dict(block=b,iterations=k,seconds=.002,radius_error=1e-6,
                gap_error=2e-6,point_relative_error=1e-6) for b in (8,12) for k in (10,20)])
        return dict(snapshots=1,results=[row],initialization='oracle warm',scope='geometry only',timing='single call')

    def test_snapshot_capacity_and_timing_are_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'audit.json';p.write_text(json.dumps(self.snapshot_fixture()))
            r=snapshot_summary(p)
            self.assertEqual(r['capacities']['top4']['omitted_snapshots'],1)
            self.assertEqual(r['capacities']['top6']['point_relative_error']['max'],0.)
            self.assertEqual(r['approximations'][0]['sum_svd_time_over_sum_approx_time'],5.)
            self.assertIn('no independent',r['limitations'])

    def test_snapshot_missing_duplicate_or_incorrect_full_return_rejected(self):
        for case in ('duplicate','missing','full'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as directory:
                d=self.snapshot_fixture()
                if case=='duplicate': d['results'].append(copy.deepcopy(d['results'][0]));d['snapshots']=2
                elif case=='missing': d['results'][0]['approximate_returns'].pop()
                else: d['results'][0]['capacities'][-1]['point_relative_error']=.01
                p=Path(directory)/'audit.json';p.write_text(json.dumps(d))
                with self.assertRaises(ValueError): snapshot_summary(p)

    def test_nonfinite_diagnostics_rejected(self):
        with self.assertRaises(ValueError): describe([float('nan')])


if __name__=='__main__':
    unittest.main()
