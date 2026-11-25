import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from audit_p2gd_snapshots import block_ritz
from p2gd_replication import ReplicationTrajectory, SnapshotSelector
from run_p2gd_replication import verify_repeated_prefix, verify_restored_state
from run_p2gd_trajectories import verify_branch, verify_initial
from test_p2gd_trajectory_integrity import IntegrityTests
from verify_p2gd_six import verify_prefix_state
from summarize_p2gd_replication import summarize


class ReplicationTests(unittest.TestCase):
    def prefix_fixture(self,path):
        fixture=IntegrityTests();values=fixture.fixture(path)
        for key in ('trajectory_update','training_stream'):
            values[key]=[dict(copy.deepcopy(values[key][0]),iteration=i) for i in range(513,769)]
        values['trajectory_validation']=[dict(copy.deepcopy(values['trajectory_validation'][0]),iteration=i)
            for i in range(512,769,32)]
        values['state']=[dict(copy.deepcopy(values['state'][0]),iteration=i,parameter='w'+str(j))
            for i in [513,*range(544,769,32)] for j in range(5)]
        fixture.save(path,values)
        return values

    def test_sparse_prefix_accepts_declared_schedule_not_dense(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);values=self.prefix_fixture(p)
            proof=verify_repeated_prefix(p,p)
            self.assertEqual(proof['updates'],256)
            self.assertEqual(proof['matched_reference_state_records'],45)
            self.assertEqual(proof['audited_iterations'],[513,544,576,608,640,672,704,736,768])
            with self.assertRaises(ValueError):
                verify_prefix_state(values['state'],values['state'],513,768)

    def test_sparse_prefix_rejects_missing_or_extra_scheduled_records(self):
        for change in ('step','component','extra'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as folder:
                p=Path(folder);values=self.prefix_fixture(p)
                if change=='step':
                    values['state']=[r for r in values['state'] if r['iteration']!=544]
                elif change=='component':
                    values['state'].pop(5)
                else:
                    values['state'].append(dict(values['state'][0],iteration=514))
                IntegrityTests().save(p,values)
                with self.assertRaises((RuntimeError,ValueError)):
                    verify_repeated_prefix(p,p)

    def test_repeated_prefix_rejects_stream_curve_and_state_changes(self):
        for key in ('state','training_stream','trajectory_validation','trajectory_update'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as folder:
                a=Path(folder)/'a';b=Path(folder)/'b';a.mkdir();b.mkdir()
                values=self.prefix_fixture(a);self.prefix_fixture(b)
                if key=='state': values[key][5]['gradient_sha256']='changed'
                elif key=='training_stream': values[key][100]['batches'][0]=['changed','tokens']
                elif key=='trajectory_update': values[key][100]['near_components']+=1
                else: values[key][3]['loss']+=.01
                IntegrityTests().save(a,values)
                with self.assertRaises((RuntimeError,ValueError)):
                    verify_repeated_prefix(a,b)

    def test_summary_preserves_failed_status_and_no_new_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);current=root/'current';prior=root/'prior'
            a=current/'common_prefix/metrics';b=prior/'branch_half_d0/metrics'
            a.mkdir(parents=True);b.mkdir(parents=True)
            self.prefix_fixture(a);self.prefix_fixture(b)
            (a/'checkpoint_post_state.jsonl').write_text(json.dumps(dict(iteration=768,
                model=['hash'],optimizer=['state']))+'\n')
            (current/'replication_submission.json').write_text(json.dumps(dict(
                prefix_job='1',branch_job='2',audit_job='3')))
            status=dict(jobs={'1':{'state':'FAILED'},'2':{'state':'CANCELLED'},'3':{'state':'CANCELLED'}},
                restore_smoke_present=False,prefix_ready=False)
            result=summarize(current,prior,status)
            self.assertEqual(result['scheduler']['jobs']['1']['state'],'FAILED')
            self.assertEqual(result['new_branch_results'],0)
            self.assertFalse(result['restore_smoke_completed'])
            self.assertFalse(result['prefix_ready'])
            (current/'restore_smoke').mkdir()
            with self.assertRaises(ValueError): summarize(current,prior,status)

    def test_snapshot_quotas_and_coverage(self):
        selector=SnapshotSelector(769)
        chosen=[]
        for step in range(769,1025):
            for component in range(8):
                item=selector.choose(step,component,6 if component<3 else 2,1024)
                if item: chosen.append(item)
        self.assertLessEqual(len(chosen),24)
        self.assertEqual({r['window'] for r in chosen},{0,1,2,3})
        self.assertTrue(all(n<=2 for n in selector.counts.values()))
        self.assertIn('large_cluster',{r['category'] for r in chosen})
        self.assertIn('repeat',{r['category'] for r in chosen})

    def test_snapshot_byte_bound_and_gap(self):
        selector=SnapshotSelector(769,budget=8)
        first=selector.choose(769,0,2,4)
        self.assertEqual(first['category'],'entry')
        later=selector.choose(771,1,2,4)
        self.assertFalse(later['repeated'])
        self.assertIsNone(selector.choose(772,2,6,1))

    def test_exact_post_save_restore(self):
        post=[dict(iteration=768,model=[['w','abc']],optimizer=[{'step':768}])]
        initial=copy.deepcopy(post);initial[0]['iteration']=769
        verify_restored_state(post,initial)
        initial[0]['optimizer'][0]['step']=767
        with self.assertRaises(RuntimeError): verify_restored_state(post,initial)
        with self.assertRaises(RuntimeError): verify_restored_state([],initial)

    def test_warm_block_rectangular_oracle(self):
        device=os.environ.get('ELR_TEST_DEVICE','cpu')
        torch.manual_seed(20261125);torch.set_num_threads(1)
        a=torch.randn(24,12,device=device,dtype=torch.float32)
        u,s,vh=torch.linalg.svd(a,full_matrices=False)
        uh,sh,vh2=block_ritz(a,vh[:8].T,10)
        torch.testing.assert_close(sh[:2],s[:2],rtol=2e-5,atol=2e-5)
        self.assertEqual(uh.shape,(24,8))
        self.assertEqual(vh2.shape,(12,8))

    def test_later_window_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);fixture=IntegrityTests();values=fixture.fixture(p)
            for rows in values.values():
                for row in rows:
                    if 'iteration' in row: row['iteration']+=256
            fixture.save(p,values)
            self.assertEqual(verify_branch(p,last=770,first=769)['updates'],2)
            verify_initial(p,p,first=769)

    def test_snapshot_does_not_change_tensors_or_rng(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{'P2GD_SNAPSHOTS':'1'}):
            owner=ReplicationTrajectory.__new__(ReplicationTrajectory)
            owner.policy='cluster_cone';owner.iteration=769;owner.near_threshold=2e-4
            owner.snapshot_root=Path(folder);owner.selector=SnapshotSelector(769)
            records=[];owner.emit=lambda name,row: records.append(row)
            x=torch.diag(torch.tensor([1.,.9999,.5,.2]));m=torch.randn_like(x)
            d0=m.clone();u=torch.eye(4);s=x.diag();h=torch.randn(8,4)
            context={'index':0,'name':'layer','calibration':h}
            state=torch.random.get_rng_state();original=x.clone()
            owner.observe_near_component(context,x,m,d0,u,s,u,1.,.01,x,x,{'effective_alpha':.01})
            torch.testing.assert_close(x,original,atol=0,rtol=0)
            self.assertTrue(torch.equal(torch.random.get_rng_state(),state))
            snapshot=torch.load(Path(folder)/records[0]['path'],weights_only=True)
            self.assertEqual(snapshot['q'],2)
            self.assertEqual(snapshot['category'],'entry')


if __name__=='__main__':
    unittest.main()
