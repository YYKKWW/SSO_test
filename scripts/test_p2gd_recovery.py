import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from prepare_p2gd_recovery import prepare, sha, verified_checkpoint
from run_p2gd_replication import recover, restore_gate
from submit_p2gd_replication import main as submit


class RecoveryTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'original';results=source/'results';results.mkdir(parents=True)
        (results/'common_prefix/metrics').mkdir(parents=True)
        (source/'common_checkpoint').mkdir()
        (source/'common_checkpoint/latest_checkpointed_iteration.txt').write_text('768')
        (source/'common_checkpoint/state.bin').write_bytes(b'saved-state')
        (source/'code.py').write_text('original = True\n')
        (source/'source.sha256').write_text(sha(source/'code.py')+'  code.py\n')
        receipt=dict(prefix_job='1',branch_job='2',audit_job='3')
        (results/'replication_submission.json').write_text(json.dumps(receipt))
        (results/'replay_preparation.json').write_text(json.dumps(dict(base='base',target=str(source))))
        status=dict(frozen_source_verified=True,checkpoint_iteration='768',
            source_manifest_sha256=sha(source/'source.sha256'),
            jobs={'1':{'state':'FAILED'},'2':{'state':'CANCELLED'},'3':{'state':'CANCELLED'}},
            checkpoint_files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p))
                for p in (source/'common_checkpoint').iterdir()})
        status_path=root/'status.json';status_path.write_text(json.dumps(status))
        file=root/'patch.py';file.write_text('fixed = True\n')
        overlay=root/'overlay.tar.gz'
        with tarfile.open(overlay,'w:gz') as archive:
            archive.add(file,arcname='studies/p2gd_elr/recovery.py')
        return source,status,status_path,overlay

    def test_frozen_source_and_checkpoint_reused_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,status,status_path,overlay=self.fixture(root)
            target=root/'recovery'
            with patch('prepare_p2gd_recovery.shutil.disk_usage') as usage:
                usage.return_value.free=10*1024**3
                record=prepare(source,target,overlay,status_path)
            self.assertEqual((target/'common_checkpoint').resolve(),(source/'common_checkpoint').resolve())
            self.assertEqual((target/'results/common_prefix').resolve(),(source/'results/common_prefix').resolve())
            self.assertEqual(record['repeated_prefix_updates'],0)
            self.assertEqual(sha(source/'source.sha256'),status['source_manifest_sha256'])
            self.assertFalse((source/'studies').exists())
            with self.assertRaises(RuntimeError): prepare(source,target,overlay,status_path)

    def test_reject_changed_checkpoint_or_advanced_source(self):
        for case in ('content','extra','branch','restore','iteration'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as folder:
                source,status,_,_=self.fixture(Path(folder))
                if case=='content': (source/'common_checkpoint/state.bin').write_bytes(b'changed')
                elif case=='extra': (source/'common_checkpoint/extra').touch()
                elif case=='branch': (source/'results/branch_top1').mkdir()
                elif case=='restore': (source/'results/restore_smoke').mkdir()
                else: (source/'common_checkpoint/latest_checkpointed_iteration.txt').write_text('512')
                with self.assertRaises(RuntimeError): verified_checkpoint(source,status)

    def test_recover_calls_only_restore_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'old';source.mkdir();(root/'results').mkdir()
            (root/'common_checkpoint').symlink_to(source/'common_checkpoint')
            (root/'results/common_prefix').symlink_to(source/'results/common_prefix')
            (root/'results/recovery_preparation.json').write_text(json.dumps(dict(source=str(source))))
            with patch.dict(os.environ,P2GD_PREVIOUS_PATHS_ROOT=str(root/'prior')), \
                 patch('run_p2gd_replication.validate'), \
                 patch('run_p2gd_replication.verify_repeated_prefix',return_value={'proof':True}), \
                 patch('run_p2gd_replication.restore_gate') as gate, \
                 patch('run_p2gd_replication.run') as training:
                recover(root)
            gate.assert_called_once();training.assert_not_called()

    def test_restore_failure_never_releases_branches(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'results').mkdir()
            with patch('run_p2gd_replication.validate',return_value=root/'checkpoint'), \
                 patch('run_p2gd_replication.settings',return_value={}), \
                 patch('run_p2gd_replication.run'), \
                 patch('run_p2gd_replication.verify_branch',side_effect=RuntimeError('mismatch')):
                with self.assertRaises(RuntimeError): restore_gate(root,{},root/'old')
            self.assertFalse((root/'results/PREFIX_READY.json').exists())

    def test_submission_uses_recovery_and_two_gpu_dependent_array(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'results').mkdir();prior=root/'prior';base=root/'base';old=root/'reference'
            for directory in (prior/'results',base/'results',old/'results/prefix'):
                directory.mkdir(parents=True)
            (prior/'results/TRAJECTORIES_COMPLETED.json').write_text(json.dumps(dict(paired_data=True,paired_initial_state=True)))
            (root/'results/recovery_preparation.json').write_text(json.dumps(dict(source='original')))
            (root/'results/replay_preparation.json').write_text(json.dumps(dict(base=str(base))))
            (base/'results/confirmation_submission.json').write_text(json.dumps(dict(original_reference=str(old))))
            config={k:'data' for k in ('TRAIN_DATA_PREFIX','VALID_DATA_PREFIX','TOKENIZER_MODEL')}
            (old/'results/prefix/launch_config.json').write_text(json.dumps(config))
            with patch.dict(os.environ,PROJECT_DIR=str(root),P2GD_PREVIOUS_PATHS_ROOT=str(prior)), \
                 patch('submit_p2gd_replication.shutil.disk_usage') as usage, \
                 patch('submit_p2gd_replication.subprocess.check_output',side_effect=['10','11','12']) as sbatch, \
                 patch('builtins.print'):
                usage.return_value.free=10*1024**3
                submit()
                self.assertEqual(sbatch.call_args_list[0].kwargs['env']['P2GD_STAGE'],'recover')
                self.assertIn('--time=00:45:00',sbatch.call_args_list[0].args[0])
                self.assertIn('--array=0-3%2',sbatch.call_args_list[1].args[0])
                self.assertIn('--dependency=afterok:10',sbatch.call_args_list[1].args[0])
                self.assertIn('--dependency=afterok:11',sbatch.call_args_list[2].args[0])
                submit();self.assertEqual(sbatch.call_count,3)


if __name__=='__main__':
    unittest.main()
