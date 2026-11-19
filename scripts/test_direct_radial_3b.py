"""Validate long-budget overrides without launching training or Slurm jobs."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_kernel_lmo_pilot import main, training_schedule


class DirectRadialBudgetTests(unittest.TestCase):
    def test_legacy_and_long_schedules(self):
        self.assertEqual(training_schedule('100m')['TRAIN_ITER'],'191')
        schedule=training_schedule('3b')
        self.assertEqual(schedule['TRAIN_ITER'],'5723')
        self.assertEqual(schedule['TRAIN_TOKENS'],'3000000000')
        self.assertEqual(schedule['LR_WARMUP_ITERS'],'250')
        self.assertEqual(schedule['LR_DECAY_ITERS'],'5723')
        self.assertEqual(schedule['EVAL_INTERVAL'],'250')
        self.assertEqual(int(schedule['TRAIN_ITER'])*128*4096,3000500224)
        with self.assertRaises(ValueError):training_schedule('12b')

    def test_selected_arms_and_audit_receive_radial_3b(self):
        for arm,method,ratio in ((0,'none','0.0'),(2,'pgd_half','0.5'),(3,'pgd_full','1.0')):
            with tempfile.TemporaryDirectory() as root:
                calls=[]
                def fake_train(env,folder,smoke):
                    folder.mkdir(parents=True,exist_ok=True)
                    calls.append((dict(env),smoke))
                env=dict(PROJECT_DIR=root,SLURM_ARRAY_JOB_ID='123',SLURM_ARRAY_TASK_ID=str(arm))
                argv=['run_kernel_lmo_pilot.py','--study','direct_pk','--arm',str(arm),
                      '--budget','3b','--ambient-return','radial']
                with patch.dict(os.environ,env),patch('sys.argv',argv),\
                     patch('run_kernel_lmo_pilot.run_training',side_effect=fake_train),\
                     patch('run_kernel_lmo_pilot.subprocess.run') as audit:
                    main()
                self.assertEqual([smoke for _,smoke in calls],[True,False])
                config=calls[-1][0]
                for key,value in training_schedule('3b').items():self.assertEqual(config[key],value)
                self.assertEqual(config['KERNEL_METHOD'],method)
                self.assertEqual(config['KERNEL_PGD_TRIAL_RATIO'],ratio)
                self.assertEqual(config['KERNEL_AMBIENT_RETURN'],'radial')
                self.assertEqual(config['SEED'],'2345')
                self.assertEqual(config['EVAL_ITERS'],'20')
                self.assertEqual(config['SAVE_CHECKPOINT'],'0')
                self.assertEqual(config['LOAD_CHECKPOINT'],'')
                self.assertEqual(config['KERNEL_SNAPSHOT_STEPS'],'2,250,1000,3000,5000,5720')
                command=audit.call_args.args[0]
                self.assertEqual(command[command.index('--ambient-return')+1],'radial')
                self.assertTrue((Path(root)/f'results/123_{arm}/COMPLETED').exists())


if __name__=='__main__':unittest.main()
