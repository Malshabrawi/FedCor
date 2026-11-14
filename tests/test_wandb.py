import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment_io import RunRecorder,SystemProfile
from wandb_tracking import WandbTracker

class WandbTests(unittest.TestCase):
    def test_disabled_requires_no_sdk(self):
        with patch.dict(sys.modules,{'wandb':None}):
            tracker=WandbTracker(SimpleNamespace(wandb=False),{},Path('unused'))
            tracker.log({});tracker.finish()
            self.assertIsNone(tracker.run)
    def test_each_seed_logs_and_finishes_separately(self):
        sdk=MagicMock();runs=[MagicMock(),MagicMock()];sdk.init.side_effect=runs
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,{'wandb':sdk}):
            for seed in [1,2]:
                args=SimpleNamespace(run_root=tmp,dataset='fmnist',afl=False,power_d=False,gpr=True,
                    dynamic_C=True,system_aware=True,adaptive_local_epochs=False,shards_per_client=2,
                    alpha=None,iid=0,wandb=True,wandb_project='test',wandb_mode='offline')
                recorder=RunRecorder(args,seed,{0:list(range(10))},SystemProfile(1,seed,1))
                recorder.round(dict(round=1,test_accuracy=.5,test_loss=1.,clients=[0],actual_epochs=[1],local_steps=[2],
                    client_validation_accuracy=[.6],participation_counts=[1],threshold=.2,
                    simulated_client_seconds=2.,gp_probe_client_seconds=1.,measured_server_seconds=.5))
                recorder.finish(1,.5)
                self.assertEqual(json.loads((recorder.path/'summary.json').read_text())['status'],'complete')
                recorder.tracker._finish_incomplete()
        self.assertEqual(sdk.init.call_count,2)
        self.assertNotEqual(sdk.init.call_args_list[0].kwargs['name'],sdk.init.call_args_list[1].kwargs['name'])
        self.assertIn('full_fixed_epochs',sdk.init.call_args.kwargs['group'])
        for run in runs:
            metrics=run.log.call_args.args[0]
            self.assertEqual(metrics['selected_clients'],1)
            self.assertEqual(metrics['cumulative_total_seconds'],3.5)
            self.assertEqual(metrics['participation_jain_index'],1.)
            self.assertNotIn('clients',metrics)
            run.finish.assert_called_once_with(exit_code=0)
    def test_failure_cleanup_marks_run_failed(self):
        sdk=MagicMock()
        args=SimpleNamespace(wandb=True,wandb_project='test',wandb_mode='offline')
        with patch.dict(sys.modules,{'wandb':sdk}):
            tracker=WandbTracker(args,{'dataset':'fmnist','method':'oort','iid':1},Path('unused'))
            tracker._finish_incomplete();tracker._finish_incomplete()
        sdk.init.return_value.finish.assert_called_once_with(exit_code=1)
    def test_enabled_missing_sdk_fails_clearly(self):
        with patch.dict(sys.modules,{'wandb':None}),self.assertRaisesRegex(RuntimeError,'pip install wandb'):
            WandbTracker(SimpleNamespace(wandb=True),{},Path('unused'))
