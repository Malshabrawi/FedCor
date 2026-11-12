import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment_io import SystemProfile
class TimingTests(unittest.TestCase):
    def test_adaptive_cost_and_synchronous_barrier(self):
        p=SystemProfile(2,1,3);p.communication=np.array([2.,3.]);p.per_epoch=np.array([1.,4.])
        self.assertEqual(p.round_seconds([0,1],[3,3]),15.)
        self.assertEqual(p.round_seconds([0,1],[3,6]),27.)
    def test_matched_profile(self):
        a=SystemProfile(100,1,3);b=SystemProfile(100,1,3)
        np.testing.assert_array_equal(a.communication,b.communication)
    def test_all_used_cuda_devices_synchronized(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from experiment_io import synchronize_devices
        with patch('torch.cuda.synchronize') as sync:
            synchronize_devices(SimpleNamespace(gpu='0',gpr_gpu='1',gpr=True))
            self.assertEqual({x.args[0] for x in sync.call_args_list},{0,1})
