import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from update import LocalUpdate
from models import MLP
class EpochTests(unittest.TestCase):
    def test_actual_steps_and_cap(self):
        data=torch.utils.data.TensorDataset(torch.randn(20,1,2,2),torch.arange(20)%2)
        for adaptive,expected in [(False,1),(True,3)]:
            args=SimpleNamespace(gpu=None,mu=0,verbose=0,local_bs=8,optimizer='sgd',lr=.01,
                momentum=0,reg=0,FedProx=False,adaptive_local_epochs=adaptive,local_ep=1,max_local_epochs=3)
            local=LocalUpdate(args,data,list(range(20)),latency=1,preferred_T=20)
            local.update_weights(MLP(4,[4],2))
            self.assertEqual(local.actual_epochs,expected)
            self.assertEqual(local.local_steps,expected*2)
