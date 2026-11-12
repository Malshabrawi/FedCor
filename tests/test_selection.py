import sys, unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from GPR import GPR
class DiagonalGP(GPR):
    def Covariance(self, ids=None):
        return torch.diag(torch.tensor([16.,4.,1.]))
class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.g=DiagonalGP(3); self.g.client_latencies=torch.tensor([100.,1.,1.])
        self.g.straggler_penalty=2;self.g.preferred_duration=20
    def test_positive_gain_stopping(self):
        self.assertEqual(self.g.Select_Clients(3,Dynamic=True,Dynamic_TH=1.5)[0],[0,1])
    def test_minimum(self):
        self.assertEqual(len(self.g.Select_Clients(3,Dynamic=True,Dynamic_TH=100,MinClients=2)[0]),2)
    def test_system_switch(self):
        self.assertEqual(self.g.Select_Clients(1,SystemAware=False)[0],[0])
        self.assertEqual(self.g.Select_Clients(1,SystemAware=True)[0],[1])
