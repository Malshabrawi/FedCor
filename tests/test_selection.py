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
    def test_epsilon_exploration_contract(self):
        selected,latencies,seconds,gains=self.g.Select_Clients(2,epsilon=1)
        self.assertEqual(len(set(selected)),2)
        self.assertEqual(len(latencies),2)
        self.assertGreater(seconds,0)
        self.assertIsNone(gains)
    def test_dense_covariance_matches_independent_numpy_reference(self):
        import numpy as np
        class DenseGP(GPR):
            def Covariance(self,ids=None):return self.cov
        for seed in range(5):
            rng=np.random.default_rng(seed);matrix=rng.normal(size=(8,8))
            covariance=matrix@matrix.T+np.eye(8)*.2
            g=DenseGP(8);g.cov=torch.tensor(covariance,dtype=torch.float64)
            g.client_latencies=torch.tensor(np.arange(8)+1.)
            g.discount=torch.tensor(np.linspace(.5,1,8))
            weights=np.arange(1,9,dtype=float)/36
            for system in [False,True]:
                g.preferred_duration=3;g.straggler_penalty=2
                remaining=list(range(8));expected=[];gains=[];sigma=covariance.copy()
                for _ in range(6):
                    block=sigma[np.ix_(remaining,remaining)]
                    score=(weights[remaining,None]*block).sum(axis=0)*g.discount.numpy()[remaining]/np.sqrt(block.diagonal())
                    if system:score*=np.minimum(1,3/(np.array(remaining)+1))**2
                    choice=int(np.argmax(score));client=remaining[choice]
                    expected.append(client);gains.append(score[choice])
                    sigma-=np.outer(sigma[:,client],sigma[client,:])/sigma[client,client]
                    remaining.remove(client)
                actual,_,_,actual_gains=g.Select_Clients(6,weights=weights,SystemAware=system)
                self.assertEqual(actual,expected)
                np.testing.assert_allclose(actual_gains,gains,rtol=1e-6,atol=1e-7)
    def test_disabled_dynamic_and_zero_penalty_equivalence(self):
        reference=self.g.Select_Clients(3)[0]
        self.assertEqual(self.g.Select_Clients(3,Dynamic=False,Dynamic_TH=1e20)[0],reference)
        self.g.straggler_penalty=0
        self.assertEqual(self.g.Select_Clients(3,SystemAware=True)[0],reference)
