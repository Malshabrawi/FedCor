import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from oort_adapter import OortSelector
class OortTests(unittest.TestCase):
    def test_feedback_selection_and_rng(self):
        args=SimpleNamespace(oort_exploration_factor=.9,oort_pacer_step=20,oort_round_penalty=2)
        a=OortSelector(args,1,[100]*20,[float(i+1) for i in range(20)])
        np.random.seed(10);expected=np.random.rand();np.random.seed(10)
        for round_no in range(1,8):
            selected=a.select(5)
            self.assertEqual(len(set(selected)),5)
            for i in selected:a.update(i,10+i,float(i+1),round_no)
        self.assertEqual(np.random.rand(),expected)
