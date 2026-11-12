import importlib.util,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
spec=importlib.util.spec_from_file_location('sensitivity',ROOT/'experiments/sensitivity.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
class SensitivityTests(unittest.TestCase):
    def test_unique_controlled_points(self):
        points=list(s.settings());self.assertEqual(len(points),8)
        base={'dynamic_th_window':10,'dynamic_th_alpha':.2,'straggler_penalty':1.}
        for point in points:self.assertLessEqual(sum(point[k]!=base[k] for k in base),1)
        self.assertEqual(sum(x==base for x in points),1)
