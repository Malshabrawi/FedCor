import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('summarize',Path(__file__).resolve().parents[1]/'analysis/summarize.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class StatisticsTests(unittest.TestCase):
    def test_pairing(self):
        r=a.paired_interval([100,1000,20],[97,997,17])
        self.assertEqual(r['mean_candidate_minus_baseline'],-3.)
        self.assertEqual(r['paired_bootstrap_95_percent_interval'],[-3.,-3.])
    def test_insufficient_pairs(self):
        with self.assertRaises(ValueError):a.paired_interval([1],[2])
    def test_fairness(self):
        f=a.fairness({'client_validation_accuracy':[.2,.8],'participation_counts':[2,2]})
        self.assertEqual(f['participation_jain_index'],1.)
        self.assertEqual(f['minimum_validation_accuracy'],.2)
