import importlib.util
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runner', ROOT/'experiments/run.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
class MatrixTests(unittest.TestCase):
    def test_matched_matrix(self):
        c = json.loads((ROOT/'experiments/core.json').read_text())
        cmds = list(runner.commands(c, 'cifar10', c['partitions'], c['variants'], c['seeds']))
        self.assertEqual(len(cmds), 60)
        self.assertEqual(len({tuple(x) for x in cmds}), 60)
        self.assertTrue(all('--warmup=15' in x and '--dataset=cifar' in x for x in cmds))
    def test_cifar_forward(self):
        import sys, torch
        from types import SimpleNamespace
        sys.path.insert(0, str(ROOT/'src'))
        from models import NaiveCNN
        args=SimpleNamespace(kernel_sizes=[3,3,3],num_filters=[32,64,64],padding=False,mlp_layers=[64],num_classes=10)
        model=NaiveCNN(args,[3,32,32],final_pool=False)
        loss=torch.nn.functional.cross_entropy(model(torch.randn(2,3,32,32)),torch.tensor([0,9]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
