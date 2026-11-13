import contextlib, io, sys, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from options import args_parser, get_dynamic_num_clients

class OptionsTests(unittest.TestCase):
    def parse(self,*flags):
        with patch.object(sys,'argv',['test',*flags]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return args_parser()
    def test_invalid_settings_fail_early(self):
        for flags in [('--epochs=0',),('--num_users=0',),('--frac_max=2',),('--frac_min=.3','--frac_max=.2'),
                      ('--GPR_interval=0',),('--dynamic_th_percentile=101',),('--seed',),
                      ('--power_d','--num_users=10'),('--power_d','--d=1'),('--dynamic_frac=5',),
                      ('--alpha=-1',),('--target_accuracy=90',)]:
            with self.subTest(flags=flags), self.assertRaises(SystemExit):self.parse(*flags)
    def test_defaults_and_alias(self):
        args=self.parse('--gpr')
        self.assertFalse(args.system_aware);self.assertFalse(args.dynamic_C)
        self.assertTrue(self.parse('--gpr','--sysetm_aware').adaptive_local_epochs)
    def test_adaptive_count_responds_to_loss(self):
        args=self.parse('--dynamic_frac=3','--warmup_rounds=2')
        self.assertEqual(get_dynamic_num_clients(0,10,args,[]),20)
        self.assertEqual(get_dynamic_num_clients(2,10,args,[1.,1.]),5)
        self.assertEqual(get_dynamic_num_clients(2,10,args,[0.,2.]),20)

    def test_auto_cpu_fallback(self):
        with patch('torch.cuda.is_available',return_value=False):
            args=self.parse('--gpr')
            self.assertIsNone(args.gpu);self.assertIsNone(args.gpr_gpu)
    def test_auto_cuda_and_inheritance(self):
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.device_count',return_value=2):
            args=self.parse('--gpr')
            self.assertEqual((args.gpu,args.gpr_gpu),('0','0'))
            args=self.parse('--gpr','--gpu=1')
            self.assertEqual((args.gpu,args.gpr_gpu),('1','1'))
    def test_explicit_cpu_and_separate_gp(self):
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.device_count',return_value=2):
            args=self.parse('--gpr','--gpu=cpu')
            self.assertEqual((args.gpu,args.gpr_gpu),(None,None))
            args=self.parse('--gpr','--gpu=1','--gpr_gpu=cpu')
            self.assertEqual((args.gpu,args.gpr_gpu),('1',None))
            args=self.parse('--gpr','--gpu=cpu','--gpr_gpu=cuda:1')
            self.assertEqual((args.gpu,args.gpr_gpu),(None,'1'))
    def test_unavailable_explicit_gpu_rejected(self):
        with patch('torch.cuda.is_available',return_value=False),self.assertRaises(SystemExit):self.parse('--gpu=0')
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.device_count',return_value=1),self.assertRaises(SystemExit):self.parse('--gpr_gpu=1')
        with self.assertRaises(SystemExit):self.parse('--gpu=invalid')
