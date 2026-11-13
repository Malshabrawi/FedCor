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
    def fixture(self,tmp):
        import json
        for method in ['baseline','full']:
            for seed in [1,2]:
                folder=Path(tmp)/f'{method}_{seed}';folder.mkdir()
                cfg=dict(dataset='fmnist',model='mlp',mlp_layers=[16],num_filters=[32],kernel_sizes=[3],
                    epochs=2,num_users=2,local_ep=1,local_bs=8,lr=.01,schedule=[],lr_decay=.5,optimizer='sgd',reg=0,momentum=0,
                    warmup=1,alpha=None,shards_per_client=1,frac=1.,timing_definition='test',method='gpr',iid=0,
                    seed=seed,partition_sha256=str(seed),system_profile_sha256=str(seed),dynamic_th_alpha=.2,
                    dynamic_C=method=='full',system_aware=method=='full',adaptive_local_epochs=method=='full')
                (folder/'config.json').write_text(json.dumps(cfg))
                accuracy=.7 if method=='baseline' or seed==1 else .5
                (folder/'summary.json').write_text(json.dumps(dict(status='complete',rounds=2,total_seconds=20,final_test_accuracy=accuracy)))
                rows=[dict(round=n,test_accuracy=accuracy,cumulative_total_seconds=n*10,
                      simulated_client_seconds=7,gp_probe_client_seconds=2,measured_server_seconds=1,
                      clients=[0,1],actual_epochs=[1,1],local_steps=[1,1],
                      client_validation_accuracy=[.5,.7],participation_counts=[n,n]) for n in [1,2]]
                (folder/'rounds.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    def test_censoring_and_partition_mismatch(self):
        import tempfile,json
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            result=a.summarize(tmp,.6,'fmnist','spc1')
            pair=result['paired_comparisons']['full']
            self.assertEqual(pair['both_reach_target'],1)
            self.assertIsNone(pair['time_to_target'])
            self.assertEqual(pair['non_reaching_seeds'],[2])
            file=Path(tmp)/'full_2/config.json';cfg=json.loads(file.read_text());cfg['partition_sha256']='different';file.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):a.summarize(tmp,.6,'fmnist','spc1')
    def test_mixed_algorithm_settings_rejected(self):
        import tempfile,json
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            file=Path(tmp)/'full_2/config.json';cfg=json.loads(file.read_text());cfg['dynamic_th_alpha']=.4;file.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'Mixed algorithm settings'):a.summarize(tmp,.6,'fmnist','spc1')
    def test_corrupt_completed_runs_rejected(self):
        import tempfile,json
        for field,value in [('round',4),('test_accuracy',float('nan')),('cumulative_total_seconds',3),
                            ('participation_counts',[0,0]),('clients',[0,0]),('actual_epochs',[0,1])]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                self.fixture(tmp)
                file=Path(tmp)/'full_2/rounds.jsonl';rows=[json.loads(x) for x in file.read_text().splitlines()]
                rows[-1][field]=value;file.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
                with self.assertRaises(ValueError):a.summarize(tmp,.6,'fmnist','spc1')
    def test_truncated_complete_run_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            file=Path(tmp)/'full_2/rounds.jsonl';file.write_text(file.read_text().splitlines()[0]+'\n')
            with self.assertRaisesRegex(ValueError,'round count'):a.summarize(tmp,.6,'fmnist','spc1')
    def test_iid_runs_are_not_mislabeled_spc1(self):
        import tempfile,json
        with tempfile.TemporaryDirectory() as tmp:
            self.fixture(tmp)
            for file in Path(tmp).glob('*/config.json'):
                c=json.loads(file.read_text());c['iid']=1;file.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'No baseline'):a.summarize(tmp,.6,'fmnist','spc1')

    def test_reuse_main_baseline_without_retraining_or_copying(self):
        import tempfile,shutil,json
        with tempfile.TemporaryDirectory() as tmp:
            main=Path(tmp)/'main';main.mkdir();candidate=Path(tmp)/'candidate';candidate.mkdir()
            self.fixture(main)
            for folder in main.glob('full_*'):
                shutil.copytree(folder,candidate/folder.name)
                file=candidate/folder.name/'config.json';c=json.loads(file.read_text());c['dynamic_th_alpha']=.4;file.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'No baseline'):a.summarize(candidate,.6,'fmnist','spc1')
            shutil.copytree(main/'baseline_2',main/'baseline_3')
            file=main/'baseline_3/config.json';c=json.loads(file.read_text());c['seed']=3;file.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'Unmatched'):a.summarize(candidate,.6,'fmnist','spc1',baseline_root=main)
            result=a.summarize(candidate,.6,'fmnist','spc1',baseline_root=main,seeds=[1,2])
            self.assertEqual(set(result['methods']),{'baseline','full'})
            self.assertEqual(result['paired_comparisons']['full']['all_matched_seeds'],2)
            self.assertEqual(result['baseline_root'],str(main.resolve()))
            with self.assertRaisesRegex(ValueError,'Requested seeds missing'):a.summarize(candidate,.6,'fmnist','spc1',baseline_root=main,seeds=[1,2,3])
            file=main/'baseline_2/config.json';c=json.loads(file.read_text());c['partition_sha256']='wrong';file.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'Unmatched'):a.summarize(candidate,.6,'fmnist','spc1',baseline_root=main,seeds=[1,2])
