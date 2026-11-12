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
    def test_censoring_and_partition_mismatch(self):
        import tempfile,json
        core=json.loads((Path(__file__).resolve().parents[1]/'experiments/core.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            for method in ['baseline','full']:
                for seed in [1,2]:
                    folder=Path(tmp)/f'{method}_{seed}';folder.mkdir()
                    cfg=dict(dataset='fmnist',model='mlp',mlp_layers=[16],num_filters=[32],kernel_sizes=[3],
                        epochs=2,num_users=2,local_ep=1,local_bs=8,lr=.01,schedule=[],lr_decay=.5,optimizer='sgd',reg=0,momentum=0,
                        warmup=1,alpha=None,shards_per_client=1,frac=1.,timing_definition='test',method='gpr',
                        seed=seed,partition_sha256=str(seed),system_profile_sha256=str(seed),
                        dynamic_C=method=='full',system_aware=method=='full',adaptive_local_epochs=method=='full')
                    (folder/'config.json').write_text(json.dumps(cfg));(folder/'summary.json').write_text('{"status":"complete"}')
                    row=dict(test_accuracy=.7 if method=='baseline' or seed==1 else .5,
                        cumulative_total_seconds=10,client_validation_accuracy=[.5,.7],participation_counts=[1,1])
                    (folder/'rounds.jsonl').write_text(json.dumps(row)+'\n')
            result=a.summarize(tmp,.6,'fmnist','spc1')
            pair=result['paired_comparisons']['full']
            self.assertEqual(pair['both_reach_target'],1)
            self.assertIsNone(pair['time_to_target'])
            self.assertEqual(pair['non_reaching_seeds'],[2])
            file=Path(tmp)/'full_2/config.json';cfg=json.loads(file.read_text());cfg['partition_sha256']='different';file.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):a.summarize(tmp,.6,'fmnist','spc1')
