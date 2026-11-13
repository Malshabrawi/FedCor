"""Paired bootstrap comparisons; failures are reported, never silently discarded."""
import argparse
import json
from pathlib import Path
import numpy as np


def paired_interval(baseline, candidate, repetitions=10000, seed=0):
    a=np.asarray(baseline,float);b=np.asarray(candidate,float)
    if len(a)!=len(b) or len(a)<2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('At least two finite matched pairs are required')
    delta=b-a
    rng=np.random.default_rng(seed)
    means=delta[rng.integers(0,len(delta),(repetitions,len(delta)))].mean(axis=1)
    return {'mean_candidate_minus_baseline':float(delta.mean()),
            'paired_bootstrap_95_percent_interval':np.quantile(means,[.025,.975]).tolist(),
            'pairs':len(delta),'bootstrap_repetitions':repetitions,'bootstrap_seed':seed}


def fairness(round_record):
    accuracy=np.asarray(round_record['client_validation_accuracy'],float)
    participation=np.asarray(round_record['participation_counts'],float)
    return {'macro_validation_accuracy':float(accuracy.mean()),'minimum_validation_accuracy':float(accuracy.min()),
            'p10_validation_accuracy':float(np.quantile(accuracy,.1)),
            'bottom_decile_mean_validation_accuracy':float(np.sort(accuracy)[:max(1,int(np.ceil(len(accuracy)*.1)))].mean()),
            'client_accuracy_sd':float(accuracy.std()),'participation_coverage':float((participation>0).mean()),
            'participation_jain_index':float(participation.sum()**2/(len(participation)*np.square(participation).sum())) if participation.sum() else 0.0}


def variant(config):
    if config['method']!='gpr':return config['method']
    if config['dynamic_C'] and config['system_aware']:
        return 'full' if config['adaptive_local_epochs'] else 'full_fixed_epochs'
    return 'dynamic' if config['dynamic_C'] else 'system' if config['system_aware'] else 'baseline'


def validate_records(config, records, summary):
    """Reject incomplete/corrupt output before it can become a paper result."""
    if not records or summary.get('rounds') != len(records) or len(records) > config['epochs']:
        raise ValueError('Incomplete or inconsistent round count')
    elapsed=0.; participation=np.zeros(config['num_users'],dtype=int)
    for number,row in enumerate(records,1):
        if row.get('round') != number:
            raise ValueError('Non-contiguous round numbering')
        accuracy=np.asarray(row['client_validation_accuracy'],float)
        if accuracy.shape != (config['num_users'],) or not np.isfinite(accuracy).all() or np.any((accuracy<0)|(accuracy>1)):
            raise ValueError('Invalid client validation accuracies')
        if not np.isfinite(row['test_accuracy']) or not 0 <= row['test_accuracy'] <= 1:
            raise ValueError('Invalid test accuracy')
        clients=row['clients']
        if not clients or len(set(clients))!=len(clients) or any(type(c)!=int or not 0<=c<config['num_users'] for c in clients):
            raise ValueError('Invalid selected clients')
        participation[clients]+=1
        if not np.array_equal(row['participation_counts'],participation):
            raise ValueError('Participation counts disagree with selections')
        for field in ['actual_epochs','local_steps']:
            values=row[field]
            if len(values)!=len(clients) or any(type(x)!=int or x<1 for x in values):
                raise ValueError('Invalid local training work: '+field)
        costs=[row[f] for f in ['simulated_client_seconds','gp_probe_client_seconds','measured_server_seconds']]
        if not np.isfinite(costs).all() or min(costs)<0:
            raise ValueError('Invalid timing components')
        elapsed+=sum(costs)
        if not np.isclose(elapsed,row['cumulative_total_seconds'],rtol=1e-9,atol=1e-8):
            raise ValueError('Cumulative timing disagrees with components')
    if not np.isclose(summary.get('total_seconds',float('nan')),elapsed) or summary.get('final_test_accuracy')!=records[-1]['test_accuracy']:
        raise ValueError('Summary disagrees with recorded rounds')
    if len(records)<config['epochs'] and (config.get('target_accuracy') is None or records[-1]['test_accuracy']<config['target_accuracy']):
        raise ValueError('Training ended before the configured horizon without reaching its stopping target')


def summarize(root,target,dataset,partition):
    groups={};configurations={};method_configurations={}
    for folder in sorted(Path(root).iterdir()):
        if not (folder/'config.json').is_file():continue
        c=json.loads((folder/'config.json').read_text())
        if c.get('iid',0):continue
        setting='dir' if c['alpha'] is not None else 'spc'+str(c['shards_per_client'])
        if c['dataset']!=dataset or setting!=partition:continue
        if not (folder/'summary.json').is_file():raise ValueError('Incomplete run: '+str(folder))
        summary=json.loads((folder/'summary.json').read_text())
        if summary['status']!='complete':raise ValueError('Failed run: '+str(folder))
        records=[json.loads(x) for x in (folder/'rounds.jsonl').read_text().splitlines()]
        validate_records(c,records,summary)
        key=(c['seed'],c['partition_sha256'],c['system_profile_sha256'])
        name=variant(c)
        ignored={'seed','partition_sha256','system_profile_sha256','run_root','data_dir','code_commit'}
        method_signature={k:v for k,v in c.items() if k not in ignored}
        if name in method_configurations and method_signature!=method_configurations[name]:
            raise ValueError('Mixed algorithm settings across seeds: '+name)
        method_configurations[name]=method_signature
        if key in groups.setdefault(name,{}):raise ValueError('Duplicate seed/configuration: '+name)
        # Comparison parameters that must match across algorithms and seeds.
        fields=['dataset','model','mlp_layers','num_filters','kernel_sizes','epochs','num_users',
                'local_ep','local_bs','lr','schedule','lr_decay','optimizer','reg','momentum','warmup',
                'alpha','shards_per_client','frac','timing_definition','iid','unequal','num_classes',
                'dynamic_frac','frac_min','frac_max','warmup_rounds','global_average','FedProx','mu','dynamic_mu',
                'target_accuracy','source_sha256','torch_version','torch_threads','package_versions',
                'gpu','gpr_gpu','gpr_begin','group_size','GPR_interval','GPR_gamma','GPR_Epoch','update_mean',
                'kernel','poly_norm','dimension','train_method','discount','epsilon_greedy','gp_noise']
        signature={f:c.get(f) for f in fields}
        if configurations and signature!=next(iter(configurations.values())):
            raise ValueError('Mixed experimental conditions; use a separate run root per suite')
        configurations[name]=signature
        hit=next((r for r in records if r['test_accuracy']>=target),None)
        groups[name][key]={'seed':c['seed'],'time_to_target':hit['cumulative_total_seconds'] if hit else None,
                           'final_accuracy':records[-1]['test_accuracy'],'fairness':fairness(records[-1])}
    if 'baseline' not in groups:raise ValueError('No baseline runs found')
    output={'dataset':dataset,'partition':partition,'target_accuracy':target,
        'timing':'hybrid estimate, including warmup','fairness_population':'client-held-out validation subsets of training partition',
        'methods':{},'paired_comparisons':{},'configurations':method_configurations}
    for name,items in groups.items():
        hits=[x['time_to_target'] for x in items.values() if x['time_to_target'] is not None]
        output['methods'][name]={'runs':list(items.values()),'successes':len(hits),'total':len(items),
            'conditional_mean_time_among_successes':float(np.mean(hits)) if hits else None,
            'conditional_sample_sd':float(np.std(hits,ddof=1)) if len(hits)>1 else None}
        if name=='baseline':continue
        baseline=groups['baseline']
        if set(items)!=set(baseline):raise ValueError('Unmatched seeds, partitions or system profiles: '+name)
        keys=sorted(items)
        complete=[k for k in keys if items[k]['time_to_target'] is not None and baseline[k]['time_to_target'] is not None]
        pair={'all_matched_seeds':len(keys),'both_reach_target':len(complete),
            'non_reaching_seeds':[k[0] for k in keys if k not in complete],
            'scope':'time interval is conditional on BOTH methods reaching target; not an unconditional speedup estimate'}
        pair['fairness']={metric:paired_interval([baseline[k]['fairness'][metric] for k in keys],[items[k]['fairness'][metric] for k in keys]) for metric in next(iter(items.values()))['fairness']} if len(keys)>1 else None
        pair['final_accuracy']=paired_interval([baseline[k]['final_accuracy'] for k in keys],[items[k]['final_accuracy'] for k in keys]) if len(keys)>1 else None
        pair['time_to_target']=paired_interval([baseline[k]['time_to_target'] for k in complete],[items[k]['time_to_target'] for k in complete]) if len(complete)>1 else None
        output['paired_comparisons'][name]=pair
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--target',type=float,required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--partition',choices=['dir','spc1','spc2'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not 0<a.target<=1:p.error('target must be a fraction in (0,1]')
    result=summarize(a.root,a.target,a.dataset,a.partition)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
