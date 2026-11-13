"""Full-only sensitivity sweep; reuse baseline/default Full from main experiments."""
import argparse
import json
from pathlib import Path
import subprocess
from run import ROOT, commands

REFERENCE = {'dynamic_th_window':10, 'dynamic_th_alpha':.2, 'straggler_penalty':1.0}


def settings(include_reference=True):
    seen=set()
    for field,values in [('dynamic_th_window',[5,10,20]),('dynamic_th_alpha',[.1,.2,.4]),('straggler_penalty',[0.,.5,1.,2.])]:
        for value in values:
            point={**REFERENCE,field:value};key=json.dumps(point,sort_keys=True)
            if key not in seen:
                seen.add(key)
                if include_reference or point!=REFERENCE:
                    yield point


def sweep_commands(config,dataset,partitions,seeds,run_root,include_reference=False,device_options=()):
    for point in settings(include_reference):
        name='_'.join(f'{k}-{v}' for k,v in point.items())
        extra=[f'--{k}={v}' for k,v in point.items()]
        extra += list(device_options)+['--run_root='+str(Path(run_root)/dataset/name)]
        yield from commands(config,dataset,partitions,['full'],seeds,extra)


if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--dataset',choices=['fmnist','cifar10'],required=True)
    p.add_argument('--partitions',nargs='+',choices=['dir','spc1','spc2'],default=['dir','spc1','spc2'])
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2])
    p.add_argument('--execute',action='store_true')
    p.add_argument('--gpu',default=None);p.add_argument('--gpr_gpu',default=None)
    p.add_argument('--config',type=Path,default=ROOT/'experiments/core.json')
    p.add_argument('--run_root',type=Path,default=ROOT/'runs/sensitivity')
    p.add_argument('--include-reference',action='store_true',help='Also train default Full if main results are unavailable')
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds) or len(set(a.partitions))!=len(a.partitions):
        p.error('Seeds and partitions must be unique')
    config=json.loads(a.config.read_text())
    devices=['--'+flag+'='+getattr(a,flag) for flag in ['gpu','gpr_gpu'] if getattr(a,flag) is not None]
    for command in sweep_commands(config,a.dataset,a.partitions,a.seeds,a.run_root,a.include_reference,devices):
        print(json.dumps(command),flush=True)
        if a.execute:
            subprocess.run(command,cwd=ROOT,check=True)
