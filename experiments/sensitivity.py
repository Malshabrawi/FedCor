"""One-factor-at-a-time sensitivity suite; execution is opt-in."""
import argparse
import json
from pathlib import Path
import subprocess
from run import ROOT, commands

def settings():
    base={'dynamic_th_window':10,'dynamic_th_alpha':.2,'straggler_penalty':1.0}
    seen=set()
    for field,values in [('dynamic_th_window',[5,10,20]),('dynamic_th_alpha',[.1,.2,.4]),('straggler_penalty',[0.,.5,1.,2.])]:
        for value in values:
            point={**base,field:value};key=json.dumps(point,sort_keys=True)
            if key not in seen:seen.add(key);yield point

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['fmnist','cifar10'],required=True)
    p.add_argument('--partitions',nargs='+',default=['dir','spc1','spc2'])
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5]);p.add_argument('--execute',action='store_true')
    p.add_argument('--gpu', default=None);p.add_argument('--gpr_gpu', default=None)
    a=p.parse_args();config=json.loads((ROOT/'experiments/core.json').read_text())
    for point in settings():
        name='_'.join(f'{k}-{v}' for k,v in point.items())
        extra=[f'--{k}={v}' for k,v in point.items()]
        for flag in ['gpu','gpr_gpu']:
            if getattr(a,flag) is not None:extra.append('--'+flag+'='+getattr(a,flag))
        extra+=['--run_root='+str(ROOT/'runs/sensitivity'/a.dataset/name)]
        for command in commands(config,a.dataset,a.partitions,['baseline','full'],a.seeds,extra):
            print(json.dumps(command),flush=True)
            if a.execute:subprocess.run(command,cwd=ROOT,check=True)
