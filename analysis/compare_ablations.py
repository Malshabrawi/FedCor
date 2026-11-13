"""Independently compare legacy logs without inferring method from file names."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

def extract(path,target):
    text=Path(path).read_text();header=text.split('Start with Random Seed:')[0]
    config=dict(re.findall(r'^([A-Za-z_]+): (.*)$',header,re.M))
    seed=None;round_no=None;accuracy=None;rows={};hits={}
    for line in text.splitlines():
        if line.startswith('Start with Random Seed:'):seed=line.split(':')[-1].strip();rows[seed]=[]
        if line.startswith('Avg Training Stats after'):round_no=int(line.split()[4]);accuracy=None
        if line.startswith('Test Accuracy:'):accuracy=float(line.split(':')[1].strip().strip('%'))
        if line.startswith('Round Simulated Wall-Clock Time:') and accuracy is not None:
            seconds=float(line.split(':')[1].split()[0]);rows[seed].append([round_no,accuracy,seconds])
            if accuracy>=target and seed not in hits:hits[seed]=seconds
    selections=re.findall(r'GPR Chosen Clients(?: \(Truncated\))?:\s*\[([^\]]*)\]',text)
    selections=[[int(x) for x in re.findall(r'\d+',s)] for s in selections]
    values=list(hits.values())
    return {'sha256':hashlib.sha256(text.encode()).hexdigest(),'config':config,'records':rows,
            'selections':selections,'crossing_seconds':hits,
            'mean_seconds':statistics.mean(values) if values else None,
            'population_sd_seconds':statistics.pstdev(values) if values else None}

def compare(left,right,target=60):
    a=extract(left,target);b=extract(right,target)
    if not a['records'] or not b['records']:raise ValueError('No cumulative-time records found')
    return {'left_path':str(left),'right_path':str(right),'target_percent':target,
        'byte_identical':a['sha256']==b['sha256'],
        'all_recorded_round_accuracy_time_equal':a['records']==b['records'],
        'all_recorded_client_selections_equal':a['selections']==b['selections'],
        'left':{k:v for k,v in a.items() if k not in ['records','selections']},
        'right':{k:v for k,v in b.items() if k not in ['records','selections']},
        'interpretation':'Check actual dynamic_C and dynamic_frac flags; a filename does not identify an active full method.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path)
    p.add_argument('--target',type=float,default=60);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=compare(a.left,a.right,a.target);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
