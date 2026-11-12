"""Launch explicit, matched experiment configurations; dry-run by default."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def commands(config, dataset, partitions, variants, seeds, extra=()):
    for partition in partitions:
        for variant in variants:
            for seed in seeds:
                yield [sys.executable, str(ROOT / 'src/federated_main.py'),
                       *config['common'], *config['datasets'][dataset],
                       *config['partitions'][partition], *config['variants'][variant],
                       '--seed', str(seed), *extra]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT / 'experiments/core.json')
    p.add_argument('--dataset', choices=['fmnist', 'cifar10'], required=True)
    p.add_argument('--partitions', nargs='+', default=['dir', 'spc1', 'spc2'])
    p.add_argument('--variants', nargs='+', default=['baseline', 'dynamic', 'system', 'full'])
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--execute', action='store_true')
    args, extra = p.parse_known_args()
    config = json.loads(args.config.read_text())
    for command in commands(config, args.dataset, args.partitions, args.variants,
                            args.seeds or config['seeds'], extra):
        print(json.dumps(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)
