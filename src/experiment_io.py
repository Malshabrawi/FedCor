"""Auditable run output and a shared synchronous client latency model."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid
import numpy as np

class SystemProfile:
    def __init__(self, count, seed, base_epochs):
        rng = np.random.default_rng(seed)
        self.communication = rng.lognormal(2.0, 0.7, count)
        computation = rng.lognormal(1.2, 0.5, count)
        mask = rng.random(count) < 0.1
        self.communication[mask] *= 6
        computation[mask] *= 6
        self.per_epoch = computation / base_epochs
    def durations(self, clients, epochs):
        clients = np.asarray(clients, dtype=int)
        return self.communication[clients] + self.per_epoch[clients] * np.asarray(epochs)
    def round_seconds(self, clients, epochs):
        times = self.durations(clients, epochs)
        return float(max(times)) if len(times) else 0.0
    def as_dict(self):
        return {'communication_seconds': self.communication.tolist(),
                'computation_seconds_per_epoch': self.per_epoch.tolist(),
                'sampler': {'communication_log_mean':2.0,'communication_log_sd':0.7,
                'base_computation_log_mean':1.2,'base_computation_log_sd':0.5,
                'straggler_probability':0.1,'straggler_multiplier':6}}

class RunRecorder:
    def __init__(self, args, seed, groups, profile):
        method = 'oort' if getattr(args,'oort',False) else 'afl' if args.afl else 'powerd' if args.power_d else 'gpr' if args.gpr else 'random'
        label = f'{method}_dynamic{int(args.dynamic_C)}_system{int(args.system_aware)}'
        self.path = Path(args.run_root) / (f'{args.dataset}_{label}_seed{seed}_'+uuid.uuid4().hex[:12])
        self.path.mkdir(parents=True,exist_ok=False)
        partition = {str(k):sorted(int(x) for x in v) for k,v in groups.items()}
        signature = hashlib.sha256(json.dumps(partition,sort_keys=True).encode()).hexdigest()
        try: commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        except (OSError, subprocess.CalledProcessError): commit='unknown'
        import torch, platform
        profile_hash = hashlib.sha256(json.dumps(profile.as_dict(),sort_keys=True).encode()).hexdigest()
        source_hash = hashlib.sha256()
        for source in sorted(Path(__file__).parent.rglob('*.py')):
            source_hash.update(str(source.relative_to(Path(__file__).parent)).encode())
            source_hash.update(source.read_bytes())
        config = dict(vars(args)); config.update(seed=seed,method=method,code_commit=commit,
            source_sha256=source_hash.hexdigest(),system_profile_sha256=profile_hash,
            partition_sha256=signature, torch_version=torch.__version__, python_version=platform.python_version(),
            torch_threads=torch.get_num_threads(), timing_definition='hybrid_simulated_clients_plus_measured_server')
        self.write('config.json',config);self.write('partition.json',partition);self.write('system_profile.json',profile.as_dict())
        self.elapsed = 0.0;self.wall_start=time.perf_counter()
    def write(self,name,data):
        (self.path/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    def round(self,data):
        self.elapsed += data['simulated_client_seconds']+data['gp_probe_client_seconds']+data['measured_server_seconds']
        data['cumulative_total_seconds']=self.elapsed
        with (self.path/'rounds.jsonl').open('a') as f:f.write(json.dumps(data,allow_nan=False)+'\n')
    def finish(self,rounds,accuracy):
        self.write('summary.json',{'status':'complete','rounds':rounds,'final_test_accuracy':accuracy,
                   'total_seconds':self.elapsed,'measured_execution_seconds':time.perf_counter()-self.wall_start})
