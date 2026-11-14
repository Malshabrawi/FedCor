"""Optional W&B metrics; credentials are read by the SDK, never stored in config."""
import atexit
import os
import numpy as np


class WandbTracker:
    def __init__(self,args,config,path):
        self.run=None;self.closed=False
        if not getattr(args,'wandb',False) or getattr(args,'wandb_mode','online')=='disabled':
            return
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError('W&B requested: install it with python3 -m pip install wandb') from error
        partition='iid' if config.get('iid') else 'dir' if config.get('alpha') is not None else 'spc'+str(config.get('shards_per_client'))
        variant=config['method']
        if variant=='gpr':
            variant='full' if config['dynamic_C'] and config['system_aware'] else 'dynamic' if config['dynamic_C'] else 'system' if config['system_aware'] else 'baseline'
            if variant=='full' and not config.get('adaptive_local_epochs',True):variant='full_fixed_epochs'
        self.run=wandb.init(project=args.wandb_project,entity=os.environ.get('WANDB_ENTITY'),
            mode=args.wandb_mode,name=path.name,group=config['dataset']+'_'+partition+'_'+variant,
            tags=[config['dataset'],partition,variant],config=config,dir=str(path))
        if self.run is None:raise RuntimeError('W&B did not initialize a run')
        self.run.define_metric('round')
        self.run.define_metric('*',step_metric='round')
        atexit.register(self._finish_incomplete)

    def log(self,row):
        if self.run is None:return
        metrics={key:value for key,value in row.items() if isinstance(value,(int,float)) and not isinstance(value,bool)}
        metrics['selected_clients']=len(row.get('clients',[]))
        for field in ['actual_epochs','local_steps']:
            if row.get(field):metrics[field+'_mean']=float(np.mean(row[field]))
        if row.get('client_validation_accuracy'):
            values=np.asarray(row['client_validation_accuracy'],float)
            metrics.update(client_accuracy_mean=float(values.mean()),client_accuracy_min=float(values.min()),client_accuracy_p10=float(np.quantile(values,.1)))
        if row.get('participation_counts'):
            counts=np.asarray(row['participation_counts'],float)
            metrics['participation_coverage']=float((counts>0).mean())
            metrics['participation_jain_index']=float(counts.sum()**2/(len(counts)*(counts**2).sum())) if counts.sum() else 0.
        self.run.log(metrics)

    def finish(self,summary=None,exit_code=0):
        if self.run is None or self.closed:return
        self.closed=True
        if summary is not None:self.run.summary.update(summary)
        self.run.finish(exit_code=exit_code)

    def _finish_incomplete(self):
        if self.closed or self.run is None:return
        try:self.finish(exit_code=1)
        except Exception:pass
