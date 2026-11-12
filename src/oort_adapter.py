"""Adapter around the authors' pinned Oort training selector."""
from types import SimpleNamespace
import numpy as np
from vendor.oort_training import _training_selector

class OortSelector:
    def __init__(self,args,seed,sizes,durations):
        defaults=dict(exploration_factor=args.oort_exploration_factor,exploration_decay=.95,
            exploration_min=.2,exploration_alpha=.3,round_threshold=10.,sample_window=5.,
            pacer_step=args.oort_pacer_step,pacer_delta=5.,blacklist_rounds=-1,
            blacklist_max_len=.3,clip_bound=.98,cut_off_util=.7,round_penalty=args.oort_round_penalty)
        self.rng=np.random.RandomState(seed)
        state=np.random.get_state()
        try:self.selector=_training_selector(SimpleNamespace(**defaults),sample_seed=seed)
        finally:np.random.set_state(state)
        for i,(size,duration) in enumerate(zip(sizes,durations)):
            self.selector.register_client(i,{'reward':float(size),'duration':float(duration)})
    def select(self,count):
        state=np.random.get_state();np.random.set_state(self.rng.get_state())
        try:
            if not any(a['count'] for a in self.selector.totalArms.values()):
                return self.rng.choice(list(self.selector.totalArms),count,replace=False).tolist()
            selected=self.selector.select_participant(count)
            self.rng.set_state(np.random.get_state())
            return [int(i) for i in selected]
        finally:np.random.set_state(state)
    def update(self,client,utility,duration,round_number):
        self.selector.update_client_util(client,{'reward':max(float(utility),1e-12),
             'duration':duration,'time_stamp':round_number,'status':True})
