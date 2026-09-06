import numpy as np
from .common import TabularAgent, valid_actions


class DynaQ(TabularAgent):
    name = "Dyna-Q"

    def __init__(self, seed=0, planning_steps=10, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if planning_steps < 0:
            raise ValueError("Planning steps cannot be negative")
        self.planning_steps = planning_steps
        self.model = {}
        self.model_keys = []
        self.planning_rng = np.random.default_rng(np.random.SeedSequence([seed, 73]))

    def update(self, s, a, reward, next_s, next_mask, next_action=None,
               epsilon=0., terminated=False, truncated=False):
        bootstrap = 0. if terminated else np.max(self.q[next_s, valid_actions(next_mask)])
        self.td_update(s, a, reward + self.gamma*bootstrap)
        key = (s, a)
        if key not in self.model:
            self.model_keys.append(key)
        # Classic last-observation Dyna model. Store the actual successor mask:
        # idle availability cannot be recovered from the binned state alone.
        self.model[key] = (reward, next_s, next_mask.copy(), terminated)
        for _ in range(self.planning_steps):
            ps, pa = self.model_keys[int(self.planning_rng.integers(len(self.model_keys)))]
            pr, pn, pm, terminal = self.model[ps, pa]
            future = 0. if terminal else np.max(self.q[pn, valid_actions(pm)])
            self.td_update(ps, pa, pr + self.gamma*future)

