import numpy as np
from .common import TabularAgent, valid_actions


class QLearning(TabularAgent):
    name = "Q-Learning"

    def update(self, s, a, reward, next_s, next_mask, next_action=None,
               epsilon=0., terminated=False, truncated=False):
        bootstrap = 0. if terminated else np.max(self.q[next_s, valid_actions(next_mask)])
        self.td_update(s, a, reward + self.gamma*bootstrap)

