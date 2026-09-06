import numpy as np
from .common import TabularAgent, policy_probabilities


class ExpectedSARSA(TabularAgent):
    name = "Expected SARSA"

    def update(self, s, a, reward, next_s, next_mask, next_action=None,
               epsilon=0., terminated=False, truncated=False):
        expectation = 0. if terminated else float(np.dot(
            policy_probabilities(self.q[next_s], next_mask, epsilon), self.q[next_s]))
        self.td_update(s, a, reward + self.gamma*expectation)

