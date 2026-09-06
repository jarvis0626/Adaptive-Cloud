from .common import TabularAgent


class SARSA(TabularAgent):
    name = "SARSA"

    def update(self, s, a, reward, next_s, next_mask, next_action=None,
               epsilon=0., terminated=False, truncated=False):
        if not terminated and (next_action not in (0, 1, 2) or not next_mask[next_action]):
            raise ValueError("SARSA needs the valid next action that will actually execute")
        bootstrap = 0. if terminated else self.q[next_s, next_action]
        self.td_update(s, a, reward + self.gamma*bootstrap)

