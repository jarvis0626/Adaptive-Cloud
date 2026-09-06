"""Masked policy utilities shared by independently implemented TD updates."""
import numpy as np
from environment.cloud_environment import state_to_index


def valid_actions(mask):
    actions = np.flatnonzero(mask)
    if not len(actions):
        raise ValueError("Action mask cannot be empty")
    return actions


def greedy_action(values, mask):
    best = np.max(values[valid_actions(mask)])
    # Fixed tie convention, shared by training, expectation and evaluation.
    for action in (1, 2, 0):
        if mask[action] and values[action] == best:
            return action
    raise ValueError("Nonfinite action values")


def policy_probabilities(values, mask, epsilon):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0, 1]")
    actions = valid_actions(mask)
    probs = np.zeros(3)
    probs[actions] = epsilon / len(actions)
    probs[greedy_action(values, mask)] += 1-epsilon
    return probs


def epsilon_at(interaction, total_interactions, initial=1., minimum=.05, decay_fraction=.8):
    decay_steps = max(1, int(total_interactions * decay_fraction) - 1)
    return initial + (minimum-initial)*min(interaction/decay_steps, 1.)


class TabularAgent:
    name = "Abstract"
    predictive = False

    def __init__(self, seed=0, alpha=.1, gamma=.95, **kwargs):
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("Invalid learning rate or discount")
        self.q = np.zeros((810 if self.predictive else 270, 3))
        self.alpha, self.gamma = alpha, gamma
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def encode(self, state, info):
        return state_to_index(state)

    def select(self, state, mask, epsilon=0.):
        if not 0 <= epsilon <= 1:
            raise ValueError("Invalid epsilon")
        if epsilon and self.rng.random() < epsilon:
            return int(self.rng.choice(valid_actions(mask)))
        return greedy_action(self.q[state], mask)

    def td_update(self, s, a, target):
        self.q[s, a] += self.alpha*(target-self.q[s, a])
        self.updates += 1

