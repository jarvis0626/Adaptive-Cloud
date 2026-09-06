import numpy as np
import pytest
import yaml
from algorithms import QLearning, SARSA, ExpectedSARSA, DynaQ
from algorithms.common import policy_probabilities, epsilon_at
from innovation import PredictiveQLearning
from environment import CloudEnvironment
from evaluation.evaluate import evaluate_policy
from evaluation.metrics import stabilization_episode, cohort_metrics
from train_all import ROOT, train_agent

CLASSES = [QLearning, SARSA, ExpectedSARSA, DynaQ, PredictiveQLearning]


@pytest.mark.parametrize("cls", CLASSES)
def test_masked_exploration_greedy_targets_and_truncation(cls):
    agent = cls(seed=1, alpha=1., gamma=.5, planning_steps=0)
    agent.q[1] = [999., 2., 4.]
    mask = np.array([False, True, True])
    for epsilon in [0., .5, 1.]:
        assert all(agent.select(1, mask, epsilon) in [1, 2] for _ in range(100))
    agent.update(0, 1, 1., 1, mask, next_action=1, epsilon=.2, truncated=True)
    expected = 2. if cls == SARSA else (2.9 if cls == ExpectedSARSA else 3.)
    assert agent.q[0, 1] == pytest.approx(expected)
    agent.update(0, 1, 1., 1, mask, next_action=1, epsilon=.2, terminated=True)
    assert agent.q[0, 1] == 1.


def test_expected_policy_probabilities_and_ties():
    np.testing.assert_allclose(policy_probabilities(np.array([999, 2, 4]), [False, True, True], .2), [0, .1, .9])
    np.testing.assert_allclose(policy_probabilities(np.zeros(3), [True]*3, .3), [.1, .8, .1])


def test_dyna_planning_uses_stored_mask():
    agent = DynaQ(seed=1, alpha=.5, gamma=.5, planning_steps=10)
    agent.q[1] = [999., 2., 999.]
    agent.update(0, 1, 1., 1, np.array([False, True, False]))
    assert agent.updates == 11
    assert agent.q[0, 1] == pytest.approx(2*(1-.5**11))


@pytest.mark.parametrize("cls", CLASSES)
def test_training_smoke_all_agents_update_and_reproduce(cls):
    config = yaml.safe_load((ROOT/"config.yaml").read_text())
    config["training"].update(episodes=2, steps=3)
    first, rows, _, _ = train_agent(cls, 11, config)
    second, repeated, _, _ = train_agent(cls, 11, config)
    assert np.any(first.q)
    np.testing.assert_array_equal(first.q, second.q)
    assert rows == repeated


@pytest.mark.parametrize("cls", CLASSES)
def test_evaluation_is_frozen_and_cohort_is_resolved(cls, tmp_path):
    config = yaml.safe_load((ROOT/"config.yaml").read_text())
    config["evaluation"]["save_request_logs"] = False
    policy = cls(seed=2)
    before = policy.q.copy()
    metrics, episodes = evaluate_policy(policy, 2, [dict(name="test", seed=999, rates=[100., 220., 400.])], config, tmp_path)
    np.testing.assert_array_equal(before, policy.q)
    assert metrics["arrivals"] == sum(metrics[k] for k in ["timely_completions", "late_completions", "rejections", "timeouts"])
    assert len(episodes) == 1


def test_epsilon_schedule_and_stabilization_censoring():
    assert epsilon_at(0, 100) == 1.
    assert epsilon_at(79, 100) == pytest.approx(.05)
    assert epsilon_at(99, 100) == pytest.approx(.05)
    assert stabilization_episode([1.]*20) is None
    assert stabilization_episode([1.]*120) == 100
    assert stabilization_episode([0.]*50+[2.]*50) is None


def test_pooled_p95_and_failures():
    log = dict(outcome=np.array([1, 2, 3, 4]), arrival_seconds=np.zeros(4), resolution_seconds=np.array([.1, .3, 0, 10.]))
    metrics = cohort_metrics(log)
    assert metrics["p95_response_time_ms"] == pytest.approx(290.)
    assert metrics["sla_violation_fraction"] == .75
