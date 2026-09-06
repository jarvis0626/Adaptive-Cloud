import numpy as np
import pytest
from environment import CloudEnvironment
from baseline import ThresholdAutoscaler


@pytest.mark.parametrize("n", range(1, 6))
def test_minimum_maximum_and_idle_masks(n):
    env = CloudEnvironment([0., 0., 0.], initial_instances=n)
    env.reset(seed=1)
    assert env.action_mask().tolist() == [n>1, True, n<5]
    if n == 1:
        with pytest.raises(ValueError): env.step(0)
    if n == 5:
        with pytest.raises(ValueError): env.step(2)


def test_scale_in_never_kills_inflight_request():
    env = CloudEnvironment([0., 0., 0.], initial_instances=2, trace=([59.99], [2.]))
    env.reset(seed=1)
    env.step(1)
    assert env.action_mask()[0]
    env.step(0)
    assert env.active_instances == 1
    assert env.outcomes[0] == 2
    assert env.resolved[0] == pytest.approx(61.99)


def test_busy_pool_forbids_scale_in():
    env = CloudEnvironment([0., 0., 0.], initial_instances=2, trace=([59.99, 59.995], [2., 2.]))
    env.reset(seed=1)
    env.step(1)
    assert not env.action_mask()[0]
    with pytest.raises(ValueError): env.step(0)


def test_no_second_scale_while_pending():
    env = CloudEnvironment([0., 0., 0.], initial_instances=4)
    env.reset(seed=0)
    env.step(2)
    assert env.active_instances+env.pending == 5
    for action in [0, 2]:
        with pytest.raises(ValueError): env.step(action)
    env.step(1)
    assert env.active_instances == 5 and env.pending == 0


def test_baseline_thresholds_cooldown_stabilization_masks():
    baseline = ThresholdAutoscaler()
    all_valid = np.ones(3, dtype=bool)
    assert baseline.select(dict(action_mask=all_valid, utilization=.9)) == 2
    assert baseline.select(dict(action_mask=all_valid, utilization=.9)) == 1
    assert baseline.select(dict(action_mask=np.array([False, True, False]), utilization=.99)) == 1
    assert baseline.select(dict(action_mask=all_valid, utilization=.1)) == 1
    assert baseline.select(dict(action_mask=all_valid, utilization=.1)) == 0

