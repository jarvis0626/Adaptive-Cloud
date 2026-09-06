import numpy as np
import pytest
from environment import CloudEnvironment
from environment.cloud_environment import state_to_index, index_to_state, arrival_band
from environment.workload_generator import request_trace, training_rates, evaluation_rates


def scripted(arrivals, service, steps=1, **kwargs):
    env = CloudEnvironment(np.zeros(steps+1), trace=(arrivals, service), **kwargs)
    env.reset(seed=0)
    return env


def test_fifo_service_latency_rejection_and_timeout():
    # One server: A starts at 0 and finishes at 2; B waits, then finishes at 3.
    # C waits behind B and expires at arrival+10. D is rejected by the full queue.
    env = scripted([0., .1, .2, .3], [2., 1., 20., 1.], initial_instances=1, queue_capacity=2)
    _, _, _, truncated, info = env.step(1)
    assert truncated
    np.testing.assert_allclose(env.starts[:3], [0, 2, 3])
    np.testing.assert_allclose(env.resolved, [2, 3, 10.2, .3])
    np.testing.assert_array_equal(env.outcomes, [2, 2, 4, 3])
    assert info["late_completions"] == 2 and info["timeouts"] == 1 and info["rejections"] == 1
    assert info["p95_response_time_ms"] == pytest.approx(2855.)


def test_waiting_timeout_cancels_without_service():
    env = scripted([0., .01], [20., .1], initial_instances=1, timeout=1.)
    env.step(1)
    assert env.outcomes.tolist() == [4, 4]
    np.testing.assert_allclose(env.resolved, [1., 1.01])


def test_drain_resolves_last_arrival_and_counts_service_timeout():
    env = scripted([59.99], [30.], initial_instances=1)
    env.step(1)
    assert env.outcomes[0] == 0
    drain = env.drain()
    assert drain["timeouts"] == 1
    assert env.outcomes[0] == 4
    assert env.resolved[0] == pytest.approx(69.99)
    assert drain["instance_seconds"] == 10
    with pytest.raises(RuntimeError):
        env.drain()


def test_warmup_is_measured_and_excluded_from_cohort():
    env = scripted([-1., 1.], [.01, .02])
    assert env.latest_rate == 1/60
    assert env.trend == 1
    env.step(1)
    env.drain()
    assert len(env.request_log()["outcome"]) == 1


def test_pending_startup_exact_timing():
    env = scripted([1., 61., 121.], [.01]*3, steps=3, initial_instances=1)
    state, _, _, _, info = env.step(2)
    assert state[2:4] == (1, 1)
    assert info["interval_active_instances"] == 1
    assert info["instance_seconds"] == 120
    assert info["action_mask"].tolist() == [False, True, False]
    state, _, _, _, info = env.step(1)
    assert state[2:4] == (2, 0)
    assert info["interval_active_instances"] == 2
    assert info["scaling_action"] == 0


def test_reproducible_trace_and_environment():
    rates = [90., 220., 450., 80.]
    left, right = CloudEnvironment(rates), CloudEnvironment(rates)
    assert left.reset(seed=72)[0] == right.reset(seed=72)[0]
    for action in [2, 1, 1]:
        a, b = left.step(action), right.step(action)
        assert a[:4] == b[:4]
        for key in a[4]:
            if key != "action_mask":
                assert a[4][key] == b[4][key]
        np.testing.assert_array_equal(a[4]["action_mask"], b[4]["action_mask"])
    np.testing.assert_array_equal(left.outcomes, right.outcomes)
    left.drain()
    assert (left.outcomes > 0).all()


@pytest.mark.parametrize("predictive,size", [(False, 270), (True, 810)])
def test_state_index_bijection(predictive, size):
    for index in range(size):
        assert state_to_index(index_to_state(index, predictive)) == index
    with pytest.raises(ValueError):
        index_to_state(size, predictive)


@pytest.mark.parametrize("rate,band", [(149.99, 0), (150, 1), (300, 1), (300.01, 2)])
def test_arrival_edges(rate, band):
    assert arrival_band(rate) == band


def test_request_trace_mean_and_pairing():
    arrivals, service = request_trace(np.full(11, 100.), seed=12)
    assert len(arrivals)/660 == pytest.approx(100, rel=.02)
    assert service.mean() == pytest.approx(.01, rel=.02)
    for a, b in zip((arrivals, service), request_trace(np.full(11, 100.), seed=12)):
        np.testing.assert_array_equal(a, b)


def test_no_future_leakage_in_initial_state_or_forecast():
    a = CloudEnvironment([90., 100., 200.], trace=([-2., 1., 61.], [.01]*3))
    b = CloudEnvironment([90., 700., 900.], trace=([-2., 1., 61.], [.01]*3))
    sa, ia = a.reset(seed=1)
    sb, ib = b.reset(seed=1)
    assert sa == sb and ia["forecast_next"] == ib["forecast_next"]
    before = ia["forecast_next"]
    _, _, _, _, info = a.step(1)
    assert info["forecast_used"] == before
    assert info["forecast_next"] == pytest.approx(.3*info["arrival_rate"]+.7*before)

