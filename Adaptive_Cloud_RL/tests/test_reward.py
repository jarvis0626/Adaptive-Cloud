import pytest
from environment.cloud_environment import compute_reward


@pytest.mark.parametrize("cost,expected", [(120, 1.43), (240, 1.03)])
def test_component_one_worked_examples(cost, expected):
    reward, c = compute_reward(98, 2, 0, 0, cost, 300, False)
    assert reward == pytest.approx(expected)
    assert c["g"]+c["v"] == 1


def test_overload_and_scaling_example():
    assert compute_reward(70, 30, 0, 0, 120, 3600, False)[0] == pytest.approx(-.8)
    assert compute_reward(70, 30, 0, 0, 120, 3600, True)[0] == pytest.approx(-1.)


def test_rejections_and_timeouts_are_resolved_failures():
    reward, terms = compute_reward(80, 10, 5, 5, 60, 0, False)
    assert terms["v"] == .2
    assert reward == pytest.approx(.6)


def test_zero_resolved():
    reward, terms = compute_reward(0, 0, 0, 0, 60, 0, False)
    assert terms["g"] == terms["v"] == 0
    assert reward == -.2

