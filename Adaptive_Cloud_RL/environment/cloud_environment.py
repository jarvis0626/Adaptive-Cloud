"""Cloud service environment matching Component 1's boundary convention."""
import numpy as np
from .simulator import advance
from .workload_generator import request_trace


def arrival_band(rate):
    return 0 if rate < 150 else (1 if rate <= 300 else 2)


def state_to_index(state):
    if len(state) not in (5, 6):
        raise ValueError("State must contain five or six components")
    a, q, n, p, d = state[:5]
    if not (a in range(3) and q in range(3) and n in range(1, 6)
            and p in range(2) and d in range(3)):
        raise ValueError("Invalid state component")
    index = ((((a * 3 + q) * 5 + n - 1) * 2 + p) * 3 + d)
    if len(state) == 6:
        if state[5] not in range(3):
            raise ValueError("Invalid forecast band")
        index = index * 3 + state[5]
    return int(index)


def index_to_state(index, predictive=False):
    if not isinstance(index, (int, np.integer)) or not 0 <= index < (810 if predictive else 270):
        raise ValueError("Index outside Q-table")
    forecast = index % 3
    if predictive:
        index //= 3
    d, index = index % 3, index // 3
    p, index = index % 2, index // 2
    n, index = index % 5 + 1, index // 5
    state = (index // 3, index % 3, n, p, d)
    return state + (forecast,) if predictive else state


def compute_reward(timely, late, rejected, timeouts, instance_seconds,
                   queue_length, changed, interval=60., queue_capacity=6000):
    total = timely + late + rejected + timeouts
    g = timely / total if total else 0.
    v = (late + rejected + timeouts) / total if total else 0.
    components = dict(g=g, c=instance_seconds / (5 * interval), v=v,
                      q=min(queue_length / queue_capacity, 1.), m=int(changed))
    return 2*g - components['c'] - 4*v - components['q'] - .2*changed, components


class CloudEnvironment:
    def __init__(self, rates, interval=60., initial_instances=2, queue_capacity=6000,
                 timeout=10., mean_service_time=.01, sla_seconds=.2,
                 beta=.3, detailed=True, trace=None):
        self.rates = np.asarray(rates, dtype=np.float64)
        if len(self.rates) < 2 or not np.isfinite(self.rates).all() or (self.rates < 0).any():
            raise ValueError("Provide nonnegative rates including warm-up and at least one step")
        if not (1 <= initial_instances <= 5 and queue_capacity > 0 and interval > 0
                and timeout > 0 and mean_service_time > 0 and 0 < beta <= 1 and sla_seconds > 0):
            raise ValueError("Invalid simulator configuration")
        self.interval, self.initial_instances = interval, initial_instances
        self.capacity, self.timeout, self.mean_service_time = queue_capacity, timeout, mean_service_time
        self.sla, self.beta, self.detailed, self.trace = sla_seconds, beta, detailed, trace
        self.steps = len(self.rates) - 1

    def reset(self, seed=None):
        self.arrivals, self.service = (request_trace(self.rates, seed, self.interval, self.mean_service_time)
                                       if self.trace is None else tuple(np.array(x, dtype=float) for x in self.trace))
        if (len(self.arrivals) != len(self.service) or not np.isfinite(self.arrivals).all()
                or not np.isfinite(self.service).all() or (np.diff(self.arrivals) < 0).any()
                or (self.service < 0).any() or (self.arrivals < -self.interval).any()
                or (self.arrivals >= self.steps * self.interval).any()):
            raise ValueError("Invalid request trace")
        count = len(self.arrivals)
        self.starts = np.full(count, -1., dtype=np.float64)
        self.resolved = np.full(count, -1., dtype=np.float64)
        self.outcomes = np.zeros(count, dtype=np.uint8)
        self.server_log = np.full(count, -1, dtype=np.int8)
        self.enabled = np.arange(5) < self.initial_instances
        self.jobs = np.full(5, -1, dtype=np.int64)
        self.finishes = np.full(5, np.inf)
        self.queue = np.empty(self.capacity, dtype=np.int64)
        self.head = self.queue_length = self.cursor = self.pending = self.step_number = 0
        self.time = -self.interval
        warmup = self._advance(0.)
        self.latest_rate = warmup["arrivals"] / self.interval
        self.forecast = self.latest_rate
        self.trend = 1
        self.last_utilization = warmup["utilization"]
        self.drained = False
        return self.state(), self._observation_info()

    @property
    def active_instances(self):
        return int(self.enabled.sum())

    def action_mask(self):
        if self.pending:
            return np.array([False, True, False])
        idle = bool(np.any(self.enabled & (self.jobs < 0)))
        return np.array([self.active_instances > 1 and idle, True, self.active_instances < 5])

    def state(self):
        q = 0 if self.queue_length == 0 else (1 if self.queue_length <= 500 else 2)
        return (arrival_band(self.latest_rate), q, self.active_instances, self.pending, self.trend)

    def _observation_info(self):
        return dict(action_mask=self.action_mask(), arrival_rate=self.latest_rate,
                    forecast_next=self.forecast, utilization=self.last_utilization,
                    active_instances=self.active_instances, pending_startup=self.pending,
                    queue_length=self.queue_length)

    def _advance(self, end, arrival_limit=None):
        before = self.cursor
        duration = end - self.time
        result = advance(self.time, end, end if arrival_limit is None else arrival_limit,
                         self.arrivals, self.service, self.starts, self.resolved, self.outcomes,
                         self.server_log, self.enabled, self.jobs, self.finishes, self.queue,
                         self.head, self.queue_length, self.cursor, self.timeout, self.sla)
        self.head, self.queue_length, self.cursor, counts, latencies, busy = result
        self.time = end
        return dict(zip(["timely_completions", "late_completions", "rejections", "timeouts"], map(int, counts)),
                    arrivals=self.cursor-before,
                    p95_response_time_ms=float(np.percentile(latencies, 95)*1000) if len(latencies) and self.detailed else None,
                    utilization=busy/(self.active_instances*duration) if duration else 0.,
                    instance_seconds=(self.active_instances + self.pending)*duration)

    def step(self, action):
        if self.step_number >= self.steps or self.drained:
            raise RuntimeError("Reset after truncation; use drain() to resolve the cohort")
        if action not in (0, 1, 2) or not self.action_mask()[action]:
            raise ValueError(f"Invalid action {action}; mask={self.action_mask()}")
        if self.pending:
            self.enabled[np.flatnonzero(~self.enabled)[0]] = True
            self.pending = 0
        elif action == 0:
            self.enabled[np.flatnonzero(self.enabled & (self.jobs < 0))[-1]] = False
        elif action == 2:
            self.pending = 1
        interval_active, interval_pending = self.active_instances, self.pending
        forecast_used = self.forecast
        info = self._advance(self.time + self.interval)
        rate = info["arrivals"] / self.interval
        delta = (rate-self.latest_rate) / max(self.latest_rate, 1.)
        self.trend = 0 if delta < -.1 else (2 if delta > .1 else 1)
        self.latest_rate = rate
        self.forecast = self.beta*rate + (1-self.beta)*self.forecast
        self.last_utilization = info["utilization"]
        reward, terms = compute_reward(info["timely_completions"], info["late_completions"],
                                      info["rejections"], info["timeouts"], info["instance_seconds"],
                                      self.queue_length, action != 1, self.interval, self.capacity)
        self.step_number += 1
        info.update(self._observation_info())
        info.update(reward_components=terms, sla_violation_fraction=terms["v"],
                    applied_action=int(action), scaling_action=int(action != 1),
                    interval_active_instances=interval_active, interval_pending_startup=interval_pending,
                    forecast_used=forecast_used, forecast_error=forecast_used-rate,
                    time_seconds=self.time, reward=float(reward))
        return self.state(), float(reward), False, self.step_number >= self.steps, info

    def drain(self):
        """Stop arrivals, hold allocation, activate pending capacity and run 10 s.

        Every cohort request must resolve by horizon + timeout. Drain cost is
        reported separately and excluded from control-window reward/cost.
        """
        if self.step_number != self.steps or self.drained:
            raise RuntimeError("Drain exactly once after the evaluation window")
        if self.pending:
            self.enabled[np.flatnonzero(~self.enabled)[0]] = True
            self.pending = 0
        horizon = self.time
        info = self._advance(horizon + self.timeout, arrival_limit=horizon)
        self.drained = True
        assert self.queue_length == 0 and np.all(self.jobs < 0)
        assert np.all(self.outcomes[self.arrivals >= 0] > 0)
        return info

    def request_log(self):
        cohort = self.arrivals >= 0
        return dict(arrival_seconds=self.arrivals[cohort], service_seconds=self.service[cohort],
                    service_start_seconds=self.starts[cohort], resolution_seconds=self.resolved[cohort],
                    outcome=self.outcomes[cohort], server=self.server_log[cohort])
