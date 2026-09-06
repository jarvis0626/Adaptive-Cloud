"""Hidden rate schedules and paired, exact piecewise-Poisson request traces."""
import numpy as np


def training_rates(rng, steps, config):
    rates = []
    while len(rates) < steps + 1:
        length = int(rng.integers(*config["segment_steps"]))
        kind = int(rng.integers(5))
        if kind < 2:
            segment = np.full(length, rng.uniform(*config["low_range" if kind == 0 else "medium_range"]))
        elif kind == 2:
            segment = np.linspace(*config["ramp_range"], length)
            if rng.random() < .5:
                segment = segment[::-1]
        elif kind == 3:
            lo, hi = config["peak_range"]
            segment = lo + (hi - lo) * np.sin(np.linspace(0, 2 * np.pi, length)) ** 2
        else:
            segment = np.full(length, rng.uniform(*config["low_range"]))
            start = int(rng.integers(1, max(2, length // 2)))
            segment[start:start + max(2, length // 3)] = rng.uniform(*config["burst_range"])
        rates.extend(segment)
    return np.asarray(rates[:steps + 1], dtype=np.float64)


def evaluation_rates(pattern, steps):
    """Fixed held-out shapes; first entry is a 60-second measurement warm-up."""
    x = np.linspace(0, 1, steps)
    if pattern == "ramp":
        values = np.interp(x, [0, .15, .5, .65, 1], [90, 90, 440, 440, 110])
    elif pattern == "recurring":
        values = 110 + 345 * np.maximum(0, np.sin(6 * np.pi * x - .7)) ** 2
    elif pattern == "burst":
        values = np.full(steps, 115.0)
        values[(x >= .25) & (x < .43)] = 680
        values[(x >= .65) & (x < .82)] = 370
    else:
        raise ValueError(f"Unknown evaluation pattern: {pattern}")
    return np.r_[values[0], values]


def request_trace(rates, seed, interval=60., mean_service_time=.01):
    """Conditional Poisson counts + sorted uniforms give exact arrival timestamps.

    Separate RNG streams keep each request's service requirement identical for
    every policy, regardless of rejection, timeout or server assignment.
    """
    arrival_seed, service_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(arrival_seed)
    counts = rng.poisson(np.asarray(rates) * interval)
    batches = [np.sort(rng.uniform(0, interval, int(n))) + (i - 1) * interval
               for i, n in enumerate(counts)]
    arrivals = np.concatenate(batches)
    service = np.random.default_rng(service_seed).exponential(mean_service_time, len(arrivals))
    return arrivals, service

