import csv
import json
from pathlib import Path
import numpy as np


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty dataset: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def stabilization_episode(rewards, window=50, tolerance=.1):
    """Earliest adjacent full windows within tolerance, sustained to the end.

    Absolute mean-reward difference <= tolerance and SD <= .5 in both
    windows. Diagnostic only; None is censored/not observed, never zero.
    """
    rewards = np.asarray(rewards)
    stable = []
    for stop in range(2*window, len(rewards)+1):
        previous, current = rewards[stop-2*window:stop-window], rewards[stop-window:stop]
        stable.append(abs(previous.mean()-current.mean()) <= tolerance
                      and max(previous.std(), current.std()) <= .5)
    for i in range(len(stable)):
        if all(stable[i:]):
            return i+2*window
    return None


def cohort_metrics(log):
    outcome = log["outcome"]
    if not len(outcome) or np.any(outcome == 0):
        raise ValueError("Cohort must be nonempty and fully resolved")
    completed = (outcome == 1) | (outcome == 2)
    response = (log["resolution_seconds"][completed]-log["arrival_seconds"][completed])*1000
    return dict(arrivals=len(outcome), timely_completions=int((outcome == 1).sum()),
                late_completions=int((outcome == 2).sum()), rejections=int((outcome == 3).sum()),
                timeouts=int((outcome == 4).sum()), sla_violation_fraction=float((outcome != 1).mean()),
                rejection_rate=float((outcome == 3).mean()), timeout_rate=float((outcome == 4).mean()),
                p95_response_time_ms=float(np.percentile(response, 95)) if len(response) else None)


def summarize_seed_rows(rows):
    summary = []
    for algorithm in dict.fromkeys(row["algorithm"] for row in rows):
        group = [r for r in rows if r["algorithm"] == algorithm]
        result = dict(algorithm=algorithm, seeds=len(group))
        for key in group[0]:
            if key in ("algorithm", "seed"):
                continue
            values = [r[key] for r in group if r[key] is not None]
            result[key+"_n"] = len(values)
            result[key+"_mean"] = float(np.mean(values)) if values else None
            result[key+"_std"] = float(np.std(values, ddof=1)) if len(values)>1 else (0. if values else None)
        rewards = [r["mean_reward"] for r in group]
        result["reward_variance_across_seeds"] = float(np.var(rewards, ddof=1)) if len(rewards)>1 else 0.
        summary.append(result)
    return summary

