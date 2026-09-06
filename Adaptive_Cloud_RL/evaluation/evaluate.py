"""No learning or exploratory selection is permitted in this module."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from baseline import ThresholdAutoscaler
from environment import CloudEnvironment
from .metrics import cohort_metrics, write_csv


def slug(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def evaluate_policy(policy, training_seed, traces, config, output):
    baseline = isinstance(policy, ThresholdAutoscaler)
    before_q = None if baseline else policy.q.copy()
    before_rng = None if baseline else deepcopy(policy.rng.bit_generator.state)
    before_updates = None if baseline else policy.updates
    episode_rows, all_timeline, combined = [], [], []
    output = Path(output)
    for trace in traces:
        env = CloudEnvironment(trace["rates"], **config["environment"], beta=config["innovation"]["beta"])
        state, info = env.reset(seed=trace["seed"])
        if baseline:
            policy.reset()
        timeline = []
        for step in range(env.steps):
            action = policy.select(info) if baseline else policy.select(policy.encode(state, info), info["action_mask"], epsilon=0.)
            assert info["action_mask"][action]
            state, reward, terminated, truncated, info = env.step(action)
            timeline.append(dict(algorithm=policy.name, seed=training_seed, trace=trace["name"],
                                 step=step, time_seconds=info["time_seconds"], arrival_rate=info["arrival_rate"],
                                 active_instances=info["interval_active_instances"], pending_startup=info["interval_pending_startup"],
                                 queue_length=info["queue_length"], action=action, reward=reward,
                                 instance_seconds=info["instance_seconds"], utilization=info["utilization"],
                                 sla_violation_fraction=info["sla_violation_fraction"],
                                 forecast_used=info["forecast_used"], forecast_error=info["forecast_error"],
                                 timely_completions=info["timely_completions"], late_completions=info["late_completions"],
                                 rejections=info["rejections"], timeouts=info["timeouts"]))
        drain = env.drain()
        log = env.request_log()
        # Attribute failures to arrival during a startup interval, even if
        # they resolve later. This is an association, not a causal estimate.
        arrival_steps = np.minimum((log["arrival_seconds"] / env.interval).astype(int), env.steps-1)
        startup = np.array([r["pending_startup"] for r in timeline], dtype=bool)[arrival_steps]
        log["arrived_during_startup"] = startup
        failure = log["outcome"] != 1
        metrics = cohort_metrics(log)
        metrics.update(algorithm=policy.name, seed=training_seed, trace=trace["name"],
                       mean_reward=float(np.mean([r["reward"] for r in timeline])),
                       instance_seconds=sum(r["instance_seconds"] for r in timeline),
                       drain_instance_seconds=drain["instance_seconds"],
                       scaling_actions=sum(r["action"] != 1 for r in timeline),
                       forecast_mae_rps=float(np.mean([abs(r["forecast_error"]) for r in timeline])),
                       forecast_rmse_rps=float(np.sqrt(np.mean([r["forecast_error"]**2 for r in timeline]))),
                       startup_arrivals=int(startup.sum()), startup_violations=int((startup & failure).sum()),
                       startup_violation_rate=float((startup & failure).sum()/startup.sum()) if startup.any() else None,
                       startup_violation_cohort_fraction=float((startup & failure).mean()))
        episode_rows.append(metrics)
        all_timeline.extend(timeline)
        combined.append(log)
        if config["evaluation"]["save_request_logs"]:
            log_dir = output / "request_logs"
            log_dir.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(log_dir / f"{slug(policy.name)}_seed{training_seed}_{trace['name']}.npz", **log)
    pooled = {key: np.concatenate([log[key] for log in combined]) for key in combined[0]}
    result = dict(algorithm=policy.name, seed=training_seed)
    result.update(cohort_metrics(pooled))
    result.update(mean_reward=float(np.mean([r["reward"] for r in all_timeline])),
                  instance_seconds=sum(r["instance_seconds"] for r in episode_rows),
                  drain_instance_seconds=sum(r["drain_instance_seconds"] for r in episode_rows),
                  scaling_actions=sum(r["scaling_actions"] for r in episode_rows),
                  forecast_mae_rps=float(np.mean([abs(r["forecast_error"]) for r in all_timeline])),
                  forecast_rmse_rps=float(np.sqrt(np.mean([r["forecast_error"]**2 for r in all_timeline]))),
                  startup_arrivals=int(pooled["arrived_during_startup"].sum()),
                  startup_violations=int(((pooled["outcome"] != 1) & pooled["arrived_during_startup"]).sum()))
    result["startup_violation_rate"] = result["startup_violations"]/result["startup_arrivals"] if result["startup_arrivals"] else None
    result["startup_violation_cohort_fraction"] = result["startup_violations"]/result["arrivals"]
    if not baseline:
        assert np.array_equal(policy.q, before_q), "Evaluation changed Q-table"
        assert policy.rng.bit_generator.state == before_rng, "Evaluation used exploration RNG"
        assert policy.updates == before_updates, "Evaluation performed learning"
    write_csv(output / "timelines" / f"{slug(policy.name)}_seed{training_seed}.csv", all_timeline)
    return result, episode_rows
