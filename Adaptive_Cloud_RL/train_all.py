"""Train all methods, evaluate frozen policies, and build reproducible artifacts."""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import yaml
from algorithms import QLearning, SARSA, ExpectedSARSA, DynaQ
from algorithms.common import epsilon_at
from baseline import ThresholdAutoscaler
from environment import CloudEnvironment
from environment.workload_generator import training_rates, evaluation_rates
from innovation import PredictiveQLearning
from evaluation.evaluate import evaluate_policy, slug
from evaluation.metrics import write_csv, write_json, stabilization_episode, summarize_seed_rows

ROOT = Path(__file__).resolve().parent
AGENTS = [QLearning, SARSA, ExpectedSARSA, DynaQ, PredictiveQLearning]


def train_agent(agent_class, seed, config):
    settings = config["training"]
    agent = agent_class(seed=seed, alpha=settings["alpha"], gamma=settings["gamma"],
                        planning_steps=settings["planning_steps"])
    rows = []
    total = settings["episodes"]*settings["steps"]
    clock = perf_counter()
    for episode in range(settings["episodes"]):
        sequence = np.random.SeedSequence([seed, episode, 101])
        rate_seed, request_seed = sequence.spawn(2)
        rates = training_rates(np.random.default_rng(rate_seed), settings["steps"], config["workload"]["training"])
        env = CloudEnvironment(rates, **config["environment"], beta=config["innovation"]["beta"], detailed=False)
        state, info = env.reset(seed=request_seed)
        index = agent.encode(state, info)
        def epsilon(i):
            return epsilon_at(i, total, settings["epsilon_initial"], settings["epsilon_min"], settings["epsilon_decay_fraction"])
        action = agent.select(index, info["action_mask"], epsilon(episode*env.steps))
        rewards = []
        for step in range(env.steps):
            assert info["action_mask"][action]
            state, reward, terminated, truncated, info = env.step(action)
            next_index = agent.encode(state, info)
            next_epsilon = epsilon(episode*env.steps+step+1)
            # SARSA's sampled successor action is carried into the next real
            # interaction. The final truncated transition still bootstraps.
            next_action = agent.select(next_index, info["action_mask"], next_epsilon)
            agent.update(index, action, reward, next_index, info["action_mask"], next_action,
                         next_epsilon, terminated=terminated, truncated=truncated)
            index, action = next_index, next_action
            rewards.append(reward)
        rows.append(dict(algorithm=agent.name, seed=seed, episode=episode+1,
                         mean_reward=float(np.mean(rewards)), epsilon=epsilon((episode+1)*env.steps-1)))
    elapsed = perf_counter()-clock
    if not agent.updates or not np.any(agent.q) or not np.isfinite(agent.q).all():
        raise AssertionError(f"Invalid/untrained Q-table: {agent.name}")
    stable = stabilization_episode([r["mean_reward"] for r in rows],
                                   config["evaluation"]["stabilization_window"],
                                   config["evaluation"]["stabilization_tolerance"])
    return agent, rows, elapsed, stable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="20 episodes, 24 steps, five seeds; preliminary only")
    parser.add_argument("--config", type=Path, default=ROOT/"config.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.quick:
        config["training"].update(episodes=config["quick"]["episodes"], steps=config["quick"]["steps"])
        config["workload"]["evaluation"]["steps"] = config["quick"]["evaluation_steps"]
    output = args.output or ROOT/"results"/("quick" if args.quick else "full")
    output.mkdir(parents=True, exist_ok=True)
    if (output/"manifest.json").exists():
        raise SystemExit(f"Results already exist at {output}; use --output with a new directory to preserve previous runs.")
    settings = config["training"]
    if settings["episodes"] < 1 or settings["steps"] < 1 or len(set(settings["seeds"])) < 5:
        raise ValueError("Use positive training budgets and at least five distinct seeds")
    eval_config = config["workload"]["evaluation"]
    if len(eval_config["patterns"]) != len(eval_config["seeds"]) or not eval_config["patterns"]:
        raise ValueError("Each held-out trace needs a distinct configured pattern/seed entry")
    traces = [dict(name=name, seed=seed, rates=evaluation_rates(name, eval_config["steps"]).tolist())
              for name, seed in zip(eval_config["patterns"], eval_config["seeds"])]
    write_json(output/"config.json", config)
    write_json(output/"workloads.json", dict(training=config["workload"]["training"], evaluation=traces,
                                            training_seed_scheme="SeedSequence([training_seed, episode_index, 101]).spawn(2)"))
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in ROOT.rglob("*.py") if not any(x in p.parts for x in (".venv", "results"))}
    manifest = dict(mode="quick/preliminary" if args.quick else "full", status="running",
                    python=platform.python_version(), platform=platform.platform(),
                    packages={p: importlib.metadata.version(p) for p in ["numpy", "numba", "llvmlite", "matplotlib", "PyYAML", "pytest"]},
                    source_sha256=source_hashes, command="python train_all.py"+(" --quick" if args.quick else ""))
    write_json(output/"manifest.json", manifest)
    # Compile/load the event kernel once before any method's training timer.
    warmup_clock = perf_counter()
    warmup = CloudEnvironment([0., 0.], **config["environment"], detailed=False)
    warmup.reset(seed=0)
    warmup.step(1)
    manifest["kernel_warmup_seconds"] = perf_counter()-warmup_clock
    rows, episodes, curves, training = [], [], [], []
    started = perf_counter()
    for cls in AGENTS:
        for seed in settings["seeds"]:
            agent, learning, seconds, stable = train_agent(cls, seed, config)
            curves.extend(learning)
            qdir = output/"q_tables"
            qdir.mkdir(exist_ok=True)
            np.save(qdir/f"{slug(agent.name)}_seed{seed}.npy", agent.q)
            metrics, trace_rows = evaluate_policy(agent, seed, traces, config, output)
            metrics.update(training_seconds=seconds, stabilization_episode=stable, q_updates=agent.updates,
                           q_nonzero_entries=int(np.count_nonzero(agent.q)))
            rows.append(metrics)
            episodes.extend(trace_rows)
            training.append(dict(algorithm=agent.name, seed=seed, training_seconds=seconds,
                                 stabilization_episode=stable, q_updates=agent.updates,
                                 q_nonzero_entries=int(np.count_nonzero(agent.q))))
            print(f"{agent.name:22} seed={seed:2} train={seconds:6.2f}s eval reward={metrics['mean_reward']:.4f} SLA violations={metrics['sla_violation_fraction']:.3%}", flush=True)
            write_csv(output/"learning_curves.csv", curves)
            write_csv(output/"seed_metrics.csv", rows)
    # The baseline has no training randomness. Evaluate it once on all held-out
    # traces and replicate its identical values explicitly for paired comparison.
    baseline = ThresholdAutoscaler(**config["baseline"])
    base_metrics, base_episodes = evaluate_policy(baseline, settings["seeds"][0], traces, config, output)
    for seed in settings["seeds"]:
        metrics = deepcopy(base_metrics)
        metrics.update(seed=seed, training_seconds=0., stabilization_episode=None, q_updates=0, q_nonzero_entries=0)
        rows.append(metrics)
        for episode in base_episodes:
            episodes.append(dict(episode, seed=seed))
    write_csv(output/"learning_curves.csv", curves)
    write_csv(output/"seed_metrics.csv", rows)
    write_csv(output/"trace_metrics.csv", episodes)
    write_csv(output/"training_metrics.csv", training)
    write_json(output/"seed_metrics.json", rows)
    write_csv(output/"comparison.csv", summarize_seed_rows(rows))
    manifest.update(status="complete", experiment_seconds=perf_counter()-started,
                    baseline_replication="One deterministic evaluation copied across training seed labels; SD=0 is not independent replication.")
    write_json(output/"manifest.json", manifest)
    from compare_results import compare
    compare(output)
    (ROOT/"results").mkdir(exist_ok=True)
    (ROOT/"results"/"latest.txt").write_text(str(output.resolve()), encoding="utf-8")
    print(f"Complete: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

