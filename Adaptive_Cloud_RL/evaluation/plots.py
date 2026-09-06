"""Headless, 300 DPI charts built exclusively from saved numeric results."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .evaluate import slug

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def moving_average(values, window):
    return np.convolve(values, np.ones(window)/window, mode="valid")


def generate_plots(output, summary, mode):
    output = Path(output)
    target = output/"plots"
    target.mkdir(exist_ok=True)
    curves = read_csv(output/"learning_curves.csv")
    names = list(dict.fromkeys(r["algorithm"] for r in curves))
    combined, combined_ax = plt.subplots(figsize=(10, 5))
    for name in names:
        group = [r for r in curves if r["algorithm"] == name]
        seeds = list(dict.fromkeys(r["seed"] for r in group))
        values = np.array([[float(r["mean_reward"]) for r in group if r["seed"] == seed] for seed in seeds])
        x = np.arange(1, values.shape[1]+1)
        fig, ax = plt.subplots(figsize=(8, 4))
        for seed, ys in zip(seeds, values):
            ax.plot(x, ys, alpha=.6, linewidth=.9, label=f"Seed {seed}")
        ax.set(xlabel="Training episode", ylabel="Mean reward / control step", title=f"{name}: raw training reward ({mode})")
        ax.legend(ncol=3)
        save(fig, target/f"reward_{slug(name)}.png")
        window = min(values.shape[1], 50, max(2, values.shape[1]//5))
        smooth = np.array([moving_average(v, window) for v in values])
        sx = x[window-1:]
        mean, std = smooth.mean(axis=0), smooth.std(axis=0, ddof=1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sx, mean, label=f"{window}-episode moving average")
        ax.fill_between(sx, mean-std, mean+std, alpha=.2, label="Across-seed ±1 SD")
        ax.set(xlabel="Training episode", ylabel="Mean reward / control step", title=f"{name}: learning curve ({mode})")
        ax.legend()
        save(fig, target/f"moving_average_{slug(name)}.png")
        combined_ax.plot(sx, mean, label=name)
        combined_ax.fill_between(sx, mean-std, mean+std, alpha=.1)
    combined_ax.set(xlabel="Training episode", ylabel="Moving-average reward / step", title=f"Combined learning curves ({mode}); bands = ±1 SD")
    combined_ax.legend(ncol=2)
    save(combined, target/"combined_learning_curves.png")
    specs = [("mean_reward", "Mean reward / control step", "Frozen-policy reward", "reward_comparison.png"),
             ("sla_violation_fraction", "Fraction of arrival cohort", "SLA violations (including drain)", "sla_comparison.png"),
             ("instance_seconds", "Active + starting instance-seconds", "Resource cost (control window)", "resource_cost_comparison.png"),
             ("training_seconds", "Training wall time (seconds)", "Training runtime (JIT warm-up excluded)", "training_time_comparison.png")]
    labels = [r["algorithm"].replace(" ", "\n") for r in summary]
    for key, ylabel, title, filename in specs:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, [r[key+"_mean"] for r in summary], yerr=[r[key+"_std"] for r in summary], capsize=5,
               color=plt.cm.tab10(np.arange(len(summary))))
        ax.set(ylabel=ylabel, title=f"{title} ({mode}); error bars = ±1 SD")
        save(fig, target/filename)
    # First configured training seed is selected in advance, never best-seed selection.
    for name in [r["algorithm"] for r in summary]:
        seed = next(r["seed"] for r in read_csv(output/"seed_metrics.csv") if r["algorithm"] == name)
        timeline = read_csv(output/"timelines"/f"{slug(name)}_seed{seed}.csv")
        traces = list(dict.fromkeys(r["trace"] for r in timeline))
        fig, axes = plt.subplots(3, len(traces), figsize=(15, 8), squeeze=False)
        for col, trace in enumerate(traces):
            group = [r for r in timeline if r["trace"] == trace]
            times = [float(r["time_seconds"])/60-1 for r in group]
            axes[0, col].step(times, [float(r["arrival_rate"]) for r in group], where="post", label="Measured demand")
            axes[0, col].plot(times, [float(r["forecast_used"]) for r in group], linestyle="--", label="Causal forecast")
            axes[0, col].set(title=trace, ylabel="Requests / second")
            axes[1, col].step(times, [int(r["active_instances"]) for r in group], where="post", label="Active")
            axes[1, col].step(times, [int(r["active_instances"])+int(r["pending_startup"]) for r in group], where="post", linestyle="--", label="Active + starting")
            axes[1, col].set(ylabel="Instances", yticks=range(1, 6), ylim=(.8, 5.2))
            axes[2, col].plot([t+1 for t in times], [int(r["queue_length"]) for r in group], label="Boundary queue")
            axes[2, col].set(ylabel="Waiting requests", xlabel="Simulated time (minutes)")
        axes[0, 0].legend(fontsize=8)
        axes[1, 0].legend(fontsize=8)
        fig.suptitle(f"{name}: held-out timelines, seed {seed} ({mode})")
        save(fig, target/f"timeline_{slug(name)}.png")
    columns = [("mean_reward", "Reward / step"), ("instance_seconds", "Instance-seconds"),
               ("sla_violation_fraction", "SLA fraction"), ("p95_response_time_ms", "p95 (ms)"),
               ("training_seconds", "Training (s)"), ("scaling_actions", "Scaling actions")]
    cells = [[r["algorithm"]]+[f"{r[k+'_mean']:.3f} ± {r[k+'_std']:.3f}" if r[k+'_mean'] is not None else "Not observed"
                              for k, _ in columns] for r in summary]
    fig, ax = plt.subplots(figsize=(17, 3.5))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=["Policy"]+[label for _, label in columns], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax.set_title(f"Frozen-policy comparison: mean ± sample SD across training seeds ({mode})", pad=22)
    save(fig, output/"comparison.png")
