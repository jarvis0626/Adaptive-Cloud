"""Regenerate tables, figures and evidence-backed report excerpts."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluation.metrics import summarize_seed_rows, write_csv, write_json
from evaluation.plots import generate_plots, read_csv

ROOT = Path(__file__).resolve().parent


def validate_results(output):
    rows = json.loads((output/"seed_metrics.json").read_text(encoding="utf-8"))
    if len(rows) != 30:
        config = json.loads((output/"config.json").read_text())
        assert len(rows) == 6*len(config["training"]["seeds"])
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                assert np.isfinite(value), (key, value)
        assert row["arrivals"] > 0
        assert sum(row[k] for k in ["timely_completions", "late_completions", "rejections", "timeouts"]) == row["arrivals"]
        assert 0 <= row["sla_violation_fraction"] <= 1
    for filename in ["learning_curves.csv", "trace_metrics.csv", "training_metrics.csv"]:
        table = read_csv(output/filename)
        assert table, filename
        for row in table:
            for value in row.values():
                try:
                    number = float(value)
                except ValueError:
                    continue
                assert np.isfinite(number), (filename, value)
    qtables = list((output/"q_tables").glob("*.npy"))
    assert len(qtables) == len([r for r in rows if r["algorithm"] != "Threshold baseline"])
    for path in qtables:
        values = np.load(path)
        assert np.isfinite(values).all() and np.any(values), path
    return rows


def compare(output):
    output = Path(output)
    rows = validate_results(output)
    summary = summarize_seed_rows(rows)
    write_csv(output/"comparison.csv", summary)
    manifest = json.loads((output/"manifest.json").read_text())
    mode = manifest["mode"]
    generate_plots(output, summary, mode)
    standard = {r["seed"]: r for r in rows if r["algorithm"] == "Q-Learning"}
    predictive = {r["seed"]: r for r in rows if r["algorithm"] == "Predictive Q-Learning"}
    deltas = [dict(seed=seed, **{key: predictive[seed][key]-standard[seed][key]
                  for key in ["mean_reward", "instance_seconds", "sla_violation_fraction", "startup_violation_cohort_fraction", "scaling_actions"]})
              for seed in standard]
    write_csv(output/"innovation_paired_differences.csv", deltas)
    lines = [f"# Actual generated results: {mode}", "",
             "Values below come from frozen policies on the saved held-out traces. Error terms are sample standard deviations across five training seeds. The baseline is deterministic on these traces and replicated for pairing; its zero SD does not estimate workload uncertainty.", "",
             "| Policy | Reward/step | Instance-seconds | SLA violation fraction | p95 completed response (ms) | Training (s) |", "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        keys = ["mean_reward", "instance_seconds", "sla_violation_fraction", "p95_response_time_ms", "training_seconds"]
        lines.append("| "+r["algorithm"]+" | "+" | ".join(f"{r[k+'_mean']:.4f} ± {r[k+'_std']:.4f}" if r[k+'_mean'] is not None else "Not observed" for k in keys)+" |")
    means = {key: float(np.mean([r[key] for r in deltas])) for key in deltas[0] if key != "seed"}
    s = next(r for r in summary if r["algorithm"] == "Predictive Q-Learning")
    lines.extend(["", "## Predictive experiment", "",
                  f"Predictive minus standard Q-learning, paired mean differences: reward {means['mean_reward']:+.4f}; instance-seconds {means['instance_seconds']:+.1f}; overall SLA violation fraction {means['sla_violation_fraction']:+.6f}; startup-arrival violation fraction of the complete cohort {means['startup_violation_cohort_fraction']:+.6f}.", "",
                  f"Forecast MAE: {s['forecast_mae_rps_mean']:.3f} req/s; RMSE: {s['forecast_rmse_rps_mean']:.3f} req/s. Forecast errors are identical across policies because the predictor uses exogenous measured arrivals only.", "",
                  "Startup-associated violations refer to requests arriving while new capacity is unavailable, followed to resolution. This association does not isolate the causal effect of the delay. Consult both conditional startup violation rate and startup exposure counts in the CSV, since policies request different amounts of startup.", "",
                  "## Interpretation", "",
                  "Use reward together with cost, SLA failures and completed-request latency. High reward alone does not establish deployment suitability. The above differences are descriptive, without a statistical significance claim.", "",
                  "Stabilization uses two adjacent 50-episode windows and requires persistent mean changes at most 0.10 reward units and within-window SD at most 0.5. Missing values mean not observed within the budget; a 20-episode quick run cannot satisfy this criterion.", "",
                  "Full-budget results: [PENDING — run python train_all.py and use that output directory.]" if mode.startswith("quick") else "These are full configured-budget results; assess limitations and seed coverage before drawing broad conclusions."])
    content = "\n".join(lines)+"\n"
    (output/"RESULTS.md").write_text(content, encoding="utf-8")
    template = ROOT/"docs"/"report_template.md"
    if template.exists():
        report = template.read_text(encoding="utf-8").replace("{{ACTUAL_RESULTS}}", content)
        (output/"TECHNICAL_REPORT_CONTENT.md").write_text(report, encoding="utf-8")
        (ROOT/"TECHNICAL_REPORT_CONTENT.md").write_text(report, encoding="utf-8")
    write_json(output/"validation.json", dict(status="passed", seed_rows=len(rows),
               q_tables=len(list((output/"q_tables").glob("*.npy"))),
               figures=len(list((output/"plots").glob("*.png")))+1,
               checks=["finite numeric datasets", "nonempty tables", "cohort conservation", "all Q-tables updated",
                       "frozen evaluation Q and RNG assertions executed during evaluation"]))
    print(f"Validated {len(rows)} seed rows; generated comparison table and plots.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    latest = ROOT/"results"/"latest.txt"
    output = args.results or (Path(latest.read_text(encoding="utf-8").strip()) if latest.exists() else ROOT/"results"/"quick")
    compare(output)


if __name__ == "__main__":
    main()
