# Executed validation

Validation completed on September 6, 2026, using the project-local Python 3.12.3 environment. The global environment had a NumPy/Numba version conflict; validation used a newly created virtual environment instead. No global package state is required by the project.

Working directory: `D:\Projects\Adaptive Cloud\Adaptive_Cloud_RL`.

Exact successful setup and validation commands:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe train_all.py --quick
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe compare_results.py
.\.venv\Scripts\python.exe validate_artifacts.py
```

The coding-agent sandbox required additional filesystem access for virtual-environment creation, compiled caches and experiment output. This was an execution-environment permission constraint; the normal user-terminal commands are the same as above.

Observed results:

| Check | Result |
|---|---|
| Fresh dependency installation | Installed only `requirements.txt` and its dependencies |
| Dependency consistency | `No broken requirements found.` |
| First clean-environment test run | 47 passed in 3.01 s |
| Final test run | 47 passed in 2.60 s |
| Quick configuration | Five training seeds, 20 episodes ×24 steps, all five learned policies |
| Frozen evaluation | Three held-out 24-step traces per trained policy and baseline |
| Training + evaluation time | 113.909 s; JIT warm-up separately 0.430 s; plotting excluded |
| Updated Q-tables | 25 |
| Comparison seed rows | 30, including five paired labels for one deterministic baseline run |
| Generated figures | 22, including the comparison-table image |
| Request-log audit | 78 files, 26,387,634 records, three held-out traces |
| Numeric checks | Finite nonempty datasets, complete outcome partitions |
| Event-log checks | FIFO starts, service durations, deadlines, outcome codes, common request inputs |
| Visual inspection | Comparison table, combined learning curve and predictive-policy timeline inspected |

All artifacts are under `results/quick`. The report at the project root and the copy inside the results directory contain actual quick-run values. `RESULTS.md`, `comparison.csv`, `seed_metrics.csv` and `trace_metrics.csv` preserve the numerical evidence; `validation.json` and `request_log_validation.json` preserve audit summaries.

Minor repeat-run directory handling and report wording were improved after the training run. The manifest retains the source hashes captured at training start rather than rewriting historical provenance. Simulator, algorithm and evaluation calculations were unchanged after that run. The comparison command regenerated figures and report text from the saved metrics after the wording changes.

The full 1,000-episode experiment was **not** run. No stabilization episode can be observed under the quick budget, and none of the evaluated policies met the pooled 200 ms p95 target. Results are a functioning preliminary experiment, not evidence of convergence or production readiness. Student names and final institutional pagination remain to be completed.
