# Adaptive Cloud Server Resource Allocation Using Reinforcement Learning

Component 2 implements the supplied Component 1 case study and presentation. A centralized controller chooses how many identical cloud instances to run, balancing timely service against billed capacity. This is a local simulator; no cloud account is required.

## Run it on Windows

Open a PowerShell terminal in your IDE and run these commands once:

```powershell
cd "D:\Projects\Adaptive Cloud\Adaptive_Cloud_RL"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the preliminary experiment:

```powershell
.\.venv\Scripts\python.exe train_all.py --quick
```

Open `results/quick/RESULTS.md` for the comparison and `results/quick/plots/` for the graphs. The terminal prints one result per algorithm and training seed. The initial compiled-kernel warm-up may take a little time before the first result appears. There is no web interface or browser server to start.

Other commands, using the same environment:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe compare_results.py
.\.venv\Scripts\python.exe validate_artifacts.py
.\.venv\Scripts\python.exe train_all.py
```

The last command runs the full 1,000-episode budget and is substantially slower. Existing experiments are protected against accidental overwrite: rerunning the default command automatically chooses a timestamped directory. You can also choose an explicit new output directory:

```powershell
.\.venv\Scripts\python.exe train_all.py --quick --output results/quick_repeat
```

If you activate the environment (`.\.venv\Scripts\Activate.ps1`), the brief's short commands work directly:

```text
python train_all.py --quick
python train_all.py
python compare_results.py
pytest
```

Activation is optional; the explicit interpreter commands above work without changing PowerShell execution policy. Python 3.12 is the validated target. On Linux/macOS, create the environment with `python3 -m venv .venv`, install with `.venv/bin/python -m pip install -r requirements.txt`, then use `.venv/bin/python` for the commands above. NumPy and Numba are pinned together; installing an unrelated newer NumPy into this environment can break their compatibility.

## Environment formulation

`CloudEnvironment` implements:

```python
from environment import CloudEnvironment

env = CloudEnvironment([100., 100., 250., 400.])  # warm-up rate + 3 control intervals
state, info = env.reset(seed=42)
next_state, reward, terminated, truncated, info = env.step(1)
```

Decisions occur every 60 simulated seconds. One to five instances each serve one request at a time, with independent exponential service requirements of mean 10 ms (100 requests/s mean capacity). Arrivals are Poisson with a piecewise-constant hidden rate. Conditional Poisson counts and sorted uniform arrival times produce an exact continuous-time arrival process within each interval. Service requirements are sampled per arrival, independently of the chosen policy. The FIFO waiting queue holds at most 6,000 requests, excluding requests already in service. Requests expire 10 seconds after arrival, including queueing and service; unfinished work is cancelled on timeout. There is no discretization of response time into control intervals.

A Numba-compiled event loop processes contiguous numeric arrays and a bounded ring queue, avoiding a Python object for each request. It retains exact timestamps and each request's outcome. This is still an event simulation, not an aggregate throughput estimate. Full experiments process billions of events; runtime depends on CPU and workload. Training has no per-request disk logging. Evaluation saves compressed request records and can consume considerable disk space at full scale.

### State and masks

The observation is `(arrival_band, queue_band, active_instances, pending_startup, demand_trend)`:

| Variable | Encoding |
|---|---|
| Arrival band | 0: <150 req/s; 1: 150–300 inclusive; 2: >300 |
| Queue band | 0: empty; 1: 1–500; 2: >500 |
| Active instances | Integer 1–5 |
| Pending startup | 0 or 1 |
| Demand trend | 0: change <−10%; 1: −10% through +10%; 2: >+10% |

Trend divides the measured change by `max(previous_rate, 1)`. Reset first runs one unscored 60-second measurement interval at the configured initial two instances, retaining its queue and in-flight requests. The initial trend is flat and the forecast is initialized from that measured arrival rate. Agents never read future rates.

`state_to_index` and `index_to_state` provide reversible mixed-radix encoding. There are 270 nominal states and 810 action values. States containing five active instances plus pending startup are unreachable (27 combinations, leaving at most 243 feasible standard states). In the forecast extension the nominal count is 810 states, with 81 capacity-impossible combinations. Additional combinations may be unvisited on particular workloads. The action mask includes idle-instance availability, which is not reconstructible from the binned observation alone; it must be passed with each transition.

| Action | Meaning | Validity |
|---|---|---|
| 0 | Remove one idle instance | Active >1, no startup pending, idle server exists |
| 1 | Maintain | Always |
| 2 | Request one additional instance | No startup pending, active + pending <5 |

Invalid actions raise an error. Scale-in removes only an idle server and never cancels an in-flight request. Every learner and the baseline use the same mask. Masks are applied in exploration, greedy selection, SARSA next-action selection, Expected SARSA probabilities, Q-learning maximization and Dyna planning.

### Exact event order

1. Process elapsed-interval resolutions through the boundary and expose the observation **before** pending activation.
2. Validate the action. If pending startup is reported, only maintain is valid; activate that instance immediately before this interval's service begins.
3. Otherwise apply a valid scale-in immediately, or mark a new instance as starting.
4. Process arrivals in `[t, t+60)` and service/timeout resolutions through `t+60`. A newly requested instance supplies no service during this interval but accrues cost.
5. Return the old active count plus pending=1 when a startup was requested. At the following call, maintain activates it, making it usable in interval `t+1`.

Equal-time service resolutions precede waiting timeouts, which precede arrivals. A completion exactly on its deadline counts as completed; expired queued requests never begin service. FIFO dispatch occurs whenever an enabled server becomes idle. Scheduled activation is not a second scaling action. The finite episode ends with `terminated=False, truncated=True`; all algorithms still bootstrap from its final observation.

### Reward

```text
reward = 2g - c - 4v - q - 0.2m
```

`g` is timely completions (response ≤200 ms) divided by all interval resolutions. `v` is late completions + rejections + timeouts divided by those resolutions. With no resolutions, both are zero. `c` is active-plus-starting instance-seconds divided by `5 × 60`; `q` is `min(waiting_queue/6000, 1)`; `m` indicates an applied scale-in/out. For 98 timely and 2 late completions, 120 instance-seconds and queue length 300, reward is `1.43`; switching subtracts `0.20`. These are arithmetic examples, not experimental outcomes.

`info` provides outcome counts, queue, active/pending counts, action mask, measured utilization, instance-seconds, interval SLA fraction, reward components, measured arrival rate and causal forecast. `p95_response_time_ms` is the exact completed-response percentile for the interval, or `None` when no completions occur. Training disables this unused percentile calculation (`detailed=False`); evaluation enables it. `interval_active_instances` and `interval_pending_startup` explicitly describe capacity used in the elapsed interval.

## Compared policies

| Method | Learning target or decision rule |
|---|---|
| Q-Learning | `r + gamma * max Q(s', valid actions)` |
| SARSA | `r + gamma * Q(s', a')`, carrying the sampled action into the next interaction |
| Expected SARSA | `r + gamma * sum pi(a'|s') Q(s',a')` over the masked epsilon-greedy policy |
| Dyna-Q | Real Q-learning update plus 10 configurable planning updates per interaction |
| Threshold baseline | Measured utilization hysteresis with cooldown and low-load stabilization |
| Predictive Q-Learning | Q-learning with a causal forecast band appended to the observation |

Greedy ties prefer maintain, then scale-out, then scale-in. Exploration is uniform over valid actions. Expected SARSA assigns epsilon uniformly over valid actions and the remaining probability to that same deterministic greedy action. Dyna-Q stores the last observed reward, successor state, successor mask and true termination flag for each visited `(state, action)`, and samples visited pairs uniformly. Its model is approximate in this stochastic, partially observed task; planning can amplify an unrepresentative last observation. It uses a separate planning RNG.

The baseline scales out above 75% measured busy-time utilization. It scales in below 30% only after two consecutive low-load observations. A one-step cooldown follows either scaling action. These are configurable in `config.yaml`, with the same idle and pending constraints as RL. This follows the idea of metric thresholds and stabilization discussed in [Kubernetes autoscaling documentation](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/); it is not an implementation of Kubernetes HPA. No thresholds are tuned on held-out workloads.

The innovation updates `forecast_next = beta * measured_rate + (1-beta) * previous_forecast`, with beta=0.30. The next decision uses the forecast band at the same 150/300 boundaries. Errors compare the forecast available **before** an interval with its subsequently measured arrival rate. Exponential smoothing may lag ramps and cannot anticipate an unannounced burst. No improvement is assumed.

## Experiment and reproducibility

Defaults: 1,000 episodes ×120 steps, alpha=0.10, gamma=0.95, epsilon 1.00→0.05 linearly over the first 80% of interactions, then 0.05; seeds 11, 22, 33, 44, 55. Quick mode uses 20×24, keeps all five seeds and all algorithms, and evaluates three 24-step traces. Full evaluation uses three 120-step traces. The innovation receives the same training budget as standard Q-learning.

Training randomly combines low and medium traffic, ramps, recurring peaks and bursts using ranges saved in `config.yaml`. Training episode seeds derive from `SeedSequence([training_seed, episode_index, 101])`; child streams separate schedule generation and request generation. Evaluation uses separately defined fixed ramp, recurring-peak and burst schedules with seeds 91001–91003. Their complete rate arrays and request seed scheme are saved. All policies and training seeds face identical evaluation arrivals and per-request service requirements. This isolates training variability; the five seeds are not five independent held-out workload samples.

Evaluation explicitly uses epsilon=0 and never calls update. Runtime assertions check that Q-tables, update counts and policy RNG states remain unchanged. The deterministic baseline is evaluated once and its metrics repeated under the five seed labels for paired comparisons; its zero SD should not be interpreted as uncertainty measured from independent trials.

After the scored window, arrivals stop, any pending capacity activates, and allocation is held for exactly 10 seconds. Every evaluated request must then have completed, been rejected, or timed out. Final failure and latency metrics use only requests arriving at time ≥0, excluding warm-up arrivals. Warm-up work can still affect the scored control intervals. Drain cost is saved separately; primary reward and resource-cost comparisons cover the control window. There are no drain control actions or learning updates.

### Metrics

| Metric | Definition |
|---|---|
| Mean reward | Arithmetic mean across evaluated control steps |
| Instance-seconds | Sum of active-plus-starting billed seconds in the control window; drain saved separately |
| SLA violation fraction | Cohort late + rejected + timed-out / cohort arrivals, after drain |
| p95 response | Exact NumPy linear 95th percentile of all completed cohort response times, pooled across traces per seed; never an average of interval percentiles |
| Stabilization episode | First endpoint with adjacent 50-episode window means within 0.10, both SD≤0.5, maintained through remaining training; missing if not observed |
| Training time | Wall time for training, including trace generation, excluding shared JIT warm-up, evaluation and plotting |
| Reward variance | Sample variance of per-seed held-out mean reward |
| Scaling actions | Number of applied scale-in/out actions during the scored window |
| Rejection/timeout rates | Each outcome count / all cohort arrivals |
| Forecast MAE/RMSE | One-step forecast error in requests/second, measured without future leakage |
| Startup-associated violations | Failed requests that arrived during a starting interval, tracked to resolution; both conditional and whole-cohort fractions are reported |

Startup association is not a causal attribution: different policies create different startup exposure. Report startup arrival counts, conditional failure rates, overall failure and cost together. Missing percentiles or stabilization values use JSON `null` / empty CSV cells, not fabricated zero or NaN. Mean, sample SD and nonmissing count appear in `comparison.csv`, and individual seed values in `seed_metrics.csv`. The quick budget cannot satisfy the 100-episode minimum stabilization criterion.

## Generated files

| File/directory | Contents |
|---|---|
| `config.json`, `workloads.json` | Resolved configuration, actual evaluation rates and seeds |
| `manifest.json` | Mode, Python/package versions, platform, source hashes, timing and completion status |
| `learning_curves.csv` | Every algorithm/seed/episode reward and epsilon |
| `seed_metrics.csv` / `.json` | Individual-seed frozen-policy results |
| `trace_metrics.csv` | Per-trace results, permitting workload-specific analysis |
| `training_metrics.csv` | Runtime, stabilization, update counts and nonzero table entries |
| `comparison.csv`, `comparison.png` | Complete summary CSV and readable main-metric table image |
| `innovation_paired_differences.csv` | Predictive minus standard Q-learning per seed |
| `q_tables/*.npy` | Final tables: 270×3 standard, 810×3 predictive |
| `timelines/*.csv` | Demand, forecast, allocation, queue, action, utilization and reward per step |
| `request_logs/*.npz` | Per-arrival timestamps, service requirement, service start, outcome, resolution, server and startup exposure |
| `plots/*.png` | Individual raw/moving-average curves, combined curve, four comparison charts and six policy timelines, 300 DPI |
| `RESULTS.md` | Actual numerical comparison and preliminary/full status |
| `TECHNICAL_REPORT_CONTENT.md` | Report draft with generated results inserted |
| `validation.json` | Automated output sanity-check result |

Request log outcome codes: 1 timely, 2 late, 3 rejected, 4 timed out. A −1 service-start/server value means the request never began service. Records are ordered by arrival; row number is a stable request identifier within a trace. Queue duration is service start minus arrival for served requests; waiting timeouts resolve at arrival+10. Set `evaluation.save_request_logs: false` to suppress disk logs, while retaining exact cohort metrics. The default saves all evaluation logs; the baseline has only one physical run/log set because it is deterministic.

The report is generated from `docs/report_template.md`, so edit that template to preserve report changes when comparisons are regenerated. `compare_results.py --results results/quick_repeat` selects a specific run. No plotting command retrains or changes policies.

## Validation and limitations

Tests cover manually constructed FIFO/timeout/rejection cases, capacity masks, idle-only removal, startup activation, reward arithmetic, index round trips, masked TD targets, Dyna planning, continuing-task truncation, deterministic replay, smoke training of every learner and frozen evaluation. `compare_results.py` rejects empty or nonfinite numerical datasets, unresolved cohorts and unchanged/invalid Q-tables.

The included quick experiment was executed in a fresh Python 3.12 environment installed only from `requirements.txt`: **47 tests passed**, **25 Q-tables**, **30 seed comparison rows**, and **22 figures** were generated. Training plus evaluation took about 114 seconds on the recorded machine, excluding final plotting. The saved-log audit passed for **26,387,634 request records across 78 files**, including FIFO, deadlines and identical policy inputs. See `docs/RUN_VALIDATION.md` for exact commands. These results are preliminary; none of the policies met the pooled 200 ms p95 target in this run.

The model omits network latency, heterogeneous servers, database bottlenecks, real billing and correlated service times. Binning and hidden request ages make it partially observable; constant-step tabular learning has no optimality guarantee here. The synthetic held-out traces are a limited test suite. Reward weights, baseline thresholds and forecast beta are initial design choices. Quick-run results demonstrate reproducibility and functioning code, not convergence, superiority or production readiness. The 15–20-page report is supplied as a content draft; final pagination, student names and institutional formatting must be completed by the authors.

## References and Component 1 continuity

The supplied 14-page PDF and 16-slide PPT were read before implementation. Their state, constraints, event ordering, request-level assumptions, reward and forecast design are retained. See `docs/COMPONENT_1_ALIGNMENT.md` for the mapping.

- C. J. C. H. Watkins and P. Dayan, “Q-learning,” *Machine Learning*, vol. 8, pp. 279–292, 1992. [doi:10.1007/BF00992698](https://doi.org/10.1007/BF00992698).
- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018. [Publisher](https://mitpress.mit.edu/9780262039246/reinforcement-learning/).
- Kubernetes Authors, “Horizontal Pod Autoscaling.” [Official documentation](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/).
