# Adaptive Cloud Server Resource Allocation Using Reinforcement Learning

## 1. Title page information

**Reinforcement Learning — CIA–3, Component 2**  
**CHRIST (Deemed to be University)**  
**B.Tech, Computer Science and Engineering**  
**Academic submission: September 2026**

| Field | To be completed by the authors |
|---|---|
| Team member 1 / register number | ______________________________ |
| Team member 2 / register number | ______________________________ |
| Team member 3 / register number | ______________________________ |
| Team member 4 / register number | ______________________________ |
| Faculty / course section | ______________________________ |
| Submission date | ______________________________ |

This is a structured content draft for a 15–20-page report. Final pagination depends on the institution's template, font, spacing, figures and front matter. Suggested allocation: title/front matter 1 page; abstract, keywords and contents 1; introduction and related work 2; methodology and environment 3; algorithms 2; experimental setup 1; measured results and comparative analysis 3–4; innovation 1–2; discussion/conclusion 1–2; references 1. Insert the supplied plots at readable size. Do not pad the report with invented experiments or observations.

## 2. Abstract

Cloud-hosted applications face changing demand while paying for provisioned capacity. Too few instances create queueing delay and failed requests; excessive capacity raises cost. Continuing from Component 1, this project implements a reproducible request-level simulator and compares Q-learning, SARSA, Expected SARSA and Dyna-Q with a threshold autoscaler. A five-component discrete observation summarizes measured arrivals, queue length, active capacity, pending startup and demand trend. All methods share a constrained three-action space, a one-minute control interval, a one-interval startup delay and a normalized service-and-cost reward. A predictive Q-learning extension appends an exponentially smoothed arrival forecast.

The simulator uses continuous request timestamps, a finite FIFO queue, exponential service requirements and an end-of-cohort drain. Numeric arrays and compiled event processing avoid allocating Python request objects. Frozen policies face identical held-out traces and per-request random inputs across five training seeds. Results include service failures, completed-request latency, billed instance-seconds, reward, learning variability and runtime. Section 10 is populated exclusively from executed experiment output and identifies whether the configured budget was quick or full. Preliminary runs demonstrate an operational, testable comparison pipeline; they cannot establish convergence, a universally superior controller or real-cloud deployment readiness.

## 3. Keywords

Reinforcement learning; cloud autoscaling; Q-learning; SARSA; Expected SARSA; Dyna-Q; FIFO queueing; workload forecasting; reproducible simulation.

## 4. Introduction

### 4.1 Problem and motivation

A cloud application accepts requests continuously, but its processing capacity changes only when the resource controller adds or removes instances. The operator faces two competing objectives: maintain responsive service and avoid paying for unnecessary capacity. Under-provisioning creates a waiting queue. Even if extra capacity eventually arrives, requests delayed during startup may already have missed their service target. Over-provisioning avoids some overload but keeps idle instances billed during quiet periods.

This project investigates horizontal scaling of one application across one to five identical instances. It retains Component 1's centralized controller, 200 ms service target and normalized reward. It does not model database capacity, network delay, geographic routing or cloud-provider billing. The restricted scope makes the consequences of scaling visible and allows inspection of a small Q-table.

### 4.2 Research questions

The principal question is whether tabular reinforcement learning can learn a useful service-cost trade-off on unseen synthetic workloads. A second question concerns differences among the four specified temporal-difference approaches under equal real interaction budgets. A third asks whether a short-horizon smoothed forecast reduces startup-associated service failures at acceptable resource cost.

These questions must be answered with evaluation data, not the training reward curve alone. Training rewards are collected under exploration, and Dyna-Q also uses simulated model updates. A controller that trains quickly may evaluate poorly; a controller with higher reward may still violate the service target. Accordingly, the project records separate cost, latency and failure metrics and presents individual seeds.

### 4.3 Contribution and continuity

The work turns the Component 1 formulation into executable code, implements independently defined TD update rules, provides a masked utilization baseline and adds a causal forecasting experiment. Its practical contribution is a reproducible comparison with explicit accounting. Exponential smoothing and Q-learning are established methods; their combination here is an application-level innovation rather than a claim of a new learning algorithm.

The implementation mapping is recorded in `docs/COMPONENT_1_ALIGNMENT.md`. The source PDF and presentation remain available in the repository. Their illustrative capacity and reward examples are retained only as design explanations and unit-test arithmetic.

## 5. Related work

Watkins and Dayan introduced the classical Q-learning formulation and studied its convergence under appropriate conditions [1]. That result motivates an inspectable tabular controller but does not guarantee optimality in this project: the observation omits queue ages and the hidden workload regime, and the implementation uses a constant learning rate.

Sutton and Barto describe temporal-difference control and the relationship between model-free learning and planning [2]. This project uses four distinct successor-value calculations: a greedy maximum, a sampled policy action, an expectation over the policy, and a greedy update supplemented by model-based planning. Keeping their observation and reward machinery identical makes the algorithmic differences easier to interpret.

Kubernetes documents autoscaling based on observed metrics and stabilization behavior [3]. It motivates a credible threshold reference with hysteresis and cooldown. The implemented baseline is a simulator-specific utilization rule, not a reproduction of Kubernetes HPA. It shares every operational mask with the RL policies and receives no deliberately weaker capacity or startup behavior.

The supplied Component 1 study [4] and presentation [5] define this project's scope and assumptions. A separate course assessment file is referenced by Component 1 but was not supplied. This draft follows the explicit Component 2 requirements provided with the task; additional institutional requirements should be checked before submission.

## 6. Methodology

### 6.1 Experimental separation

The workflow separates training, frozen evaluation and artifact generation. Training interactions update action values using randomized rate schedules. Evaluation uses fixed held-out schedules and seeds without exploration or updates. Plotting and report generation read saved results and do not rerun learning. This separation allows a reader to trace every displayed value to a CSV or request log.

A training replicate consists of one algorithm and one training seed. Every algorithm receives the same episode count, control horizon, learning rate, discount and exploration schedule. Dyna-Q additionally receives the explicitly reported planning budget. Standard and predictive Q-learning use equal interaction budgets even though the latter has a larger table.

### 6.2 Randomness and paired comparisons

Training schedule and request seeds are derived from a seed sequence containing the training seed and episode index. Separate streams generate arrivals and service requirements. Evaluation schedules are defined separately and saved in full. A request's service requirement is assigned at arrival generation, so a policy's rejection or dispatch decisions do not alter the service samples later policies receive.

The five training seeds share the same evaluation traces. Variation across those seeds therefore describes training variability under a fixed test suite; it does not estimate uncertainty across all possible traffic distributions. The baseline has no training randomness and is evaluated once, with identical results repeated under seed labels for pairing. Its zero standard deviation is disclosed as a consequence of this design.

### 6.3 Continuing-task learning

The application has no natural terminal state. Episodes are finite training windows, and their final transition is a time-limit truncation. Each TD target retains its successor value unless true termination is explicitly indicated. Reset uses the same warm-up convention for all methods. This avoids treating the last minute of a training window as a real shutdown of the service.

## 7. Environment design

### 7.1 Request process

Arrivals follow a piecewise-constant Poisson rate. For each minute, a Poisson count is drawn and arrival times are sampled as sorted uniforms conditional on that count. Service requirements are independent exponential random variables with mean 0.01 seconds. Each enabled instance processes one request at a time and immediately takes the oldest eligible waiting request when idle.

The waiting queue capacity is 6,000 requests, excluding those in service. Overflow arrivals are rejected immediately. Every admitted request has an absolute deadline of arrival time plus 10 seconds, including both queueing and service. A timeout removes waiting work or cancels remaining service. Each request resolves exactly once as timely completion, late completion, rejection or timeout.

The compiled event loop retains continuous timestamps and does not approximate 200 ms compliance using minute-level utilization. Arrays store arrivals, service requirements, service starts, resolutions, outcome codes and server assignment. The FIFO queue is a bounded ring of request indices. This reduces object overhead while preserving the measurements needed for latency and cohort accounting.

### 7.2 State representation

The observation is `s=(arrival_band, queue_band, active_instances, pending_startup, demand_trend)`.

| Component | Values |
|---|---|
| Arrival band | Low <150; medium 150–300; high >300 requests/second |
| Queue band | Empty 0; short 1–500; long >500 waiting requests |
| Active instances | 1, 2, 3, 4 or 5 |
| Pending startup | 0 or 1 |
| Trend | Falling below −10%; flat from −10% through +10%; rising above +10% |

The nominal table contains `3×3×5×2×3=270` states and three action values per state. Five active instances plus pending startup are impossible; 27 nominal combinations are therefore excluded by the environment. The encoder still reserves those indices to keep decoding simple and reversible. Operational idle availability is exposed through an action mask rather than added to the table dimensions.

Measured trend uses `(latest-previous)/max(previous,1)`. Reset runs an unscored minute at the initial two instances, sets the initial trend flat and retains outstanding warm-up work. Future rates are hidden from policy selection. These conventions produce a useful initial measurement while maintaining the source formulation's partial observability.

### 7.3 Actions and startup convention

Action IDs 0, 1 and 2 mean remove one idle instance, maintain and request one additional instance. Maintain is always valid. Scale-in needs more than one active instance and at least one idle server. Scale-out requires active plus pending capacity below five. While startup is pending, only maintain is allowed.

If scale-out is requested at boundary `t`, the new server is billed but unavailable throughout interval `t`. The observation at the next boundary reports the old active count and pending=1. After the required maintain action, the new server activates before service processing in interval `t+1`. This order preserves both a full-interval delay and a meaningful pending observation. Scheduled activation carries no second switching penalty.

Scale-in only disables an idle server. It does not terminate work. Equal-time event precedence is service resolution, waiting timeout, then arrival. Arrivals belong to the half-open control interval, while resolutions at its ending boundary are processed before the next decision. Completion exactly at its deadline is a completion; an expired waiting request cannot begin service.

### 7.4 Reward and accounting

The interval reward is `r=2g-c-4v-q-0.2m`. Timely completion fraction `g` and failure fraction `v` partition interval resolutions; failures include late completions, rejections and timeouts. With no resolutions both fractions are zero. Cost `c` divides billed active-plus-starting seconds by the maximum `5×60`. Queue penalty `q` is capped at one and uses waiting length/6,000. Switching indicator `m` is one only when an agent-requested scale-in/out is applied.

For the source's illustrative 98 timely and 2 late completions, 120 instance-seconds and queue length 300, the reward is `2(0.98)-0.4-4(0.02)-0.05=1.43`. Raising cost to 240 seconds lowers it to 1.03. A scaling action subtracts another 0.20. Unit tests verify these calculations; they are not experimental results.

Reward fractions use interval resolutions, whereas final failure metrics use arrival cohorts. The distinction matters when requests cross boundaries. Evaluation stops new arrivals after its scored window and drains for 10 seconds at held allocation, activating any pending server. It then checks that all evaluated arrivals have resolved. Drain resource consumption is reported separately from control-window cost.

## 8. Explanation of the four approaches

### 8.1 Shared masked behavior policy

All agents explore uniformly among valid actions with probability epsilon. Otherwise they choose the largest valid action value. Ties prefer maintain, then scale-out, then scale-in. The same convention is used in evaluation and in Expected SARSA's probability calculation. Invalid actions are excluded from both behavior and targets, including the stored next masks used in Dyna planning.

The shared TD operation is `Q(s,a) ← Q(s,a)+alpha[target-Q(s,a)]`. The agents implement their targets separately to make review and arithmetic tests straightforward.

### 8.2 Q-learning

Q-learning uses `target=r+gamma max_valid Q(s',a')`. It updates toward a greedy successor value while behavior can remain exploratory. The method provides the reference table for the innovation comparison. Its compact representation is easy to inspect, but aliased observations can associate the same table row with different hidden queue conditions.

### 8.3 SARSA

SARSA uses `target=r+gamma Q(s',a')` for the next action sampled from the current masked behavior policy. The runner carries that sampled action into the next real interaction, ensuring that the target action is not discarded and independently resampled. Exploration therefore affects its learned target directly.

### 8.4 Expected SARSA

Expected SARSA replaces the single successor action value with `sum_a' pi(a'|s')Q(s',a')`. Each valid action receives epsilon divided by the number of valid actions, and the deterministic greedy action receives the remaining probability. This expectation uses the same successor epsilon as the behavior selector. Tests check the calculation when an invalid action has an artificially large value.

### 8.5 Dyna-Q

Dyna-Q performs a real Q-learning update, stores the observed reward, successor state, successor mask and termination flag, then executes ten planning updates by default. Planning samples previously visited state-action pairs uniformly. The stored model uses the latest observed transition for each pair. This simple model is reproducible and compact, but it is an approximation of a stochastic environment; extra updates can reinforce a noisy or unrepresentative transition. Both update counts and training time must accompany comparisons at equal real-interaction budgets.

### 8.6 Threshold reference

The baseline uses measured fraction of active server time spent busy. It requests scale-out above 0.75 utilization and scale-in below 0.30 after two consecutive low observations. One control-step cooldown follows a change. The same mask can block scale-in if no instance is idle or force maintain during startup. Its thresholds are initial configurable choices rather than values selected to make RL appear better.

## 9. Experimental setup

| Setting | Full default | Quick default |
|---|---|---|
| Episodes per learner/seed | 1,000 | 20 |
| Control steps per episode | 120 | 24 |
| Training seeds | 11, 22, 33, 44, 55 | Same five |
| Alpha / gamma | 0.10 / 0.95 | Same |
| Epsilon | 1.00 to 0.05 over first 80% of interactions | Same schedule scaled to budget |
| Dyna planning steps | 10 | 10 |
| Held-out traces | Ramp, recurring peaks, sudden bursts | Same types |
| Control steps per evaluation trace | 120 | 24 |
| Forecast beta | 0.30 | 0.30 |

Training combines randomized low/medium segments, ramps, periodic peaks and abrupt bursts. Evaluation seeds 91001, 91002 and 91003 generate requests on separately specified fixed schedules. Their actual arrays are stored in `workloads.json`. Initial allocation and warm-up are identical for every method. Policies are frozen: epsilon is zero, update is never called, and assertions compare Q-tables, update counters and random-generator state before and after evaluation.

Runtime is measured with a monotonic wall clock. A shared kernel compilation/load warm-up is excluded from each algorithm's training timer and recorded separately. Training time includes trace generation and learning; evaluation, request-log compression and plotting are excluded. Absolute times depend on the recorded machine and environment. Source hashes, package versions and configuration accompany each run.

## 10. Actual results generated by the project

# Actual generated results: quick/preliminary

Values below come from frozen policies on the saved held-out traces. Error terms are sample standard deviations across five training seeds. The baseline is deterministic on these traces and replicated for pairing; its zero SD does not estimate workload uncertainty.

| Policy | Reward/step | Instance-seconds | SLA violation fraction | p95 completed response (ms) | Training (s) |
|---|---:|---:|---:|---:|---:|
| Q-Learning | -0.6334 ± 0.5023 | 16152.0000 ± 2306.4518 | 0.4894 ± 0.1297 | 9994.2602 ± 3.5789 | 0.8505 ± 0.0877 |
| SARSA | -0.3531 ± 0.6084 | 16224.0000 ± 2649.7321 | 0.4157 ± 0.1558 | 9994.2802 ± 3.9551 | 0.8405 ± 0.1079 |
| Expected SARSA | -0.5893 ± 0.5597 | 16032.0000 ± 2525.1376 | 0.4783 ± 0.1482 | 9994.4138 ± 3.7541 | 1.1065 ± 0.0479 |
| Dyna-Q | -1.0120 ± 0.9407 | 15528.0000 ± 3811.6296 | 0.4857 ± 0.1071 | 9995.3160 ± 3.6365 | 1.3937 ± 0.0836 |
| Predictive Q-Learning | -0.1972 ± 0.5427 | 17304.0000 ± 2244.0321 | 0.3743 ± 0.1505 | 9993.0815 ± 4.2131 | 1.2148 ± 0.1799 |
| Threshold baseline | -0.2518 ± 0.0000 | 15540.0000 ± 0.0000 | 0.3985 ± 0.0000 | 9996.2369 ± 0.0000 | 0.0000 ± 0.0000 |

## Predictive experiment

Predictive minus standard Q-learning, paired mean differences: reward +0.4363; instance-seconds +1152.0; overall SLA violation fraction -0.115193; startup-arrival violation fraction of the complete cohort -0.037903.

Forecast MAE: 111.564 req/s; RMSE: 153.923 req/s. Forecast errors are identical across policies because the predictor uses exogenous measured arrivals only.

The predictive policy changes mean resource cost by +7.13% and overall violations by -11.52 percentage points. No maximum acceptable cost increase was specified, so cost acceptability cannot be declared from these differences alone.

Startup-associated violations refer to requests arriving while new capacity is unavailable, followed to resolution. This association does not isolate the causal effect of the delay. Consult both conditional startup violation rate and startup exposure counts in the CSV, since policies request different amounts of startup.

## Interpretation

Use reward together with cost, SLA failures and completed-request latency. High reward alone does not establish deployment suitability. The above differences are descriptive, without a statistical significance claim.

None of the policies meets the 200 ms target on its across-seed mean pooled completed-request p95 in this run. High demand and startup delays produce substantial failures; these results do not demonstrate service-target compliance.

Stabilization uses two adjacent 50-episode windows and requires persistent mean changes at most 0.10 reward units and within-window SD at most 0.5. Missing values mean not observed within the budget; a 20-episode quick run cannot satisfy this criterion.

Full-budget results: [PENDING — run python train_all.py and use that output directory.]


### Figure placement and traceability

Insert the generated figures from the selected results directory:

1. `plots/combined_learning_curves.png`: all five learned methods, with across-seed variability.
2. `plots/reward_comparison.png`: held-out reward with error bars.
3. `plots/sla_comparison.png`: cohort service violations after drain.
4. `plots/resource_cost_comparison.png`: control-window billed capacity.
5. `plots/training_time_comparison.png`: computational cost.
6. `comparison.png`: readable main-metric comparison.
7. Selected `plots/timeline_*.png`: demand, capacity and boundary queue for the preselected first training seed.

Individual raw and moving-average learning plots are also available for every learner. The complete comparison CSV contains metrics omitted from the compact image, including rejection/timeout rates, scaling actions, stabilization availability and forecast measures. `seed_metrics.csv` preserves individual values; `trace_metrics.csv` supports separate ramp, recurring and burst analysis. No best-seed plot is selected.

## 11. Comparative analysis

### 11.1 Assess service and cost together

The central comparison is between failure avoidance and resource use. Inspect reward alongside SLA violation fraction, completed-response p95 and instance-seconds. A low completed-response percentile can coexist with rejected or timed-out requests because failures are not completed-response samples. For that reason, p95 is never used as the sole service metric.

A cost-efficient policy may retain less spare capacity during quiet periods but respond too late to a sudden increase. Conversely, a policy that holds five servers can improve service while paying unnecessary cost. The timeline figures expose when scaling occurs and whether the queue persists during startup. The numeric comparison should be discussed in terms of these observable behaviors, avoiding an unqualified winner chosen from training reward.

### 11.2 Variability and learning progress

The summary uses the mean and sample standard deviation of per-seed evaluation metrics. Reward variance is computed with one degree of freedom removed. Request-level outcomes are pooled across traces before each seed's percentile and failure fractions are calculated. Percentiles are not averaged across minutes.

Training stabilization is a diagnostic rather than proof of convergence. It requires two adjacent 50-episode windows with mean difference at most 0.10 reward units and SD at most 0.5 in each, sustained for remaining windows. Missing values are censored observations, not zero episodes. A quick experiment cannot meet the minimum length; full experiments can also remain unstabilized.

The larger predictive table may need more experience, while Dyna planning consumes extra computation. Equal real interactions do not imply equal runtime or equal number of updates. Report both budget and timing before interpreting faster apparent learning.

## 12. Innovation component: Predictive Q-Learning for Proactive Cloud Scaling

### 12.1 Design

After interval `t` is observed, exponential smoothing updates `F_(t+1)=beta*lambda_t+(1-beta)*F_t`, with beta=0.30. The initial forecast is the first measured warm-up rate. Its low, medium or high band is appended to the standard state. The predictive Q-table has 810 nominal rows and 2,430 action values; 81 rows include impossible active-plus-pending capacity combinations.

The forecast supplies information to the learner rather than directly forcing a scaling decision. It uses only elapsed measurements. Forecast error for an interval compares the estimate available before that interval with the arrival rate observed afterward. Because arrival input is shared and policy-independent, the forecast error is also shared across evaluated policies.

### 12.2 Evaluation and interpretation

The paired difference file subtracts standard Q-learning from predictive Q-learning for every training seed. Examine reward, overall service failure, resource cost, scaling frequency and startup-associated failures. Startup-associated failures are outcomes of requests arriving during an interval in which new capacity was still unavailable, including requests resolving during the drain. Both their conditional rate and their share of all arrivals are reported.

This measure is an association. A policy that rarely scales has less startup exposure, so a lower startup-associated total does not by itself prove that its forecast improved readiness. Check the startup arrival counts and overall SLA outcomes. A causal startup-delay effect would require an additional matched intervention, such as a zero-delay simulator ablation, which is not claimed here.

Smoothing may help characterize persistent demand levels but tends to lag increases and cannot predict an abrupt unannounced burst. The larger state space can also slow learning. The generated differences in Section 10 determine whether the executed run shows any practical benefit; the report must retain unfavorable or inconclusive findings.

## 13. Discussion and limitations

### 13.1 Model validity

Poisson arrivals and exponential service requirements are controllable assumptions, not universal descriptions of production traffic. Real services can contain heavy tails, correlated arrivals, heterogeneous hosts and downstream bottlenecks. The simulator excludes network and storage delay, models immediate cancellation at timeout, and treats billed seconds uniformly. These assumptions limit transfer of numerical results to a provider environment.

The small observation deliberately loses information about exact rate, request age and hidden demand regime. Even with a mask, it is not a fully observed Markov state. Constant-step tabular learning and Dyna's last-observation model therefore have no optimality guarantee in this setup. Reward weights and the baseline thresholds are unoptimized design choices. Held-out workloads are a limited synthetic suite, and no significance claim is made from five training seeds.

### 13.2 Measurement strengths and remaining gaps

The implementation retains exact response times and resolves each evaluated arrival once. Drain accounting prevents the final backlog from disappearing from failure metrics. Common request-level random inputs reduce confounding from different sampled workloads. Frozen-policy assertions distinguish learning from testing, and saved configuration and source hashes support reruns.

The full budget can still require substantial compute and disk space. Request logs are compressed arrays rather than a graphical event viewer. The baseline's repeated seed rows are intentionally not independent trials. A preliminary budget is useful for checking the pipeline but too short for learning claims. Full-run placeholders must remain until the full experiment is actually executed.

### 13.3 Further experiments

Future work can vary service-time distributions, workload families, forecast beta, reward weights and Dyna planning budget using validation traces rather than the held-out test set. Additional ablations could remove trend from the predictive state, compare a persistence forecast, or isolate startup delay. Any such experiment needs its own recorded configuration and actual outputs. None is implied to have been performed by the present report.

## 14. Conclusion

The project provides an executable continuation of the Component 1 cloud-allocation formulation. Four masked TD methods, a utilization baseline and a forecast-augmented Q-learning policy operate on a shared continuous-time request simulator. The evaluation pipeline reports frozen-policy service, cost and learning metrics, with complete arrival cohorts and reproducible inputs.

The scope of the empirical conclusion is the executed budget labeled in Section 10. Quick results support functionality and reproducibility, while full-budget learning and broader generalization remain separate questions. A defensible final recommendation must consider failure fraction, completed latency, instance cost and variability together rather than selecting a controller from training reward alone.

## 15. IEEE-style references

[1] C. J. C. H. Watkins and P. Dayan, “Q-learning,” *Machine Learning*, vol. 8, pp. 279–292, 1992, doi: 10.1007/BF00992698. [Online]. Available: https://doi.org/10.1007/BF00992698

[2] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, 2018. [Online]. Available: https://mitpress.mit.edu/9780262039246/reinforcement-learning/

[3] Kubernetes Authors, “Horizontal Pod Autoscaling,” *Kubernetes Documentation*. [Online]. Available: https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/ . Accessed: Sep. 5, 2026.

[4] “Adaptive Cloud Server Resource Allocation: A Reinforcement Learning Formulation,” *Reinforcement Learning, CIA–3, Component 1 Case Study Report*, CHRIST (Deemed to be University), Sep. 2026, pp. 1–14. User-supplied project document.

[5] “Adaptive Cloud Server Resource Allocation: Learning When to Scale with Reinforcement Learning,” *CIA–3 Component 1 Presentation*, Sep. 2026, slides 1–16. User-supplied project document.

## Appendix A. Reproduction checklist

Use Python 3.12 in a clean virtual environment, install `requirements.txt`, execute `python -m pytest`, and run `python train_all.py --quick` or the full `python train_all.py`. Use `--output` to preserve earlier runs. Execute `python compare_results.py --results <directory>` to rebuild the figures and report. Archive the selected results directory with source and the original Component 1 documents. Complete the title-page names and institutional layout before submission.

## Appendix B. Request-log schema

Each NPZ stores numeric arrays indexed by request arrival order: `arrival_seconds`, `service_seconds`, `service_start_seconds`, `resolution_seconds`, `outcome`, `server` and `arrived_during_startup`. Outcomes are 1 timely, 2 late, 3 rejected and 4 timed out; zero is unresolved and disallowed after drain. Service start/server −1 indicates no service began. Times are relative to the scored evaluation window. Only requests arriving at time ≥0 are included in final cohort logs.
