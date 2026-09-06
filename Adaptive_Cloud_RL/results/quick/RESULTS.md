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
