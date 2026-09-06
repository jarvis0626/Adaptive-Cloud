# Component 1 implementation alignment

Source documents, read before implementation:

- `../../Adaptive_Cloud_Server_Resource_Allocation_Case_Study.pdf`: 14 pages.
- `../../Adaptive_Cloud_Server_Resource_Allocation.pptx`: 16 slides.

Paths above are relative to this documentation directory and refer to the supplied files in the repository root, one level above the Python project.

| Source | Required design | Component 2 implementation |
|---|---|---|
| PDF pp. 3–5; slides 2–5 | One application, horizontal scaling, 1–5 identical servers | `environment/cloud_environment.py` |
| PDF p. 5 | Poisson arrivals, exponential mean 10 ms service, FIFO, 6,000 waiting capacity, 10 s total timeout | `workload_generator.py` and compiled `simulator.py` |
| PDF p. 6; slide 6 | Five-component observation, 270 nominal states | `state_to_index`, `index_to_state` |
| PDF p. 7; slide 7 | Idle-only scale-in, capacity limits, maintain while pending | `CloudEnvironment.action_mask` and `step` |
| PDF p. 7 §5.2 | Observe pending before activation; requested capacity usable in next interval | `step` boundary ordering, covered by timing tests |
| PDF p. 8; slide 8 | Reward `2g-c-4v-q-0.2m`, all resolutions partitioned | `compute_reward`, worked-example tests |
| PDF pp. 9–10 | Continuing task, truncation bootstrapping, alpha .1, gamma .95 | Independent TD updates and `train_all.py` |
| PDF p. 11 | Measurement warm-up, same transition machinery | Unscored reset interval retaining outstanding work |
| PDF p. 12; slide 14 | Exponential smoothing beta .30, 810 predictive states | `innovation/predictive_q_learning.py` and causal environment forecast |
| PDF p. 13; slide 15 | Four algorithms and threshold comparison, frozen policies, held-out traces | Training/evaluation pipeline with five seeds |
| PDF pp. 5, 13 | Request-level records and complete evaluation cohorts | Compressed per-request arrays; 10-second no-arrival drain |

Clarifications made for reproducibility rather than changes to the formulation:

- Integer action IDs `0/1/2` map to the source's `−1/0/+1` scaling changes.
- Busy time is the utilization metric for the threshold baseline; its defaults are 75% out, 30% in, one-step cooldown and two low observations before scale-in.
- Exact equal-time event rules, fixed greedy tie-breaking and censored stabilization semantics are documented in the README.
- Percentiles are computed from individual completed-request response times, never inferred from minute-level throughput.
- Evaluation records use compact numeric columns instead of individual Python request objects. These retain arrival, queue/service start and resolution timestamps, outcome, serving instance and original service requirement.
- Quick results are explicitly preliminary. Component 1's illustrative rewards and capacities are never presented as measured training results.

The Component 1 references mention a separate course assessment brief. That brief was not included, so its additional formatting or grading rules have not been independently checked.
