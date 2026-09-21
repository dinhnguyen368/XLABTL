# Experiments

## E1 — Paired Challenge Evidence

**Input:** 15 Evaluation images.

**Method:** Baseline only.

**Goal:** quantify whether edge detection performance changes across Light / Medium / Heavy acquisition conditions.

**Primary metric:** Strict F1.

Outputs:

```text
E1_raw.csv
E1_paired_analysis.csv
E1_summary.csv
E1_three_failure_cases.csv
```

Paired deltas:

```text
drop_L_to_M = F1_light - F1_medium
drop_L_to_H = F1_light - F1_heavy
```

Positive means lower F1 at the heavier level. Negative values are observed outcomes and must be retained.

## E2 — Development-only Parameter Study

**Input:** 15 Development images.

Predefined ISO strata:

```text
ISO <= 400
400 < ISO <= 1600
ISO > 1600
```

Bilateral grid:

```text
d: [3, 5, 7]
sigma_color: [25, 50, 75, 100]
sigma_space: [25, 50, 75]
```

For each ISO stratum, E2 evaluates all parameter combinations by mean Strict F1.

A representative image is saved for every parameter combination to support visual analysis.

Outputs:

```text
outputs/tables/E2_*_params.csv
outputs/figures/E2_param_study_*.png
outputs/intermediate/E2/...
outputs/configs/e2_adaptive_rules.json
```

The adaptive rule is frozen after E2.

E2 refuses to overwrite an existing frozen rules file.

## E3 — Ablation

**Input:** 15 Evaluation images.

Comparison:

```text
Baseline:
Grayscale -> Canny

Full Proposed:
Grayscale -> ISO-adaptive Bilateral -> Canny
```

The Evaluation set is never used to tune the E2 rule.

Outputs:

```text
E3_raw.csv
E3_summary.csv
E3_ablation_strict_f1.png
```

## E4 — Controlled Gaussian Robustness Stress Test

E4 is explicitly a controlled synthetic stress test, not a substitute for real-world Evaluation data.

Starting input:

```text
5 Light Evaluation scenes -> grayscale
```

Gaussian noise sigma:

```text
0, 10, 25, 50, 75, 100, 125, 150
```

Noise pattern is deterministic using a hash of:

```text
scene_id + sigma + global seed
```

The Proposed method uses the frozen Heavy rule from E2.

Failure criterion:

```text
Strict F1 < 0.40
```

This is an operational threshold defined for this experiment.

Report:

- mean Baseline F1;
- mean Proposed F1;
- per-scene F1;
- first tested sigma below threshold for each method;
- joint failure sigma, only if both methods are below threshold at the same tested sigma.

Use the phrase **first tested sigma below threshold**, not “exact failure point”.

## No partial results

E1, E2, E3, and E4 must process all expected inputs. If any required image fails:

```text
STOP
EXIT CODE 1
```

No final summary is generated from a partial subset.

## Scientific reporting

Do not pre-assert that the Proposed method wins. Report the measured result, including cases where the Baseline is better.
