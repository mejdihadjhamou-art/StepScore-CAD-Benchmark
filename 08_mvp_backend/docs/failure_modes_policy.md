# STEPScore Failure Modes Policy

## Purpose
Define two distinct evaluation modes so benchmark results separate:
- raw model capability (strict first pass), and
- practical recoverability (assisted with repair).

This prevents ambiguity when discussing "model failure."

## Failure Mode A: Strict First-Pass Failure

### Definition
A job is **failed** if the model does not produce executable/exportable CAD on the initial generation attempt.

### Configuration
- `CADQUERY_AUTO_REPAIR_ATTEMPTS=0`
- Harness retries can remain enabled for transient API/network issues, but each attempt must still be first-pass only.
- Recommended grading profile:
  - primary: `full_44`
  - optional parallel report: `refined_metrics`

### What it measures
- True first-try reliability
- Prompt-to-valid-CAD robustness without helper loops
- Best metric for model ranking/comparison

## Failure Mode B: Assisted Recovery Failure

### Definition
A job is **failed** only if generation still fails after the allowed repair loop(s).

### Configuration
- `CADQUERY_AUTO_REPAIR_ATTEMPTS=1` (recommended)
- Keep `max_retries` small (e.g., `1-2`) to avoid inflated call counts.

### What it measures
- Recoverability under production-style guardrails
- Practical throughput when auto-fix is available

## Required Reporting (both modes)

For every run, report these fields:
- `mode` (`strict_first_pass` or `assisted_recovery`)
- `jobs_total`
- `jobs_success`
- `jobs_failed`
- `overall_pass_rate`
- `quality_score_mean`
- `first_pass_success_rate` (must be explicit in assisted mode)
- `auto_repair_attempts_used` distribution
- `mean_api_calls_per_job` (estimated or measured)

## Interpretation Rules

1. Model quality claims must use **Strict First-Pass** as primary evidence.
2. Assisted results can be presented as deployment utility, not core capability.
3. Never mix strict and assisted scores in one leaderboard column.
4. If assisted gain is large, call that out as a model weakness signal (fragile first pass).

## Recommended Run Matrix

For each model/prompt set:
1. Strict run:
   - `CADQUERY_AUTO_REPAIR_ATTEMPTS=0`
   - Evaluate and store results.
2. Assisted run:
   - `CADQUERY_AUTO_REPAIR_ATTEMPTS=1`
   - Same manifest/settings otherwise.
3. Publish delta:
   - `assisted_pass_rate - strict_pass_rate`
   - `assisted_quality - strict_quality`

## Example Commands

### Strict first-pass
```bash
export CADQUERY_AUTO_REPAIR_ATTEMPTS=0
python harness_runner.py \
  --manifest benchmark_v1/harness_manifest.generated.csv \
  --run-id strict_first_pass_run \
  --max-workers 2 \
  --max-retries 2 \
  --retry-backoff-seconds 5 \
  --resume
```

### Assisted recovery
```bash
export CADQUERY_AUTO_REPAIR_ATTEMPTS=1
python harness_runner.py \
  --manifest benchmark_v1/harness_manifest.generated.csv \
  --run-id assisted_recovery_run \
  --max-workers 2 \
  --max-retries 2 \
  --retry-backoff-seconds 5 \
  --resume
```

## Decision Guidance

- Use **strict** metrics for buyer-facing "how good is the model?" claims.
- Use **assisted** metrics for "how usable is the pipeline in production?" claims.
