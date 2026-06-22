# Labeling Guide For Threshold Calibration

Use this guide when filling the `label` column in `pairs_for_labeling.csv`.

## Label options
- `positive`
  - Generated model is usable without geometric rework.
  - Key dimensions/features match intent.
  - Topology is acceptable.
- `negative`
  - Model is wrong part, wrong topology, major dimensional miss, or unusable.
  - Would require meaningful geometry rebuild.
- `review`
  - Borderline case, unclear intent compliance, or uncertainty between positive/negative.
  - These rows are ignored by automatic tuning unless relabeled.

## Recommended process
1. Start with obvious rows first:
   - clear pass -> `positive`
   - clear fail -> `negative`
2. Mark uncertain rows as `review`.
3. Resolve review rows with a second engineer where possible.
4. Keep notes in the `notes` column for auditability.

## Minimum data quality for tuning
- At least 10 `positive` and 10 `negative` rows per metric (higher is better).
- Prefer 300+ labeled rows total for stable thresholds.
- Ensure part-family coverage and both L2/L3 prompt coverage.

## Consistency policy
- Use the same interpretation of “usable CAD” across all reviewers.
- Do not relabel after tuning unless you version the calibration set.

