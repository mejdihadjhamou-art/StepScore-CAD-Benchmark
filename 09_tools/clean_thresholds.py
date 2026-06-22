#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metric_engine import DEFAULT_THRESHOLDS


SUSPECT_METRICS = {
    "registration_failure_rate",
    "cross_section_iou",
    "render_ssim",
    "render_lpips",
    "silhouette_iou",
    "volume_diff_percent",
    "signed_volume_diff_percent",
    "surface_area_diff_percent",
    "curvature_distribution_error",
    "occupancy_precision",
    "occupancy_recall",
    "occupancy_f1",
    "voxel_iou",
    "tolerance_band_pass_rate",
}


def load_thresholds(path: Path) -> dict:
    raw = json.loads(path.read_text())
    if raw and isinstance(next(iter(raw.values())), dict):
        return {k: v.get("threshold") for k, v in raw.items() if "threshold" in v}
    return {k: v for k, v in raw.items()}


def main() -> int:
    src = ROOT / "threshold_tuning/output_global/threshold_overrides.json"
    dst = ROOT / "threshold_tuning/output_global/threshold_overrides_cleaned.json"
    if not src.exists():
        print(f"missing: {src}")
        return 1

    tuned = load_thresholds(src)
    cleaned = dict(tuned)

    for m in SUSPECT_METRICS:
        if m in DEFAULT_THRESHOLDS:
            cleaned[m] = DEFAULT_THRESHOLDS[m]
        else:
            cleaned.pop(m, None)

    dst.write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
    print(f"wrote: {dst}")
    print(f"overrides_reset={len(SUSPECT_METRICS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
