from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from generation_pipeline import GenerationError, generate_and_export
from metric_engine import ALIGNMENT_METHODS, DEFAULT_ALIGNMENT_METHOD, GRADING_PROFILE_FULL_44, compare_models, get_grading_profiles
from step_utils import StepConversionError, ensure_mesh_path


@dataclass
class JobSpec:
    job_key: str
    source_row_idx: int
    source_row: Dict[str, str]
    provider: str
    model: str
    prompt_text: str
    reference_path: str
    generation_mode: str
    base_step_path: Optional[str]
    grading_profile: str
    sample_points: int
    voxel_pitch_mm: float
    fast_mode: bool
    alignment_method: str
    threshold_overrides: Dict[str, float]
    timeout_seconds: int
    part_id: str
    family: str
    prompt_level: str
    replicate: int


def _utc_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "item"


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    s = str(value).strip().lower()
    if not s:
        return default
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    try:
        return float(s)
    except Exception:
        return default


def _parse_json_dict(raw: str, label: str) -> Dict[str, float]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {label}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{label} must be a JSON object.")
    out: Dict[str, float] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = float(v)
        except Exception as exc:
            raise ValueError(f"Threshold override '{k}' is not numeric: {v!r}") from exc
    return out


def _resolve_path(raw_path: str, manifest_dir: Path) -> str:
    p = Path((raw_path or "").strip()).expanduser()
    if not p.is_absolute():
        p = (manifest_dir / p).resolve()
    return str(p)


def _classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg:
        return "rate_limit"
    if "timeout" in msg:
        return "timeout"
    if isinstance(exc, StepConversionError):
        return "conversion_error"
    if isinstance(exc, GenerationError):
        return "generation_error"
    return "runtime_error"


def _build_job_key(
    row: Dict[str, str],
    source_row_idx: int,
    provider: str,
    model: str,
    prompt_level: str,
    replicate: int,
) -> str:
    raw = row.get("job_id", "").strip()
    if raw:
        base = _sanitize_token(raw)
        if replicate != 1:
            return f"{base}__r{replicate}"
        return base
    else:
        part_id = _sanitize_token(row.get("part_id", "") or row.get("task_id", "") or f"row{source_row_idx+1}")
        base = f"{part_id}__{_sanitize_token(prompt_level or 'NA')}__{_sanitize_token(provider)}__{_sanitize_token(model)}"
    return f"{base}__r{replicate}"


def _load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _prepare_jobs(
    rows: List[Dict[str, str]],
    manifest_dir: Path,
    args: argparse.Namespace,
) -> Tuple[List[JobSpec], List[Dict[str, Any]]]:
    profiles = get_grading_profiles()
    global_thresholds = _parse_json_dict(args.threshold_overrides_json, "global threshold overrides")

    jobs: List[JobSpec] = []
    row_meta: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        status = (row.get("status", "") or "").strip().lower()
        enabled = _as_bool(row.get("enabled"), True)
        if args.status_filter and status != args.status_filter.strip().lower():
            row_meta.append({"row_idx": idx, "selected": False, "reason": "status_filter"})
            continue
        if not enabled:
            row_meta.append({"row_idx": idx, "selected": False, "reason": "disabled"})
            continue

        provider = (row.get("provider") or args.default_provider or "").strip().lower()
        model = (row.get("model") or args.default_model or "").strip()
        prompt_text = (row.get(args.prompt_column) or "").strip()
        reference_raw = (row.get(args.reference_column) or "").strip()

        if not provider:
            row_meta.append({"row_idx": idx, "selected": False, "reason": "missing_provider"})
            continue
        if provider not in {"openai", "anthropic"}:
            row_meta.append({"row_idx": idx, "selected": False, "reason": f"unsupported_provider:{provider}"})
            continue
        if not model:
            row_meta.append({"row_idx": idx, "selected": False, "reason": "missing_model"})
            continue
        if not prompt_text:
            row_meta.append({"row_idx": idx, "selected": False, "reason": f"missing_prompt_column:{args.prompt_column}"})
            continue
        if not reference_raw:
            row_meta.append({"row_idx": idx, "selected": False, "reason": f"missing_reference_column:{args.reference_column}"})
            continue

        generation_mode = (row.get("generation_mode") or args.default_generation_mode).strip().lower()
        if generation_mode not in {"generate", "modify"}:
            row_meta.append({"row_idx": idx, "selected": False, "reason": f"bad_generation_mode:{generation_mode}"})
            continue

        base_step_path: Optional[str] = None
        if generation_mode == "modify":
            base_raw = (row.get("base_step_path") or "").strip()
            if not base_raw:
                row_meta.append({"row_idx": idx, "selected": False, "reason": "missing_base_step_path_for_modify"})
                continue
            base_step_path = _resolve_path(base_raw, manifest_dir)

        grading_profile = (row.get("grading_profile") or args.grading_profile).strip().lower()
        if grading_profile not in profiles:
            grading_profile = GRADING_PROFILE_FULL_44

        row_thresholds = _parse_json_dict(row.get("threshold_overrides_json", ""), f"row {idx+1} threshold overrides")
        merged_thresholds = dict(global_thresholds)
        merged_thresholds.update(row_thresholds)

        sample_points = _as_int(row.get("sample_points"), args.sample_points)
        voxel_pitch_mm = _as_float(row.get("voxel_pitch_mm"), args.voxel_pitch_mm)
        fast_mode = _as_bool(row.get("fast_mode"), args.fast_mode)
        alignment_method = (row.get("alignment_method") or args.alignment_method).strip().lower()
        timeout_seconds = _as_int(row.get("timeout_seconds"), args.timeout_seconds)
        part_id = (row.get("part_id") or row.get("task_id") or f"row_{idx+1}").strip()
        family = (row.get("family") or "").strip()
        prompt_level = (row.get("prompt_level") or args.prompt_column).strip()

        if row.get("replicate", "").strip():
            replicate_values = [_as_int(row.get("replicate"), 1)]
        else:
            replicate_values = list(range(1, max(1, args.replicates) + 1))

        reference_path = _resolve_path(reference_raw, manifest_dir)
        for rep in replicate_values:
            job_key = _build_job_key(
                row=row,
                source_row_idx=idx,
                provider=provider,
                model=model,
                prompt_level=prompt_level,
                replicate=rep,
            )
            jobs.append(
                JobSpec(
                    job_key=job_key,
                    source_row_idx=idx,
                    source_row=row,
                    provider=provider,
                    model=model,
                    prompt_text=prompt_text,
                    reference_path=reference_path,
                    generation_mode=generation_mode,
                    base_step_path=base_step_path,
                    grading_profile=grading_profile,
                    sample_points=sample_points,
                    voxel_pitch_mm=voxel_pitch_mm,
                    fast_mode=fast_mode,
                    alignment_method=alignment_method,
                    threshold_overrides=merged_thresholds,
                    timeout_seconds=timeout_seconds,
                    part_id=part_id,
                    family=family,
                    prompt_level=prompt_level,
                    replicate=rep,
                )
            )

        row_meta.append({"row_idx": idx, "selected": True, "reason": ""})

    if args.limit > 0:
        jobs = jobs[: args.limit]
    return jobs, row_meta


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _run_one_job(
    job: JobSpec,
    run_dir: Path,
    max_retries: int,
    retry_backoff_seconds: float,
    resume: bool,
) -> Dict[str, Any]:
    job_dir = run_dir / "jobs" / job.job_key
    job_dir.mkdir(parents=True, exist_ok=True)
    final_result_path = job_dir / "result.json"
    final_summary_path = job_dir / "summary.json"

    if resume and final_result_path.exists() and final_summary_path.exists():
        summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
        summary["status"] = "skipped_completed"
        summary["resume_skipped"] = True
        return summary

    attempts_log: List[Dict[str, Any]] = []
    start_total = time.time()
    last_error: Optional[str] = None
    last_error_type = ""

    for attempt in range(1, max_retries + 2):
        attempt_dir = job_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)

        attempt_input = {
            "job_key": job.job_key,
            "attempt": attempt,
            "provider": job.provider,
            "model": job.model,
            "part_id": job.part_id,
            "family": job.family,
            "prompt_level": job.prompt_level,
            "replicate": job.replicate,
            "reference_path": job.reference_path,
            "generation_mode": job.generation_mode,
            "base_step_path": job.base_step_path,
            "grading_profile": job.grading_profile,
            "sample_points": job.sample_points,
            "voxel_pitch_mm": job.voxel_pitch_mm,
            "fast_mode": job.fast_mode,
            "alignment_method": job.alignment_method,
            "timeout_seconds": job.timeout_seconds,
            "threshold_overrides": job.threshold_overrides,
        }
        _write_json(attempt_dir / "inputs.json", attempt_input)

        gen_s = time.time()
        try:
            generation_info = generate_and_export(
                prompt=job.prompt_text,
                provider=job.provider,
                model=job.model,
                run_dir=str(attempt_dir),
                generation_mode=job.generation_mode,
                base_step_path=job.base_step_path,
                timeout_seconds=job.timeout_seconds,
            )
            gen_dur = time.time() - gen_s

            cmp_s = time.time()
            ref_mesh_path = ensure_mesh_path(
                input_path=job.reference_path,
                run_dir=str(attempt_dir),
                prefix="reference",
            )
            gen_mesh_path = ensure_mesh_path(
                input_path=generation_info["generated_step_path"],
                run_dir=str(attempt_dir),
                prefix="generated",
            )
            compare_result = compare_models(
                reference_path=ref_mesh_path,
                generated_path=gen_mesh_path,
                sample_points=job.sample_points,
                voxel_pitch_mm=float(job.voxel_pitch_mm),
                thresholds=job.threshold_overrides,
                fast_mode=job.fast_mode,
                grading_profile=job.grading_profile,
                alignment_method=job.alignment_method,
            )
            cmp_dur = time.time() - cmp_s
            if not compare_result.get("ok"):
                raise RuntimeError(compare_result.get("error") or "compare_models returned ok=False")

            total_dur = time.time() - start_total
            payload = {
                "job": attempt_input,
                "attempts_used": attempt,
                "attempts_log": attempts_log,
                "generation": {
                    "provider": generation_info.get("provider"),
                    "model": generation_info.get("model"),
                    "generation_mode": generation_info.get("generation_mode"),
                    "generated_code_path": generation_info.get("generated_code_path"),
                    "generated_step_path": generation_info.get("generated_step_path"),
                    "generated_stl_path": generation_info.get("generated_stl_path"),
                    "auto_repair_attempts_used": generation_info.get("auto_repair_attempts_used"),
                },
                "compare": compare_result,
                "timing_seconds": {
                    "generation": gen_dur,
                    "compare": cmp_dur,
                    "total": total_dur,
                },
            }
            _write_json(attempt_dir / "result.json", payload)
            _write_json(final_result_path, payload)

            summary = {
                "job_key": job.job_key,
                "status": "success",
                "provider": job.provider,
                "model": job.model,
                "part_id": job.part_id,
                "family": job.family,
                "prompt_level": job.prompt_level,
                "replicate": job.replicate,
                "generation_mode": job.generation_mode,
                "grading_profile": job.grading_profile,
                "attempts_used": attempt,
                "retries_used": attempt - 1,
                "duration_generation_s": gen_dur,
                "duration_compare_s": cmp_dur,
                "duration_total_s": total_dur,
                "reference_path": job.reference_path,
                "generated_step_path": generation_info.get("generated_step_path", ""),
                "result_json_path": str(final_result_path),
                "pass_rate": compare_result["summary"]["pass_rate"],
                "overall_pass": compare_result["summary"]["overall_pass"],
                "quality_score_0_100": compare_result["summary"].get("quality_score_0_100"),
                "scored_metrics": compare_result["summary"]["total_metrics"],
                "computed_metrics": compare_result["summary"].get("total_metrics_computed", compare_result["summary"]["total_metrics"]),
                "pass_count": compare_result["summary"]["pass_count"],
                "fail_count": compare_result["summary"]["fail_count"],
                "error_type": "",
                "error_message": "",
                "resume_skipped": False,
                "source_row_idx": job.source_row_idx,
            }
            _write_json(final_summary_path, summary)
            return summary

        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            last_error_type = _classify_error(exc)
            attempts_log.append(
                {
                    "attempt": attempt,
                    "error_type": last_error_type,
                    "error_message": last_error,
                }
            )
            (attempt_dir / "error.txt").write_text(last_error, encoding="utf-8")
            if attempt <= max_retries:
                sleep_s = retry_backoff_seconds * (2 ** (attempt - 1))
                time.sleep(max(0.0, sleep_s))

    total_dur = time.time() - start_total
    summary = {
        "job_key": job.job_key,
        "status": "failed",
        "provider": job.provider,
        "model": job.model,
        "part_id": job.part_id,
        "family": job.family,
        "prompt_level": job.prompt_level,
        "replicate": job.replicate,
        "generation_mode": job.generation_mode,
        "grading_profile": job.grading_profile,
        "attempts_used": max_retries + 1,
        "retries_used": max_retries,
        "duration_generation_s": None,
        "duration_compare_s": None,
        "duration_total_s": total_dur,
        "reference_path": job.reference_path,
        "generated_step_path": "",
        "result_json_path": str(final_result_path),
        "pass_rate": None,
        "overall_pass": False,
        "quality_score_0_100": None,
        "scored_metrics": None,
        "computed_metrics": None,
        "pass_count": None,
        "fail_count": None,
        "error_type": last_error_type,
        "error_message": last_error or "unknown error",
        "resume_skipped": False,
        "source_row_idx": job.source_row_idx,
    }
    _write_json(final_summary_path, summary)
    _write_json(
        final_result_path,
        {
            "job": {
                "job_key": job.job_key,
                "provider": job.provider,
                "model": job.model,
                "part_id": job.part_id,
                "prompt_level": job.prompt_level,
                "replicate": job.replicate,
            },
            "attempts_log": attempts_log,
            "error": summary["error_message"],
        },
    )
    return summary


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _group_stats(rows: List[Dict[str, Any]], keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in rows:
        key = tuple(r.get(k, "") for k in keys)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    for key, items in groups.items():
        success = [x for x in items if x.get("status") == "success"]
        passed = [x for x in success if bool(x.get("overall_pass"))]
        avg_quality = (
            sum(float(x.get("quality_score_0_100") or 0.0) for x in success) / len(success)
            if success
            else None
        )
        avg_total_s = (
            sum(float(x.get("duration_total_s") or 0.0) for x in success) / len(success)
            if success
            else None
        )
        row = {
            **{k: v for k, v in zip(keys, key)},
            "jobs_total": len(items),
            "jobs_success": len(success),
            "jobs_failed": len([x for x in items if x.get("status") == "failed"]),
            "jobs_skipped_completed": len([x for x in items if x.get("status") == "skipped_completed"]),
            "overall_pass_count": len(passed),
            "overall_pass_rate_on_success": (len(passed) / len(success)) if success else None,
            "avg_quality_score_0_100_success": avg_quality,
            "avg_duration_total_s_success": avg_total_s,
        }
        out.append(row)
    return out


def _build_status_manifest(
    original_rows: List[Dict[str, str]],
    run_results: List[Dict[str, Any]],
    run_id: str,
) -> List[Dict[str, Any]]:
    results_by_row: Dict[int, List[Dict[str, Any]]] = {}
    for r in run_results:
        idx = int(r.get("source_row_idx", -1))
        if idx >= 0:
            results_by_row.setdefault(idx, []).append(r)

    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(original_rows):
        entries = results_by_row.get(idx, [])
        merged = dict(row)
        if not entries:
            merged["harness_status"] = "not_selected_or_not_run"
            merged["harness_run_id"] = run_id
            merged["harness_jobs"] = 0
            merged["harness_success"] = 0
            merged["harness_failed"] = 0
        else:
            merged["harness_status"] = "success" if any(e["status"] == "success" for e in entries) else "failed"
            merged["harness_run_id"] = run_id
            merged["harness_jobs"] = len(entries)
            merged["harness_success"] = sum(1 for e in entries if e["status"] == "success")
            merged["harness_failed"] = sum(1 for e in entries if e["status"] == "failed")
            merged["harness_skipped_completed"] = sum(1 for e in entries if e["status"] == "skipped_completed")
        out.append(merged)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STEPScore batch testing harness runner.")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV manifest path.")
    parser.add_argument("--run-id", type=str, default="", help="Optional run id. Default is UTC timestamp.")
    parser.add_argument("--output-root", type=Path, default=Path(".stepscore_harness_runs"), help="Root output directory.")
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2), help="Concurrent job workers.")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries per job after first failure.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=3.0, help="Initial exponential backoff.")
    parser.add_argument("--timeout-seconds", type=int, default=240, help="CadQuery execution timeout per attempt.")
    parser.add_argument("--resume", action="store_true", help="Skip jobs already completed in run folder.")
    parser.add_argument("--limit", type=int, default=0, help="Run only first N jobs after filtering (0 = all).")
    parser.add_argument("--dry-run", action="store_true", help="Validate and plan jobs without executing model calls.")

    parser.add_argument("--status-filter", type=str, default="", help="Only rows whose `status` matches this value.")
    parser.add_argument("--replicates", type=int, default=1, help="Replicate count when manifest row has no `replicate`.")
    parser.add_argument("--prompt-column", type=str, default="prompt_text", help="Manifest column containing prompt.")
    parser.add_argument("--reference-column", type=str, default="reference_path", help="Manifest column containing reference path.")
    parser.add_argument("--default-provider", type=str, default="openai", help="Fallback provider if missing in row.")
    parser.add_argument("--default-model", type=str, default="gpt-5.2", help="Fallback model if missing in row.")
    parser.add_argument("--default-generation-mode", type=str, default="generate", choices=["generate", "modify"], help="Fallback mode if missing in row.")
    parser.add_argument("--grading-profile", type=str, default=GRADING_PROFILE_FULL_44, help="Default grading profile.")
    parser.add_argument("--sample-points", type=int, default=10000, help="Default compare sample points.")
    parser.add_argument("--voxel-pitch-mm", type=float, default=2.0, help="Default compare voxel pitch.")
    parser.add_argument("--fast-mode", action="store_true", default=False, help="Default compare fast mode if row omits fast_mode.")
    parser.add_argument(
        "--alignment-method",
        type=str,
        default=DEFAULT_ALIGNMENT_METHOD,
        choices=list(ALIGNMENT_METHODS),
        help="Alignment method for mesh registration (default: pca_icp).",
    )
    parser.add_argument(
        "--threshold-overrides-json",
        type=str,
        default="{}",
        help='Global threshold overrides JSON object. Example: {"chamfer_distance_mm":0.8}',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    profiles = get_grading_profiles()
    if args.grading_profile not in profiles:
        raise ValueError(f"Unknown grading profile '{args.grading_profile}'. Available: {sorted(profiles.keys())}")

    run_id = args.run_id.strip() or _utc_ts()
    run_dir = (args.output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "jobs").mkdir(parents=True, exist_ok=True)

    manifest_rows = _load_manifest_rows(args.manifest)
    jobs, row_meta = _prepare_jobs(manifest_rows, args.manifest.resolve().parent, args)

    plan_payload = {
        "run_id": run_id,
        "manifest": str(args.manifest.resolve()),
        "output_root": str(args.output_root.resolve()),
        "job_count": len(jobs),
        "max_workers": args.max_workers,
        "max_retries": args.max_retries,
        "resume": args.resume,
        "dry_run": args.dry_run,
        "grading_profiles": profiles,
        "row_selection_meta": row_meta,
    }
    _write_json(run_dir / "run_plan.json", plan_payload)

    if args.dry_run:
        print(f"dry_run=1 planned_jobs={len(jobs)} run_dir={run_dir}")
        return

    print(f"run_id={run_id}")
    print(f"jobs_planned={len(jobs)}")
    print(f"max_workers={args.max_workers}")

    results: List[Dict[str, Any]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        fut_map = {
            ex.submit(
                _run_one_job,
                job,
                run_dir,
                max(0, args.max_retries),
                max(0.0, args.retry_backoff_seconds),
                bool(args.resume),
            ): job
            for job in jobs
        }
        for i, fut in enumerate(as_completed(fut_map), start=1):
            job = fut_map[fut]
            try:
                out = fut.result()
            except Exception as exc:  # noqa: BLE001
                out = {
                    "job_key": job.job_key,
                    "status": "failed",
                    "provider": job.provider,
                    "model": job.model,
                    "part_id": job.part_id,
                    "family": job.family,
                    "prompt_level": job.prompt_level,
                    "replicate": job.replicate,
                    "generation_mode": job.generation_mode,
                    "grading_profile": job.grading_profile,
                    "attempts_used": 0,
                    "retries_used": 0,
                    "duration_total_s": None,
                    "reference_path": job.reference_path,
                    "generated_step_path": "",
                    "result_json_path": "",
                    "pass_rate": None,
                    "overall_pass": False,
                    "quality_score_0_100": None,
                    "error_type": _classify_error(exc),
                    "error_message": str(exc),
                    "resume_skipped": False,
                    "source_row_idx": job.source_row_idx,
                }
            results.append(out)
            print(f"[{i}/{len(jobs)}] {job.job_key} -> {out['status']}")

    elapsed = time.time() - started
    results = sorted(results, key=lambda x: str(x.get("job_key", "")))

    _write_csv(run_dir / "results.csv", results)
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    summary_overall = {
        "run_id": run_id,
        "manifest": str(args.manifest.resolve()),
        "jobs_total": len(results),
        "jobs_success": sum(1 for r in results if r.get("status") == "success"),
        "jobs_failed": sum(1 for r in results if r.get("status") == "failed"),
        "jobs_skipped_completed": sum(1 for r in results if r.get("status") == "skipped_completed"),
        "elapsed_seconds": elapsed,
    }
    _write_json(run_dir / "summary_overall.json", summary_overall)

    _write_csv(run_dir / "summary_by_model.csv", _group_stats(results, ("provider", "model", "grading_profile")))
    _write_csv(run_dir / "summary_by_family.csv", _group_stats(results, ("family", "grading_profile")))
    _write_csv(run_dir / "summary_by_prompt_level.csv", _group_stats(results, ("prompt_level", "grading_profile")))

    status_manifest = _build_status_manifest(manifest_rows, results, run_id)
    _write_csv(run_dir / "manifest_with_harness_status.csv", status_manifest)

    print(f"run_dir={run_dir}")
    print(f"jobs_success={summary_overall['jobs_success']}")
    print(f"jobs_failed={summary_overall['jobs_failed']}")
    print(f"jobs_skipped_completed={summary_overall['jobs_skipped_completed']}")


if __name__ == "__main__":
    main()
