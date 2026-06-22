#!/usr/bin/env python3
"""
StepScore CLI — unified command-line interface.

Subcommands:
    run          Run the batch harness on a manifest CSV
    score        Score a single reference/generated STEP pair
    report       Print a formatted report from a completed run
    leaderboard  Compare models across one or more run directories
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


# ─── Subcommand: run ────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    """Thin wrapper around harness_runner.main(), forwarding CLI args."""
    forward: list[str] = ["--manifest", str(args.manifest)]
    if args.run_id:
        forward += ["--run-id", args.run_id]
    if args.output_root:
        forward += ["--output-root", str(args.output_root)]
    if args.max_workers is not None:
        forward += ["--max-workers", str(args.max_workers)]
    if args.resume:
        forward.append("--resume")
    if args.dry_run:
        forward.append("--dry-run")
    if args.fast_mode:
        forward.append("--fast-mode")
    if args.limit:
        forward += ["--limit", str(args.limit)]
    if args.grading_profile:
        forward += ["--grading-profile", args.grading_profile]
    if args.default_provider:
        forward += ["--default-provider", args.default_provider]
    if args.default_model:
        forward += ["--default-model", args.default_model]
    if args.alignment_method:
        forward += ["--alignment-method", args.alignment_method]

    from harness_runner import main as harness_main
    harness_main(argv=forward)


# ─── Subcommand: score ──────────────────────────────────────────────────────

def cmd_score(args: argparse.Namespace) -> None:
    """Score a single reference vs generated STEP pair."""
    from metric_engine import compare_models

    ref = str(Path(args.reference).resolve())
    gen = str(Path(args.generated).resolve())

    print(f"  Reference : {ref}")
    print(f"  Generated : {gen}")
    print(f"  Profile   : {args.grading_profile}")
    print(f"  Alignment : {args.alignment_method}")
    print(f"  Fast mode : {args.fast_mode}")
    print()

    result = compare_models(
        reference_path=ref,
        generated_path=gen,
        sample_points=args.sample_points,
        voxel_pitch_mm=args.voxel_pitch,
        fast_mode=args.fast_mode,
        grading_profile=args.grading_profile,
        alignment_method=args.alignment_method,
    )

    if not result.get("ok"):
        print(f"ERROR: {result.get('error', 'unknown')}")
        sys.exit(1)

    summary = result["summary"]
    metrics = result["metrics"]

    # ── Header
    print("=" * 80)
    print("  STEPSCORE RESULT")
    print("=" * 80)
    print(f"  Quality Score : {summary['quality_score_0_100']:.1f} / 100")
    print(f"  Pass Rate     : {summary['pass_rate']:.1%}  ({summary['pass_count']}/{summary['total_metrics']} metrics)")
    print(f"  Overall Pass  : {'YES' if summary['overall_pass'] else 'NO'}")
    print()

    # ── Per-metric table
    print(f"  {'Metric':<38} {'Value':>10} {'Thresh':>10} {'Pass':>6}")
    print("  " + "-" * 68)
    for m in sorted(metrics, key=lambda x: x["name"]):
        v = m["value"]
        t = m["threshold"]
        p = "PASS" if m["passed"] else "FAIL"
        marker = "" if m["passed"] else " <<<"
        profile_mark = "*" if m.get("in_selected_profile") else " "
        print(f" {profile_mark}{m['name']:<37} {v:>10.4f} {t:>10.4f} {p:>6}{marker}")
    print()
    print("  * = in selected grading profile")
    print()

    # ── Save JSON if requested
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved JSON: {out}")


# ─── Subcommand: report ─────────────────────────────────────────────────────

def cmd_report(args: argparse.Namespace) -> None:
    """Print a formatted report from a completed harness run."""
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"ERROR: Run directory not found: {run_dir}")
        sys.exit(1)

    # ── Load summary_overall.json
    overall_path = run_dir / "summary_overall.json"
    if overall_path.exists():
        overall = json.loads(overall_path.read_text())
    else:
        overall = {}

    # ── Load results.csv
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        print(f"ERROR: results.csv not found in {run_dir}")
        sys.exit(1)

    rows = _load_csv(results_path)

    print("=" * 90)
    print(f"  STEPSCORE RUN REPORT")
    print("=" * 90)
    print(f"  Run ID         : {overall.get('run_id', run_dir.name)}")
    print(f"  Jobs total     : {overall.get('jobs_total', len(rows))}")
    print(f"  Jobs success   : {overall.get('jobs_success', sum(1 for r in rows if r.get('status') == 'success'))}")
    print(f"  Jobs failed    : {overall.get('jobs_failed', sum(1 for r in rows if r.get('status') == 'failed'))}")
    if overall.get("elapsed_seconds"):
        mins = overall["elapsed_seconds"] / 60
        print(f"  Elapsed        : {mins:.1f} min")
    print()

    # ── Success rows only
    success = [r for r in rows if r.get("status") == "success"]
    if not success:
        print("  No successful jobs to report on.")
        return

    # ── Overall stats
    scores = [float(r["quality_score_0_100"]) for r in success if r.get("quality_score_0_100")]
    pass_rates = [float(r["pass_rate"]) for r in success if r.get("pass_rate")]
    avg_score = sum(scores) / len(scores) if scores else 0
    avg_pass = sum(pass_rates) / len(pass_rates) if pass_rates else 0
    full_pass = sum(1 for r in success if r.get("overall_pass") == "True")

    print(f"  Avg Quality Score  : {avg_score:.1f} / 100")
    print(f"  Avg Pass Rate      : {avg_pass:.1%}")
    print(f"  Full Pass Count    : {full_pass} / {len(success)}")
    print()

    # ── By family
    by_family_path = run_dir / "summary_by_family.csv"
    if by_family_path.exists():
        fam_rows = _load_csv(by_family_path)
        print(f"  {'Family':<25} {'Jobs':>6} {'Pass%':>8} {'Avg Score':>10}")
        print("  " + "-" * 53)
        for fr in sorted(fam_rows, key=lambda x: -float(x.get("avg_quality_score_0_100_success", 0))):
            name = fr.get("family", "?")
            jobs = fr.get("jobs_success", fr.get("jobs_total", "?"))
            pr = float(fr.get("overall_pass_rate_on_success", 0))
            qs = float(fr.get("avg_quality_score_0_100_success", 0))
            print(f"  {name:<25} {jobs:>6} {pr:>7.1%} {qs:>10.1f}")
        print()

    # ── By prompt level
    by_level_path = run_dir / "summary_by_prompt_level.csv"
    if by_level_path.exists():
        lvl_rows = _load_csv(by_level_path)
        print(f"  {'Prompt Level':<25} {'Jobs':>6} {'Pass%':>8} {'Avg Score':>10}")
        print("  " + "-" * 53)
        for lr in lvl_rows:
            name = lr.get("prompt_level", "?")
            jobs = lr.get("jobs_success", lr.get("jobs_total", "?"))
            pr = float(lr.get("overall_pass_rate_on_success", 0))
            qs = float(lr.get("avg_quality_score_0_100_success", 0))
            print(f"  {name:<25} {jobs:>6} {pr:>7.1%} {qs:>10.1f}")
        print()

    # ── Bottom 10 jobs
    ranked = sorted(success, key=lambda x: float(x.get("quality_score_0_100", 0)))
    print(f"  BOTTOM 10 JOBS (lowest quality)")
    print(f"  {'Job Key':<55} {'Score':>7} {'Pass%':>7}")
    print("  " + "-" * 73)
    for r in ranked[:10]:
        jk = r.get("job_key", "?")
        qs = float(r.get("quality_score_0_100", 0))
        pr = float(r.get("pass_rate", 0))
        print(f"  {jk:<55} {qs:>6.1f} {pr:>6.1%}")
    print()

    # ── Top 10 jobs
    print(f"  TOP 10 JOBS (highest quality)")
    print(f"  {'Job Key':<55} {'Score':>7} {'Pass%':>7}")
    print("  " + "-" * 73)
    for r in reversed(ranked[-10:]):
        jk = r.get("job_key", "?")
        qs = float(r.get("quality_score_0_100", 0))
        pr = float(r.get("pass_rate", 0))
        print(f"  {jk:<55} {qs:>6.1f} {pr:>6.1%}")
    print()
    print("=" * 90)


# ─── Subcommand: leaderboard ────────────────────────────────────────────────

def cmd_leaderboard(args: argparse.Namespace) -> None:
    """Load summary_by_model.csv from one or more runs and print a ranked table."""
    run_dirs = [Path(d).resolve() for d in args.run_dirs]

    all_models: Dict[str, Dict[str, Any]] = {}

    for rd in run_dirs:
        model_csv = rd / "summary_by_model.csv"
        if not model_csv.exists():
            print(f"  WARN: No summary_by_model.csv in {rd.name}, skipping")
            continue

        rows = _load_csv(model_csv)
        # Also load per-family and per-level for enrichment
        fam_csv = rd / "summary_by_family.csv"
        lvl_csv = rd / "summary_by_prompt_level.csv"
        results_csv = rd / "results.csv"

        fam_rows = _load_csv(fam_csv) if fam_csv.exists() else []
        lvl_rows = _load_csv(lvl_csv) if lvl_csv.exists() else []
        result_rows = _load_csv(results_csv) if results_csv.exists() else []

        for mr in rows:
            key = f"{mr.get('provider', '?')}/{mr.get('model', '?')}"
            entry = all_models.setdefault(key, {
                "provider": mr.get("provider", "?"),
                "model": mr.get("model", "?"),
                "runs": [],
            })
            avg_score = float(mr.get("avg_quality_score_0_100_success", 0))
            pass_rate = float(mr.get("overall_pass_rate_on_success", 0))
            jobs = int(mr.get("jobs_success", 0))

            # Compute difficulty breakdown from result rows
            easy_scores, med_scores, hard_scores = [], [], []
            for rr in result_rows:
                if rr.get("status") != "success":
                    continue
                qs = float(rr.get("quality_score_0_100", 0))
                fam = rr.get("family", "")
                # Simple heuristic: families with high avg = easy, low avg = hard
                if qs >= 80:
                    easy_scores.append(qs)
                elif qs >= 55:
                    med_scores.append(qs)
                else:
                    hard_scores.append(qs)

            # Per-level delta
            l2_score = 0
            l3_score = 0
            for lr in lvl_rows:
                if lr.get("prompt_level") == "L2":
                    l2_score = float(lr.get("avg_quality_score_0_100_success", 0))
                elif lr.get("prompt_level") == "L3":
                    l3_score = float(lr.get("avg_quality_score_0_100_success", 0))

            entry["runs"].append({
                "run_dir": rd.name,
                "avg_score": avg_score,
                "pass_rate": pass_rate,
                "jobs": jobs,
                "l2_score": l2_score,
                "l3_score": l3_score,
                "easy_pct": (sum(easy_scores) / len(easy_scores)) if easy_scores else 0,
                "med_pct": (sum(med_scores) / len(med_scores)) if med_scores else 0,
                "hard_pct": (sum(hard_scores) / len(hard_scores)) if hard_scores else 0,
            })

    if not all_models:
        print("  No model data found in any run directory.")
        sys.exit(1)

    # Rank by average score across runs
    ranked: List[tuple] = []
    for key, entry in all_models.items():
        avg = sum(r["avg_score"] for r in entry["runs"]) / len(entry["runs"])
        pr = sum(r["pass_rate"] for r in entry["runs"]) / len(entry["runs"])
        jobs_total = sum(r["jobs"] for r in entry["runs"])
        l2 = sum(r["l2_score"] for r in entry["runs"]) / len(entry["runs"])
        l3 = sum(r["l3_score"] for r in entry["runs"]) / len(entry["runs"])
        easy = sum(r["easy_pct"] for r in entry["runs"]) / len(entry["runs"])
        med = sum(r["med_pct"] for r in entry["runs"]) / len(entry["runs"])
        hard = sum(r["hard_pct"] for r in entry["runs"]) / len(entry["runs"])
        ranked.append((key, avg, pr, jobs_total, l2, l3, easy, med, hard))

    ranked.sort(key=lambda x: -x[1])

    print()
    print("=" * 100)
    print("  STEPSCORE LEADERBOARD")
    print("=" * 100)
    print()
    print(f"  {'#':>3}  {'Model':<35} {'Score':>7} {'Pass%':>7} {'Jobs':>6} {'L2':>7} {'L3':>7} {'Easy':>7} {'Med':>7} {'Hard':>7}")
    print("  " + "-" * 96)

    for i, (key, avg, pr, jobs, l2, l3, easy, med, hard) in enumerate(ranked, 1):
        print(
            f"  {i:>3}  {key:<35} {avg:>6.1f} {pr:>6.1%} {jobs:>6}"
            f" {l2:>6.1f} {l3:>6.1f} {easy:>6.1f} {med:>6.1f} {hard:>6.1f}"
        )

    print()
    print("=" * 100)
    print()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> List[Dict[str, str]]:
    """Load a CSV file into a list of dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ─── Argument parser ────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stepscore",
        description="StepScore CLI \u2014 benchmark, score, and analyze CAD generation quality.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run
    p_run = sub.add_parser("run", help="Run the batch harness on a manifest CSV.")
    p_run.add_argument("--manifest", type=Path, required=True, help="CSV manifest path.")
    p_run.add_argument("--run-id", type=str, default="", help="Optional run ID (default: UTC timestamp).")
    p_run.add_argument("--output-root", type=Path, default=None, help="Root output directory.")
    p_run.add_argument("--max-workers", type=int, default=None, help="Concurrent job workers.")
    p_run.add_argument("--resume", action="store_true", help="Skip already-completed jobs.")
    p_run.add_argument("--dry-run", action="store_true", help="Validate without executing.")
    p_run.add_argument("--fast-mode", action="store_true", help="Use fast comparison settings.")
    p_run.add_argument("--limit", type=int, default=0, help="Run only first N jobs.")
    p_run.add_argument("--grading-profile", type=str, default=None, help="Grading profile name.")
    p_run.add_argument("--default-provider", type=str, default=None, help="Fallback LLM provider.")
    p_run.add_argument("--default-model", type=str, default=None, help="Fallback LLM model.")
    p_run.add_argument("--alignment-method", type=str, default=None,
                        choices=["pca_icp", "pca_icp_symmetric", "icp_only", "centroid_only", "none"],
                        help="Alignment method for mesh registration.")
    p_run.set_defaults(func=cmd_run)

    # ── score
    p_score = sub.add_parser("score", help="Score a single reference vs generated STEP pair.")
    p_score.add_argument("--reference", "-r", type=str, required=True, help="Path to reference STEP file.")
    p_score.add_argument("--generated", "-g", type=str, required=True, help="Path to generated STEP file.")
    p_score.add_argument("--grading-profile", type=str, default="full_44", help="Grading profile.")
    p_score.add_argument("--sample-points", type=int, default=30000, help="Surface sample points.")
    p_score.add_argument("--voxel-pitch", type=float, default=1.0, help="Voxel pitch in mm.")
    p_score.add_argument("--fast-mode", action="store_true", default=True, help="Use fast settings (default: on).")
    p_score.add_argument("--no-fast-mode", dest="fast_mode", action="store_false", help="Disable fast mode for full precision.")
    p_score.add_argument("--alignment-method", type=str, default="pca_icp",
                          choices=["pca_icp", "pca_icp_symmetric", "icp_only", "centroid_only", "none"],
                          help="Alignment method for mesh registration (default: pca_icp).")
    p_score.add_argument("--output-json", "-o", type=str, default=None, help="Save full result as JSON.")
    p_score.set_defaults(func=cmd_score)

    # ── report
    p_report = sub.add_parser("report", help="Print a formatted report from a completed run.")
    p_report.add_argument("--run-dir", type=str, required=True, help="Path to completed run directory.")
    p_report.set_defaults(func=cmd_report)

    # ── leaderboard
    p_lb = sub.add_parser("leaderboard", help="Compare models across run directories.")
    p_lb.add_argument("run_dirs", nargs="+", help="One or more run directory paths.")
    p_lb.set_defaults(func=cmd_leaderboard)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
