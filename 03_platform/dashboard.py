#!/usr/bin/env python3
"""
StepScore Results Dashboard — Streamlit app for analyzing harness runs.

Launch:  streamlit run dashboard.py

Six views:
  1. Leaderboard
  2. Per-Family Radar Chart
  3. Failure Taxonomy
  4. Prompt Sensitivity
  5. Metric Importance
  6. Threshold Comparison
"""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Suppress sklearn convergence warnings in the UI
warnings.filterwarnings("ignore", category=UserWarning)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StepScore Dashboard",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────

HARNESS_ROOT = Path(".stepscore_harness_runs")

# Metric categories for failure taxonomy
FAILURE_CATEGORIES = {
    "Rotational Error": [
        "centroid_offset_mm",
        "inertia_tensor_error",
        "normal_angle_error_deg",
    ],
    "Scaling Error": [
        "volume_diff_percent",
        "surface_area_diff_percent",
        "bbox_error_max_mm",
    ],
    "Topological Error": [
        "component_count_match",
        "euler_genus_match",
        "void_hole_count_match",
        "watertight_manifold_pass",
    ],
    "Missing Feature": [
        "feature_count_match",
        "critical_dimension_error_mm",
        "feature_edge_distance_mm",
    ],
    "Surface Deviation": [
        "chamfer_distance_mm",
        "hausdorff_95p_mm",
        "hausdorff_99p_mm",
        "point_to_surface_mean_mm",
    ],
    "Total Failure": [
        "alignment_quality_icp_fitness",
        "registration_failure_rate",
        "voxel_iou",
    ],
}


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def discover_runs() -> List[str]:
    """Return list of run directory names sorted newest first."""
    if not HARNESS_ROOT.is_dir():
        return []
    runs = [d.name for d in HARNESS_ROOT.iterdir() if d.is_dir() and (d / "results.csv").exists()]
    return sorted(runs, reverse=True)


@st.cache_data(ttl=60)
def load_results(run_name: str) -> pd.DataFrame:
    return _load_csv(HARNESS_ROOT / run_name / "results.csv")


@st.cache_data(ttl=60)
def load_family_summary(run_name: str) -> pd.DataFrame:
    p = HARNESS_ROOT / run_name / "summary_by_family.csv"
    return _load_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_level_summary(run_name: str) -> pd.DataFrame:
    p = HARNESS_ROOT / run_name / "summary_by_prompt_level.csv"
    return _load_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_model_summary(run_name: str) -> pd.DataFrame:
    p = HARNESS_ROOT / run_name / "summary_by_model.csv"
    return _load_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=60)
def load_overall_summary(run_name: str) -> Dict[str, Any]:
    p = HARNESS_ROOT / run_name / "summary_overall.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


@st.cache_data(ttl=60)
def load_tuned_thresholds(run_name: str) -> Dict[str, Any]:
    p = HARNESS_ROOT / run_name / "tuning_results" / "threshold_overrides.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


@st.cache_data(ttl=60)
def load_labeled_data(run_name: str) -> pd.DataFrame:
    """Try xlsx first, then csv."""
    xlsx = HARNESS_ROOT / run_name / "labeled_pairs_for_review.xlsx"
    csv_path = HARNESS_ROOT / run_name / "labeled_pairs.csv"
    if xlsx.exists():
        return pd.read_excel(xlsx)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data(ttl=120)
def load_per_metric_data(run_name: str) -> pd.DataFrame:
    """Load all result.json files to get per-metric values for every job."""
    run_dir = HARNESS_ROOT / run_name / "jobs"
    if not run_dir.is_dir():
        return pd.DataFrame()

    rows = []
    for job_dir in run_dir.iterdir():
        if not job_dir.is_dir():
            continue
        rj = job_dir / "result.json"
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text())
        except Exception:
            continue
        compare = data.get("compare", {})
        if not compare.get("ok"):
            continue
        job_info = data.get("job", {})
        row = {
            "job_key": job_info.get("job_key", job_dir.name),
            "family": job_info.get("family", ""),
            "prompt_level": job_info.get("prompt_level", ""),
            "model": job_info.get("model", ""),
            "provider": job_info.get("provider", ""),
        }
        for m in compare.get("metrics", []):
            row[m["name"]] = m["value"]
            row[f"{m['name']}__passed"] = m["passed"]
        summary = compare.get("summary", {})
        row["quality_score_0_100"] = summary.get("quality_score_0_100", 0)
        row["pass_rate"] = summary.get("pass_rate", 0)
        row["overall_pass"] = summary.get("overall_pass", False)
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Default thresholds (imported lazily to avoid heavy deps at import) ───────

def get_default_thresholds() -> Dict[str, Any]:
    """Get default thresholds from compare_thresholds.py."""
    try:
        from compare_thresholds import DEFAULTS
        return dict(DEFAULTS)
    except ImportError:
        return {}


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("StepScore Dashboard")
runs = discover_runs()
if not runs:
    st.error("No harness runs found. Run the harness first, then refresh.")
    st.stop()

selected_run = st.sidebar.selectbox("Select Run", runs, index=0)
overall = load_overall_summary(selected_run)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run:** `{selected_run}`")
st.sidebar.markdown(f"**Jobs:** {overall.get('jobs_total', '?')} total, {overall.get('jobs_success', '?')} success")
if overall.get("elapsed_seconds"):
    st.sidebar.markdown(f"**Elapsed:** {overall['elapsed_seconds']/60:.1f} min")

# View selector
view = st.sidebar.radio(
    "View",
    [
        "Leaderboard",
        "Per-Family Radar",
        "Failure Taxonomy",
        "Prompt Sensitivity",
        "Metric Importance",
        "Threshold Comparison",
    ],
    index=0,
)


# ════════════════════════════════════════════════════════════════════════════
# VIEW 1: LEADERBOARD
# ════════════════════════════════════════════════════════════════════════════

def view_leaderboard():
    st.header("Model Leaderboard")

    results = load_results(selected_run)
    success = results[results["status"] == "success"].copy()
    if success.empty:
        st.warning("No successful jobs in this run.")
        return

    # Overall stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Quality Score", f"{success['quality_score_0_100'].mean():.1f}")
    col2.metric("Avg Pass Rate", f"{success['pass_rate'].mean():.1%}")
    col3.metric("Full Pass", f"{(success['overall_pass'] == True).sum()} / {len(success)}")
    col4.metric("Total Jobs", len(success))

    st.markdown("---")

    # By-model table
    model_df = load_model_summary(selected_run)
    if not model_df.empty:
        st.subheader("By Model")
        display_cols = [c for c in [
            "provider", "model", "jobs_success",
            "overall_pass_rate_on_success", "avg_quality_score_0_100_success",
            "avg_duration_total_s_success"
        ] if c in model_df.columns]
        st.dataframe(
            model_df[display_cols].rename(columns={
                "provider": "Provider",
                "model": "Model",
                "jobs_success": "Jobs",
                "overall_pass_rate_on_success": "Pass Rate",
                "avg_quality_score_0_100_success": "Avg Score",
                "avg_duration_total_s_success": "Avg Time (s)",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # By-family table
    fam_df = load_family_summary(selected_run)
    if not fam_df.empty:
        st.subheader("By Family")
        fam_df_sorted = fam_df.sort_values("avg_quality_score_0_100_success", ascending=False)

        # Bar chart
        fig = px.bar(
            fam_df_sorted,
            x="family",
            y="avg_quality_score_0_100_success",
            color="avg_quality_score_0_100_success",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            labels={"family": "Family", "avg_quality_score_0_100_success": "Avg Score"},
            title="Average Quality Score by Part Family",
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            fam_df_sorted.rename(columns={
                "family": "Family",
                "jobs_success": "Jobs",
                "overall_pass_rate_on_success": "Pass Rate",
                "avg_quality_score_0_100_success": "Avg Score",
                "avg_duration_total_s_success": "Avg Time (s)",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # Distribution of quality scores
    st.subheader("Quality Score Distribution")
    fig_hist = px.histogram(
        success,
        x="quality_score_0_100",
        nbins=30,
        color_discrete_sequence=["#636EFA"],
        labels={"quality_score_0_100": "Quality Score (0-100)"},
        title="Distribution of Quality Scores Across All Jobs",
    )
    fig_hist.update_layout(height=350)
    st.plotly_chart(fig_hist, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# VIEW 2: PER-FAMILY RADAR CHART
# ════════════════════════════════════════════════════════════════════════════

def view_radar():
    st.header("Per-Family Radar Chart")

    metric_df = load_per_metric_data(selected_run)
    if metric_df.empty:
        st.warning("No per-metric data available. Make sure result.json files exist in the jobs directory.")
        return

    # Select metric categories for radar
    radar_metrics = [
        "chamfer_distance_mm",
        "hausdorff_95p_mm",
        "volume_diff_percent",
        "surface_area_diff_percent",
        "bbox_error_max_mm",
        "centroid_offset_mm",
        "feature_count_match",
        "voxel_iou",
        "normal_consistency",
        "composite_weighted_score",
    ]
    available = [m for m in radar_metrics if m in metric_df.columns]
    if not available:
        st.warning("No radar metrics found in result data.")
        return

    families = sorted(metric_df["family"].unique())
    selected_families = st.multiselect(
        "Select families to compare",
        families,
        default=families[:min(4, len(families))],
    )

    if not selected_families:
        st.info("Select at least one family.")
        return

    # Normalize metrics to 0-1 for radar display
    # For "lower_better" metrics, invert so higher = better on radar
    lower_better = {
        "chamfer_distance_mm", "hausdorff_95p_mm", "volume_diff_percent",
        "surface_area_diff_percent", "bbox_error_max_mm", "centroid_offset_mm",
    }

    fig = go.Figure()
    for fam in selected_families:
        fam_data = metric_df[metric_df["family"] == fam]
        values = []
        for m in available:
            raw = fam_data[m].dropna()
            if raw.empty:
                values.append(0)
                continue
            avg_val = raw.mean()
            # Normalize to 0-1 score where 1 = perfect
            if m in lower_better:
                # Use threshold as reference: score = max(0, 1 - val/threshold*2)
                score = max(0, min(1, 1 - avg_val / 10.0))  # rough normalization
            else:
                score = min(1, avg_val)  # already 0-1 ish
            values.append(round(score, 3))

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # close the loop
            theta=[m.replace("_", " ").replace(" mm", "").title() for m in available] + [available[0].replace("_", " ").replace(" mm", "").title()],
            fill="toself",
            name=fam,
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Metric Performance Radar (normalized, higher = better)",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw averages table
    st.subheader("Raw Metric Averages by Family")
    agg = metric_df.groupby("family")[available].mean().round(4)
    st.dataframe(agg.loc[selected_families] if selected_families else agg, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# VIEW 3: FAILURE TAXONOMY
# ════════════════════════════════════════════════════════════════════════════

def view_failure_taxonomy():
    st.header("Failure Taxonomy")
    st.markdown(
        "Classify each failed job by its **dominant failure mode** based on "
        "which metric category has the most failures."
    )

    metric_df = load_per_metric_data(selected_run)
    if metric_df.empty:
        st.warning("No per-metric data available.")
        return

    # For each job, count failures per category
    taxonomy_rows = []
    for _, row in metric_df.iterrows():
        if row.get("overall_pass", False):
            continue  # skip passing jobs

        category_fails = {}
        for cat, metrics in FAILURE_CATEGORIES.items():
            fails = 0
            total = 0
            for m in metrics:
                passed_col = f"{m}__passed"
                if passed_col in row.index and pd.notna(row.get(passed_col)):
                    total += 1
                    if not row[passed_col]:
                        fails += 1
            if total > 0:
                category_fails[cat] = fails / total
            else:
                category_fails[cat] = 0

        # Primary failure mode = category with highest failure fraction
        if category_fails:
            primary = max(category_fails, key=category_fails.get)
            if category_fails[primary] > 0:
                taxonomy_rows.append({
                    "job_key": row.get("job_key", ""),
                    "family": row.get("family", ""),
                    "prompt_level": row.get("prompt_level", ""),
                    "primary_failure": primary,
                    "quality_score": row.get("quality_score_0_100", 0),
                    **{f"{cat}_rate": round(v, 2) for cat, v in category_fails.items()},
                })

    if not taxonomy_rows:
        st.success("No failures to classify!")
        return

    tax_df = pd.DataFrame(taxonomy_rows)

    # Summary pie chart
    col1, col2 = st.columns([1, 1])
    with col1:
        failure_counts = tax_df["primary_failure"].value_counts()
        fig_pie = px.pie(
            values=failure_counts.values,
            names=failure_counts.index,
            title="Primary Failure Mode Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # By family x failure mode heatmap
        cross = pd.crosstab(tax_df["family"], tax_df["primary_failure"])
        fig_heat = px.imshow(
            cross,
            labels=dict(x="Failure Mode", y="Family", color="Count"),
            title="Failure Modes by Part Family",
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

    # Detailed table
    st.subheader("Failure Details")
    st.dataframe(
        tax_df.sort_values("quality_score").head(50),
        use_container_width=True,
        hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# VIEW 4: PROMPT SENSITIVITY
# ════════════════════════════════════════════════════════════════════════════

def view_prompt_sensitivity():
    st.header("Prompt Sensitivity: L2 vs L3")
    st.markdown(
        "Compare how much detail level affects quality. "
        "L2 = structured parameters, L3 = full CadQuery-style prompt."
    )

    results = load_results(selected_run)
    success = results[results["status"] == "success"].copy()
    if success.empty:
        st.warning("No successful jobs.")
        return

    if "prompt_level" not in success.columns:
        st.warning("No prompt_level column in results.")
        return

    # Overall L2 vs L3
    level_summary = load_level_summary(selected_run)
    if not level_summary.empty:
        col1, col2, col3 = st.columns(3)
        for _, lr in level_summary.iterrows():
            level = lr.get("prompt_level", "?")
            score = float(lr.get("avg_quality_score_0_100_success", 0))
            jobs = int(lr.get("jobs_success", 0))
            if level == "L2":
                col1.metric(f"{level} Avg Score", f"{score:.1f}")
            elif level == "L3":
                col2.metric(f"{level} Avg Score", f"{score:.1f}")
        if len(level_summary) >= 2:
            l2_score = float(level_summary[level_summary["prompt_level"] == "L2"]["avg_quality_score_0_100_success"].iloc[0]) if "L2" in level_summary["prompt_level"].values else 0
            l3_score = float(level_summary[level_summary["prompt_level"] == "L3"]["avg_quality_score_0_100_success"].iloc[0]) if "L3" in level_summary["prompt_level"].values else 0
            delta = l3_score - l2_score
            col3.metric("L3 - L2 Delta", f"{delta:+.1f}")

    st.markdown("---")

    # Per-family L2 vs L3
    st.subheader("Per-Family Prompt Sensitivity")
    fam_level = success.groupby(["family", "prompt_level"])["quality_score_0_100"].mean().unstack(fill_value=0)

    if "L2" in fam_level.columns and "L3" in fam_level.columns:
        fam_level["delta"] = fam_level["L3"] - fam_level["L2"]
        fam_level = fam_level.sort_values("delta", ascending=False)

        # Grouped bar chart
        plot_df = fam_level.reset_index().melt(
            id_vars="family",
            value_vars=["L2", "L3"],
            var_name="Prompt Level",
            value_name="Avg Quality Score",
        )
        fig = px.bar(
            plot_df,
            x="family",
            y="Avg Quality Score",
            color="Prompt Level",
            barmode="group",
            title="Quality Score by Family and Prompt Level",
            color_discrete_map={"L2": "#EF553B", "L3": "#636EFA"},
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Delta chart
        fig_delta = px.bar(
            fam_level.reset_index(),
            x="family",
            y="delta",
            color="delta",
            color_continuous_scale="RdYlGn",
            title="L3 - L2 Quality Delta by Family (positive = L3 better)",
            labels={"delta": "Delta", "family": "Family"},
        )
        fig_delta.update_layout(height=350)
        st.plotly_chart(fig_delta, use_container_width=True)

        # Table
        st.dataframe(fam_level.round(2), use_container_width=True)
    else:
        st.info("Need both L2 and L3 prompt levels for comparison.")

    # Per-job scatter: L2 vs L3 paired by part_id
    st.subheader("Paired Part Comparison")
    pivot = success.pivot_table(
        index="part_id",
        columns="prompt_level",
        values="quality_score_0_100",
        aggfunc="first",
    )
    if "L2" in pivot.columns and "L3" in pivot.columns:
        pivot = pivot.dropna()
        fig_scatter = px.scatter(
            pivot.reset_index(),
            x="L2",
            y="L3",
            hover_data=["part_id"] if "part_id" in pivot.reset_index().columns else None,
            title="L2 vs L3 Quality Score (each dot = one part)",
            labels={"L2": "L2 Score", "L3": "L3 Score"},
        )
        fig_scatter.add_shape(
            type="line", x0=0, y0=0, x1=100, y1=100,
            line=dict(dash="dash", color="gray"),
        )
        fig_scatter.update_layout(height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# VIEW 5: METRIC IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════

def view_metric_importance():
    st.header("Metric Importance")
    st.markdown(
        "Train a logistic regression on human-labeled data to identify "
        "which metrics best predict positive/negative labels."
    )

    labeled = load_labeled_data(selected_run)
    if labeled.empty:
        st.warning("No labeled data found. Need labeled_pairs_for_review.xlsx or labeled_pairs.csv.")
        return

    if "label" not in labeled.columns:
        st.warning("No 'label' column in labeled data.")
        return

    # Prepare features
    y = (labeled["label"] == "positive").astype(int).values

    # Get numeric columns that look like metrics
    exclude = {"label", "pair_id", "model", "prompt_level", "family", "overall_pass", "job_key"}
    feature_cols = [c for c in labeled.columns if c not in exclude and labeled[c].dtype in ["float64", "int64", "float32"]]

    if len(feature_cols) < 3:
        st.warning(f"Only {len(feature_cols)} numeric features found. Need more for meaningful analysis.")
        return

    X = labeled[feature_cols].copy()
    # Fill NaN with column median
    X = X.fillna(X.median())

    st.info(f"Training on {len(y)} samples ({y.sum()} positive, {(1-y).sum()} negative) with {len(feature_cols)} features")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(
            C=1.0,
            penalty="l1",
            solver="saga",
            max_iter=5000,
            random_state=42,
        )
        model.fit(X_scaled, y)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="f1")
        st.metric("5-Fold CV F1 Score", f"{cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        # Coefficients
        coef_df = pd.DataFrame({
            "metric": feature_cols,
            "coefficient": model.coef_[0],
            "abs_coefficient": np.abs(model.coef_[0]),
        }).sort_values("abs_coefficient", ascending=False)

        # Filter out zero coefficients (L1 pruned)
        nonzero = coef_df[coef_df["abs_coefficient"] > 1e-6].copy()
        if nonzero.empty:
            st.warning("All coefficients are zero. Need more training data.")
            return

        st.subheader(f"Top {len(nonzero)} Non-Zero Coefficients (L1 selected)")

        # Horizontal bar chart
        fig = px.bar(
            nonzero.head(20),
            x="coefficient",
            y="metric",
            orientation="h",
            color="coefficient",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Logistic Regression Coefficients (positive = predicts PASS)",
            labels={"coefficient": "Coefficient", "metric": "Metric"},
        )
        fig.update_layout(height=max(400, len(nonzero.head(20)) * 28), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        # Full table
        st.subheader("All Coefficients")
        st.dataframe(
            coef_df.rename(columns={
                "metric": "Metric",
                "coefficient": "Coefficient",
                "abs_coefficient": "|Coefficient|",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Interpretation
        st.markdown("---")
        st.subheader("Interpretation")
        top_positive = nonzero[nonzero["coefficient"] > 0].head(3)
        top_negative = nonzero[nonzero["coefficient"] < 0].head(3)

        if not top_positive.empty:
            st.markdown("**Most predictive of PASS (positive coefficient):**")
            for _, r in top_positive.iterrows():
                st.markdown(f"- `{r['metric']}` (coef={r['coefficient']:.3f})")

        if not top_negative.empty:
            st.markdown("**Most predictive of FAIL (negative coefficient):**")
            for _, r in top_negative.iterrows():
                st.markdown(f"- `{r['metric']}` (coef={r['coefficient']:.3f})")

    except ImportError:
        st.error(
            "scikit-learn is required for this view. "
            "Install it with: `pip install scikit-learn`"
        )


# ════════════════════════════════════════════════════════════════════════════
# VIEW 6: THRESHOLD COMPARISON
# ════════════════════════════════════════════════════════════════════════════

def view_threshold_comparison():
    st.header("Threshold Comparison: Default vs Tuned")

    defaults = get_default_thresholds()
    tuned = load_tuned_thresholds(selected_run)

    if not defaults:
        st.warning("Could not load default thresholds from compare_thresholds.py")
        return

    if not tuned:
        st.warning("No tuned thresholds found for this run.")
        return

    # Load labeled data for F1 evaluation
    labeled = load_labeled_data(selected_run)
    has_labels = not labeled.empty and "label" in labeled.columns

    if has_labels:
        y_true = (labeled["label"] == "positive").astype(int).values

    # Build comparison table
    all_metrics = sorted(set(list(defaults.keys()) | set(tuned.keys())))
    comp_rows = []

    for metric_name in all_metrics:
        row: Dict[str, Any] = {"metric": metric_name}

        d = defaults.get(metric_name)
        t = tuned.get(metric_name)

        if d:
            row["default_threshold"] = d["threshold"]
            row["default_direction"] = d["direction"]
        if t:
            row["tuned_threshold"] = t["threshold"]
            row["tuned_direction"] = t["direction"]
            row["tuned_f1"] = t.get("f1", None)
            row["tuned_precision"] = t.get("precision", None)
            row["tuned_recall"] = t.get("recall", None)

        if d and t:
            row["threshold_delta"] = t["threshold"] - d["threshold"]
        comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows)

    # Summary metrics
    if "tuned_f1" in comp_df.columns:
        avg_f1 = comp_df["tuned_f1"].dropna().mean()
        st.metric("Average Tuned F1 Score", f"{avg_f1:.3f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Default Thresholds", len(defaults))
    col2.metric("Tuned Thresholds", len(tuned))
    col3.metric("Total Metrics", len(all_metrics))

    st.markdown("---")

    # F1 bar chart for tuned thresholds
    if "tuned_f1" in comp_df.columns:
        st.subheader("Tuned Threshold F1 Scores")
        f1_df = comp_df.dropna(subset=["tuned_f1"]).sort_values("tuned_f1", ascending=True)
        fig = px.bar(
            f1_df,
            x="tuned_f1",
            y="metric",
            orientation="h",
            color="tuned_f1",
            color_continuous_scale="RdYlGn",
            range_color=[0.5, 1.0],
            title="F1 Score per Metric (tuned thresholds)",
            labels={"tuned_f1": "F1 Score", "metric": "Metric"},
        )
        fig.update_layout(height=max(500, len(f1_df) * 18), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Threshold delta chart
    if "threshold_delta" in comp_df.columns:
        st.subheader("Threshold Changes (Tuned - Default)")
        delta_df = comp_df.dropna(subset=["threshold_delta"]).copy()
        delta_df["abs_delta"] = delta_df["threshold_delta"].abs()
        delta_df = delta_df.sort_values("abs_delta", ascending=False).head(25)

        fig_delta = px.bar(
            delta_df,
            x="threshold_delta",
            y="metric",
            orientation="h",
            color="threshold_delta",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title="Threshold Delta (Tuned - Default)",
            labels={"threshold_delta": "Delta", "metric": "Metric"},
        )
        fig_delta.update_layout(height=max(400, len(delta_df) * 22))
        st.plotly_chart(fig_delta, use_container_width=True)

    # Full comparison table
    st.subheader("Full Comparison Table")
    display_cols = [c for c in [
        "metric", "default_threshold", "tuned_threshold",
        "threshold_delta", "default_direction",
        "tuned_f1", "tuned_precision", "tuned_recall",
    ] if c in comp_df.columns]

    st.dataframe(
        comp_df[display_cols].sort_values("metric"),
        use_container_width=True,
        hide_index=True,
    )


# ── Router ───────────────────────────────────────────────────────────────────

VIEWS = {
    "Leaderboard": view_leaderboard,
    "Per-Family Radar": view_radar,
    "Failure Taxonomy": view_failure_taxonomy,
    "Prompt Sensitivity": view_prompt_sensitivity,
    "Metric Importance": view_metric_importance,
    "Threshold Comparison": view_threshold_comparison,
}

VIEWS[view]()
