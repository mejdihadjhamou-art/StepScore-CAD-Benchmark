from __future__ import annotations

import json
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import trimesh

from generation_pipeline import GenerationError, generate_and_export
from metric_engine import (
    ALIGNMENT_METHODS,
    DEFAULT_ALIGNMENT_METHOD,
    DEFAULT_THRESHOLDS,
    GRADING_PROFILE_FULL_44,
    compare_models,
    get_grading_profiles,
)
from prompt_linter import lint_prompt
from step_qa import StepQAError, answer_step_question
from step_utils import STEP_EXTENSIONS, SUPPORTED_EXTENSIONS, StepConversionError, ensure_mesh_path

# ── Threshold sets: default vs tuned ──────────────────────────────
def _load_tuned_thresholds() -> tuple[dict, str]:
    # 1) explicit env override
    env_path = os.getenv("STEPSCORE_TUNED_THRESHOLDS", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))

    # 2) latest tuning outputs (prefer cleaned if present)
    candidates.append(Path("tuning_output_global/threshold_overrides_cleaned.json"))
    candidates.append(Path("tuning_output_global/threshold_overrides.json"))
    candidates.append(Path("threshold_tuning/output_global/threshold_overrides_cleaned.json"))
    candidates.append(Path("threshold_tuning/output_global/threshold_overrides.json"))

    # 3) legacy location (previous runs)
    candidates.append(
        Path(".stepscore_harness_runs/final73_anthropic_run_02/tuning_results/threshold_overrides.json")
    )

    for p in candidates:
        try:
            if not p.exists():
                continue
            raw = json.loads(p.read_text())
            # Accept either {metric: value} or {metric: {threshold: value}}
            if isinstance(raw, dict):
                if raw and isinstance(next(iter(raw.values())), dict) and "threshold" in next(iter(raw.values())):
                    tuned = {k: v.get("threshold") for k, v in raw.items() if "threshold" in v}
                else:
                    tuned = {k: v for k, v in raw.items() if isinstance(v, (int, float))}
                if tuned:
                    return tuned, str(p)
        except Exception:
            continue
    return {}, ""


TUNED_THRESHOLDS, _TUNED_THRESHOLDS_SOURCE = _load_tuned_thresholds()

THRESHOLD_SETS = {
    "Set 1 – Original defaults": {
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "description": "Hand-crafted default thresholds shipped with StepScore.",
    },
}
if TUNED_THRESHOLDS:
    THRESHOLD_SETS["Set 2 – Tuned (human-labeled)"] = {
        "thresholds": TUNED_THRESHOLDS,
        "description": (
            "Calibrated thresholds loaded from your tuning output "
            f"({ _TUNED_THRESHOLDS_SOURCE or 'unknown source' })."
        ),
    }

TASK_GENERATE = "generate"
TASK_MODIFY = "modify"
TASK_QA = "qa"
OPENAI_DEFAULT_MODEL = "gpt-5.2"
ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-1-20250805"
GENERATE_STACK_GPT = "ChatGPT 5.2 (OpenAI)"
GENERATE_STACK_CLAUDE = "Claude Opus (Anthropic)"
ANTHROPIC_MODEL_OPTIONS = [
    ("Claude Opus Latest (alias)", "claude-opus-latest"),
    ("Claude Opus 4.1 (20250805)", "claude-opus-4-1-20250805"),
    ("Custom model ID", ""),
]
GRADING_PROFILES = get_grading_profiles()


def _task_label(mode: str) -> str:
    labels = {
        TASK_GENERATE: "1) Generate STEP from Prompt",
        TASK_MODIFY: "2) Modify Existing STEP",
        TASK_QA: "3) Ask Question About STEP",
    }
    return labels.get(mode, mode)


def _anthropic_model_widget(choice_key: str, custom_key: str) -> str:
    labels = [label for label, _ in ANTHROPIC_MODEL_OPTIONS]
    id_by_label = {label: model_id for label, model_id in ANTHROPIC_MODEL_OPTIONS}
    default_index = 0
    for i, (_, model_id) in enumerate(ANTHROPIC_MODEL_OPTIONS):
        if model_id == ANTHROPIC_DEFAULT_MODEL:
            default_index = i
            break

    selection = st.selectbox(
        "Anthropic model",
        options=labels,
        index=default_index,
        key=choice_key,
        help=(
            "Pick a preset or select Custom model ID. "
            "If your account doesn't support aliases, use the exact model ID."
        ),
    )
    model_id = id_by_label.get(selection, "")
    if model_id:
        return model_id
    return st.text_input(
        "Anthropic model ID",
        key=custom_key,
        help="Example: claude-opus-4-1-20250805",
    ).strip()


def _build_structured_prompt(
    part_goal: str,
    units_and_frame: str,
    geometry: str,
    additive_features: str,
    subtractive_features: str,
    constraints: str,
    topology_output: str,
    notes: str,
) -> str:
    sections = [
        ("Part goal", part_goal),
        ("Units and frame", units_and_frame),
        ("Geometry", geometry),
        ("Additive features", additive_features),
        ("Subtractive features", subtractive_features),
        ("Constraints", constraints),
        ("Topology/output constraints", topology_output),
    ]
    lines = []
    for title, value in sections:
        value_norm = value.strip() or "(not specified)"
        lines.append(f"{title}:")
        lines.append(value_norm)
        lines.append("")
    if notes.strip():
        lines.append("Additional notes:")
        lines.append(notes.strip())
        lines.append("")
    return "\n".join(lines).strip()


st.set_page_config(page_title="STEPScore", layout="wide")
st.title("STEPScore")
st.caption(
    "Three workflows: generate STEP, modify existing STEP, or ask questions about STEP geometry. "
    "STEP/STP files are auto-converted to STL where mesh metrics are required."
)

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "last_run_dir" not in st.session_state:
    st.session_state.last_run_dir = None
if "last_generation" not in st.session_state:
    st.session_state.last_generation = None
if "last_qa" not in st.session_state:
    st.session_state.last_qa = None
if "last_task_mode" not in st.session_state:
    st.session_state.last_task_mode = None
if "last_compare_mesh_paths" not in st.session_state:
    st.session_state.last_compare_mesh_paths = None
if "gen_model" not in st.session_state:
    st.session_state.gen_model = OPENAI_DEFAULT_MODEL
if "gen_generate_stack" not in st.session_state:
    st.session_state.gen_generate_stack = GENERATE_STACK_GPT
if "gen_model_claude" not in st.session_state:
    st.session_state.gen_model_claude = ANTHROPIC_DEFAULT_MODEL
if "qa_model" not in st.session_state:
    st.session_state.qa_model = OPENAI_DEFAULT_MODEL
if "grading_profile" not in st.session_state:
    st.session_state.grading_profile = GRADING_PROFILE_FULL_44


@st.cache_data(show_spinner=False)
def _save_uploaded(uploaded, target_name: str) -> str:
    out_dir = Path(".cad42_uploads")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / target_name
    path.write_bytes(uploaded.getbuffer())
    return str(path)


def _new_run_dir() -> Path:
    root = Path(".cad42_runs")
    root.mkdir(exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = root / run_id
    run_dir.mkdir(exist_ok=True)
    return run_dir


def _check_supported(path: str) -> None:
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )


def _check_step_only(path: str) -> None:
    ext = Path(path).suffix.lower()
    if ext not in STEP_EXTENSIONS:
        raise ValueError(f"Expected STEP/STP file. Got '{ext}'.")


def _load_mesh_for_view(path: str, max_faces: int = 12000) -> tuple[np.ndarray, np.ndarray, list[float]]:
    loaded = trimesh.load(path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded

    if mesh is None or mesh.is_empty:
        raise ValueError(f"Could not load a valid mesh from: {path}")

    face_count = int(len(mesh.faces))
    if face_count > max_faces:
        selected = np.linspace(0, face_count - 1, num=max_faces, dtype=int)
        mesh = mesh.submesh([selected], append=True, repair=False)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    dims = (mesh.bounds[1] - mesh.bounds[0]).astype(float).tolist()
    return vertices, faces, dims


def _mesh_to_plotly_figure(vertices: np.ndarray, faces: np.ndarray, title: str):
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError("plotly is required for 3D viewer. Install with: pip install plotly") from exc

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#7aa6ff",
                opacity=1.0,
                flatshading=True,
                lighting={
                    "ambient": 0.45,
                    "diffuse": 0.7,
                    "specular": 0.15,
                    "roughness": 0.9,
                },
                lightposition={"x": 100, "y": 100, "z": 200},
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=430,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "camera": {"eye": {"x": 1.5, "y": 1.4, "z": 1.2}},
        },
    )
    return fig


def _render_compare_3d_views(reference_mesh_path: str, generated_mesh_path: str) -> None:
    st.subheader("3D Views")
    c_ref, c_gen = st.columns(2)

    with c_ref:
        st.markdown("**Reference Model**")
        st.caption(reference_mesh_path)
        try:
            ref_v, ref_f, ref_dims = _load_mesh_for_view(reference_mesh_path)
            st.plotly_chart(
                _mesh_to_plotly_figure(ref_v, ref_f, "Reference"),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                f"BBox (mm): X={ref_dims[0]:.2f}, Y={ref_dims[1]:.2f}, Z={ref_dims[2]:.2f}"
            )
        except Exception as exc:
            st.warning(f"Could not render reference model: {exc}")

    with c_gen:
        st.markdown("**Generated Model**")
        st.caption(generated_mesh_path)
        try:
            gen_v, gen_f, gen_dims = _load_mesh_for_view(generated_mesh_path)
            st.plotly_chart(
                _mesh_to_plotly_figure(gen_v, gen_f, "Generated"),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.caption(
                f"BBox (mm): X={gen_dims[0]:.2f}, Y={gen_dims[1]:.2f}, Z={gen_dims[2]:.2f}"
            )
        except Exception as exc:
            st.warning(f"Could not render generated model: {exc}")


def _render_compare_result(result: dict) -> None:
    summary = result["summary"]
    details = result["details"]
    metrics_df = pd.DataFrame(result["metrics"])

    profile_label = summary.get("grading_profile_label", "Full 44 Metrics")
    profile_desc = summary.get("grading_profile_description", "")
    total_computed = int(summary.get("total_metrics_computed", summary["total_metrics"]))
    total_scored = int(summary["total_metrics"])

    st.subheader("Grading profile")
    st.markdown(f"**{profile_label}**")
    if profile_desc:
        st.caption(profile_desc)
    st.caption(f"Scored metrics: {total_scored} | Computed metrics: {total_computed}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scored metrics", total_scored)
    c2.metric("Pass count", int(summary["pass_count"]))
    c3.metric("Fail count", int(summary["fail_count"]))
    c4.metric("Pass rate", f"{summary['pass_rate'] * 100:.1f}%")
    c5.metric("Quality score", f"{summary.get('quality_score_0_100', 0.0):.1f}/100")

    st.subheader("Overall verdict")
    if summary["overall_pass"]:
        st.success(f"PASS: all {total_scored} scored metrics passed configured thresholds.")
    else:
        st.error(f"FAIL: one or more of the {total_scored} scored metrics failed configured thresholds.")

    st.subheader("Metric table")
    show_only_scored = False
    if "in_selected_profile" in metrics_df.columns:
        show_only_scored = st.checkbox("Show only scored metrics", value=False)
        if show_only_scored:
            metrics_df = metrics_df[metrics_df["in_selected_profile"] == True].copy()  # noqa: E712
        metrics_df = metrics_df.sort_values(
            ["in_selected_profile", "passed", "name"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    else:
        metrics_df = metrics_df.sort_values(["passed", "name"], ascending=[True, True]).reset_index(drop=True)
    st.dataframe(metrics_df, use_container_width=True, height=520)

    st.subheader("Pass/fail distribution")
    dist_source = metrics_df
    if "in_selected_profile" in metrics_df.columns and not show_only_scored:
        dist_source = metrics_df[metrics_df["in_selected_profile"] == True]  # noqa: E712
    dist = dist_source["passed"].value_counts().rename(index={True: "Pass", False: "Fail"})
    st.bar_chart(dist)

    st.subheader("Metric values vs thresholds")
    chart_df = metrics_df.copy()
    chart_df["value"] = pd.to_numeric(chart_df["value"], errors="coerce")
    chart_df["threshold"] = pd.to_numeric(chart_df["threshold"], errors="coerce")
    st.line_chart(chart_df.set_index("name")[["value", "threshold"]])

    st.subheader("Run details")
    st.json(details)

    st.download_button(
        "Download result JSON",
        data=json.dumps(result, indent=2),
        file_name="cad42_result.json",
        mime="application/json",
    )


def _render_qa_result(result: dict) -> None:
    st.subheader("Answer")
    st.success(result.get("answer", "No answer"))
    gtfa_eval = result.get("gtfa_evaluation") or {}
    if gtfa_eval.get("enabled"):
        st.subheader("GTFA Comparison")
        if gtfa_eval.get("passed"):
            st.success("PASS: model answer matches GTFA criteria.")
        else:
            st.error("FAIL: model answer does not match GTFA criteria.")
        st.json(gtfa_eval)
    st.subheader("Extracted facts")
    st.json(result.get("facts", {}))
    analysis = result.get("analysis") or {}
    if analysis:
        st.subheader("Analysis details")
        st.json(analysis)

    st.download_button(
        "Download Q&A JSON",
        data=json.dumps(result, indent=2),
        file_name="cad42_step_qa_result.json",
        mime="application/json",
    )


with st.sidebar:
    st.header("Workflow")
    task_mode = st.selectbox(
        "Task type",
        options=[TASK_GENERATE, TASK_MODIFY, TASK_QA],
        format_func=_task_label,
    )
    upload_types = [ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)]

    run_btn = False

    ref_mode = "Upload"
    ref_file = None
    ref_path_text = ""

    base_mode = "Upload"
    base_file = None
    base_path_text = ""

    qa_mode = "Upload"
    qa_file = None
    qa_path_text = ""

    ai_provider = "openai"
    ai_model = OPENAI_DEFAULT_MODEL
    generate_stack = GENERATE_STACK_GPT
    ai_api_key = ""
    ai_prompt = ""
    prompt_input_mode = "Raw Prompt"
    prompt_builder_fields = {}
    prompt_builder_collated = ""
    qa_question = ""
    qa_use_ai = False
    qa_provider = "openai"
    qa_model = OPENAI_DEFAULT_MODEL
    qa_api_key = ""
    qa_gtfa = ""
    qa_gtfa_tol = 2.0

    fast_mode = True
    sample_points = 10000
    voxel_pitch = 2.0
    grading_profile = GRADING_PROFILE_FULL_44
    threshold_blob = "{}"

    if task_mode in {TASK_GENERATE, TASK_MODIFY}:
        st.header("Reference (Target)")
        ref_mode = st.radio("Reference source", ["Upload", "Path"], horizontal=True, key="ref_source_mode")
        ref_file = (
            st.file_uploader("Reference model", type=upload_types, key="ref_file_upload")
            if ref_mode == "Upload"
            else None
        )
        ref_path_text = st.text_input("Reference path", key="ref_path_input") if ref_mode == "Path" else ""

        if task_mode == TASK_MODIFY:
            st.header("Base Model To Modify")
            base_mode = st.radio("Base STEP source", ["Upload", "Path"], horizontal=True, key="base_source_mode")
            base_file = (
                st.file_uploader("Base STEP", type=["step", "stp"], key="base_file_upload")
                if base_mode == "Upload"
                else None
            )
            base_path_text = st.text_input("Base STEP path", key="base_path_input") if base_mode == "Path" else ""

        st.header("AI Generation")
        if task_mode == TASK_GENERATE:
            generate_stack = st.selectbox(
                "Generation model",
                options=[GENERATE_STACK_GPT, GENERATE_STACK_CLAUDE],
                key="gen_generate_stack",
            )
            if generate_stack == GENERATE_STACK_GPT:
                ai_provider = "openai"
                ai_model = OPENAI_DEFAULT_MODEL
                st.caption(f"Provider: `{ai_provider}`")
                st.caption(f"Model: `{ai_model}`")
            else:
                ai_provider = "anthropic"
                ai_model = _anthropic_model_widget(
                    choice_key="gen_model_claude_choice",
                    custom_key="gen_model_claude_custom",
                )
                st.caption(f"Provider: `{ai_provider}`")
                st.caption(f"Model: `{ai_model}`")
        else:
            ai_provider = st.selectbox("Provider", options=["openai", "anthropic"], index=0, key="gen_provider")
            if ai_provider == "anthropic":
                ai_model = _anthropic_model_widget(
                    choice_key="gen_model_anthropic_choice",
                    custom_key="gen_model_anthropic_custom",
                )
            else:
                ai_model = st.text_input("Model", key="gen_model")
        api_key_label = "OpenAI API key (optional, else env var)"
        if ai_provider == "anthropic":
            api_key_label = "Anthropic API key (optional, else env var)"
        ai_api_key = st.text_input(api_key_label, type="password", key="gen_api_key")
        if task_mode == TASK_GENERATE:
            prompt_input_mode = st.radio(
                "Prompt input style",
                options=["Step Builder", "Raw Prompt"],
                horizontal=True,
                key="gen_prompt_input_mode",
            )
            if prompt_input_mode == "Step Builder":
                part_goal = st.text_area(
                    "Part goal",
                    height=90,
                    key="gen_sb_part_goal",
                    placeholder="What part should be created and its intended purpose.",
                )
                units_and_frame = st.text_area(
                    "Units and frame",
                    height=90,
                    key="gen_sb_units_frame",
                    placeholder="Units (mm), origin, axis orientation, coordinate frame assumptions.",
                )
                geometry = st.text_area(
                    "Geometry",
                    height=130,
                    key="gen_sb_geometry",
                    placeholder="Core primitives, dimensions, positions, key relationships.",
                )
                additive_features = st.text_area(
                    "Additive features",
                    height=100,
                    key="gen_sb_additive",
                    placeholder="Bosses, ribs, handles, protrusions, fillets to add.",
                )
                subtractive_features = st.text_area(
                    "Subtractive features",
                    height=100,
                    key="gen_sb_subtractive",
                    placeholder="Holes, pockets, cutouts, chamfers to remove material.",
                )
                constraints = st.text_area(
                    "Constraints",
                    height=100,
                    key="gen_sb_constraints",
                    placeholder="Symmetry, alignments, fixed dimensions, clearances, intersections.",
                )
                topology_output = st.text_area(
                    "Topology/output constraints",
                    height=90,
                    key="gen_sb_topology",
                    placeholder="Single connected solid, no floating parts, manifold requirements.",
                )
                notes = st.text_area(
                    "Additional notes (optional)",
                    height=80,
                    key="gen_sb_notes",
                    placeholder="Any extra rules or assumptions.",
                )
                prompt_builder_fields = {
                    "part_goal": part_goal,
                    "units_and_frame": units_and_frame,
                    "geometry": geometry,
                    "additive_features": additive_features,
                    "subtractive_features": subtractive_features,
                    "constraints": constraints,
                    "topology_output": topology_output,
                    "notes": notes,
                }
                prompt_builder_collated = _build_structured_prompt(
                    part_goal=part_goal,
                    units_and_frame=units_and_frame,
                    geometry=geometry,
                    additive_features=additive_features,
                    subtractive_features=subtractive_features,
                    constraints=constraints,
                    topology_output=topology_output,
                    notes=notes,
                )
                ai_prompt = prompt_builder_collated
                st.session_state["gen_sb_collated_preview"] = prompt_builder_collated
                st.text_area(
                    "Collated prompt (auto-generated)",
                    height=220,
                    key="gen_sb_collated_preview",
                    disabled=True,
                    help="This is the exact prompt that will be sent to the model.",
                )
            else:
                ai_prompt = st.text_area(
                    "Prompt",
                    height=240,
                    key="gen_prompt",
                    placeholder=(
                        "Create a single connected mechanical CAD part.\n"
                        "Part identity: ...\n"
                        "Units and frame: ...\n"
                        "Geometry: ...\n"
                        "Additive features: ...\n"
                        "Subtractive features: ...\n"
                        "Constraints: ...\n"
                        "Topology/output constraints: ..."
                    ),
                )
        else:
            ai_prompt = st.text_area(
                "Modification prompt",
                height=240,
                key="gen_prompt_modify",
                placeholder=(
                    "Modify the base STEP model according to:\n"
                    "- dimensional changes\n"
                    "- added/removed features\n"
                    "- constraints to preserve\n"
                    "- output requirements"
                ),
            )

        st.subheader("Prompt Lint")
        enforce_lint = st.checkbox(
            "Enforce prompt lint (block generation on errors)",
            value=True,
            key="enforce_prompt_lint",
        )
        lint_result = lint_prompt(ai_prompt if task_mode == TASK_GENERATE else ai_prompt)
        if lint_result.errors:
            st.error("Prompt lint errors:\n- " + "\n- ".join(lint_result.errors))
        if lint_result.warnings:
            st.warning("Prompt lint warnings:\n- " + "\n- ".join(lint_result.warnings))

        st.header("Evaluation Settings")
        fast_mode = st.checkbox("Fast mode (recommended)", value=True, key="fast_mode")
        sample_points = st.slider(
            "Sample points",
            min_value=5000,
            max_value=80000,
            value=10000,
            step=5000,
            key="sample_points",
        )
        voxel_pitch = st.number_input(
            "Voxel pitch (mm)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            key="voxel_pitch",
        )
        grading_profile = st.selectbox(
            "Grading system",
            options=list(GRADING_PROFILES.keys()),
            format_func=lambda k: str(GRADING_PROFILES[k]["label"]),
            key="grading_profile",
            help="Choose which metric set determines pass/fail and summary scores.",
        )
        st.caption(str(GRADING_PROFILES[grading_profile].get("description", "")))

        alignment_method = st.selectbox(
            "Alignment method",
            options=list(ALIGNMENT_METHODS),
            index=0,
            key="alignment_method",
            help="How to align the generated mesh to the reference before computing metrics.",
        )

        # ── Threshold set picker ──
        st.divider()
        st.subheader("Threshold Set")
        selected_set = st.radio(
            "Choose threshold calibration",
            options=list(THRESHOLD_SETS.keys()),
            index=0,
            key="threshold_set_choice",
            help="Pick a threshold set to use for pass/fail scoring.",
        )
        set_info = THRESHOLD_SETS[selected_set]
        st.caption(set_info["description"])

        # Show side-by-side comparison of the two sets
        if len(THRESHOLD_SETS) > 1:
            with st.expander("Compare threshold sets side-by-side"):
                set_names = list(THRESHOLD_SETS.keys())
                s1 = THRESHOLD_SETS[set_names[0]]["thresholds"]
                s2 = THRESHOLD_SETS[set_names[1]]["thresholds"]
                all_keys = sorted(set(list(s1.keys()) + list(s2.keys())))
                comp_rows = []
                for k in all_keys:
                    v1 = s1.get(k)
                    v2 = s2.get(k)
                    if v1 is not None and v2 is not None:
                        try:
                            delta = float(v2) - float(v1)
                            change = f"{delta:+.4f}"
                        except (TypeError, ValueError):
                            change = "–"
                    else:
                        change = "NEW" if v2 is not None else "REMOVED"
                    comp_rows.append({
                        "metric": k,
                        set_names[0]: f"{v1}" if v1 is not None else "–",
                        set_names[1]: f"{v2}" if v2 is not None else "–",
                        "change": change,
                    })
                st.dataframe(
                    pd.DataFrame(comp_rows),
                    use_container_width=True,
                    height=400,
                )

        # Pre-fill overrides from selected set (only non-default values)
        prefill = json.dumps(set_info["thresholds"], indent=2) if selected_set != list(THRESHOLD_SETS.keys())[0] else "{}"
        st.divider()
        threshold_blob = st.text_area(
            "Threshold overrides JSON (auto-filled from set above, editable)",
            value=prefill,
            height=120,
            key="threshold_overrides",
            help='Editing here takes priority. Clear to use pure defaults.',
        )
        run_btn = st.button("Run Generation + Compare", type="primary", key="run_compare")

    else:
        st.header("STEP Q&A")
        qa_mode = st.radio("STEP source", ["Upload", "Path"], horizontal=True, key="qa_source_mode")
        qa_file = (
            st.file_uploader("STEP model", type=["step", "stp"], key="qa_file_upload")
            if qa_mode == "Upload"
            else None
        )
        qa_path_text = st.text_input("STEP path", key="qa_path_input") if qa_mode == "Path" else ""
        qa_question = st.text_area(
            "Question",
            height=120,
            key="qa_question",
            placeholder='Example: "How many teeth does this gear have?"',
        )
        qa_gtfa = st.text_input(
            "GTFA (Ground Truth Final Answer)",
            key="qa_gtfa",
            placeholder='Example: "24" or "blue"',
            help="Optional. If provided, the platform compares the model answer against this expected answer.",
        )
        qa_gtfa_tol = st.number_input(
            "GTFA numeric tolerance (%)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
            step=0.1,
            key="qa_gtfa_tol",
            help="Used only when GTFA and predicted answer are interpreted as numeric.",
        )
        qa_use_ai = st.checkbox(
            "Use AI for open-ended Q&A",
            value=False,
            key="qa_use_ai",
            help=(
                "When enabled, the platform sends extracted STEP facts to the selected model to answer broader questions."
            ),
        )
        if qa_use_ai:
            qa_provider = st.selectbox(
                "Q&A Provider",
                options=["openai", "anthropic"],
                index=0,
                key="qa_provider",
            )
            if qa_provider == "anthropic":
                qa_model = _anthropic_model_widget(
                    choice_key="qa_model_anthropic_choice",
                    custom_key="qa_model_anthropic_custom",
                )
            else:
                qa_model = st.text_input(
                    "Q&A Model",
                    key="qa_model",
                )
            qa_api_key = st.text_input(
                "Q&A API key (optional, else env var)",
                type="password",
                key="qa_api_key",
            )
        run_btn = st.button("Run STEP Q&A", type="primary", key="run_qa")


if run_btn:
    st.session_state.last_error = None
    st.session_state.last_result = None
    st.session_state.last_run_dir = None
    st.session_state.last_generation = None
    st.session_state.last_qa = None
    st.session_state.last_task_mode = task_mode
    st.session_state.last_compare_mesh_paths = None

    run_dir = _new_run_dir()
    st.session_state.last_run_dir = str(run_dir.resolve())

    try:
        if task_mode == TASK_QA:
            if qa_mode == "Upload":
                if qa_file is None:
                    raise ValueError("Please upload a STEP file.")
                qa_input_path = _save_uploaded(qa_file, f"qa_{qa_file.name}")
            else:
                qa_input_path = qa_path_text.strip()

            if not qa_input_path:
                raise ValueError("STEP input path is required.")
            _check_step_only(qa_input_path)

            with st.spinner("Analyzing STEP and answering question..."):
                qa_result = answer_step_question(
                    input_path=qa_input_path,
                    question=qa_question.strip(),
                    run_dir=str(run_dir),
                    use_ai=qa_use_ai,
                    provider=qa_provider,
                    model=qa_model.strip(),
                    api_key=qa_api_key.strip() or None,
                    gtfa=qa_gtfa,
                    numeric_tolerance_percent=float(qa_gtfa_tol),
                )

            (run_dir / "inputs.json").write_text(
                json.dumps(
                    {
                        "task_mode": task_mode,
                        "step_input_path": qa_input_path,
                        "question": qa_question,
                        "gtfa": qa_gtfa,
                        "gtfa_numeric_tolerance_percent": float(qa_gtfa_tol),
                        "qa_use_ai": qa_use_ai,
                        "qa_provider": qa_provider if qa_use_ai else None,
                        "qa_model": qa_model if qa_use_ai else None,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "qa_result.json").write_text(json.dumps(qa_result, indent=2), encoding="utf-8")
            st.session_state.last_qa = qa_result

        else:
            generation_info = None
            if ref_mode == "Upload":
                if ref_file is None:
                    raise ValueError("Please upload a reference model.")
                ref_path = _save_uploaded(ref_file, f"reference_{ref_file.name}")
            else:
                ref_path = ref_path_text.strip()
            if not ref_path:
                raise ValueError("Reference path is required.")
            _check_supported(ref_path)

            if task_mode == TASK_GENERATE and prompt_input_mode == "Step Builder":
                required_builder_fields = {
                    "part goal": (prompt_builder_fields.get("part_goal") or "").strip(),
                    "units and frame": (prompt_builder_fields.get("units_and_frame") or "").strip(),
                    "geometry": (prompt_builder_fields.get("geometry") or "").strip(),
                }
                missing = [k for k, v in required_builder_fields.items() if not v]
                if missing:
                    raise ValueError(
                        "Step Builder is missing required rows: " + ", ".join(missing)
                    )

            if not ai_prompt.strip():
                raise ValueError("Please provide a prompt.")
            if task_mode == TASK_GENERATE and st.session_state.get("enforce_prompt_lint", True):
                lint = lint_prompt(ai_prompt.strip())
                if lint.errors:
                    raise ValueError("Prompt lint failed: " + "; ".join(lint.errors))

            effective_provider = ai_provider.strip().lower()
            effective_model = ai_model.strip()
            if task_mode == TASK_GENERATE:
                if generate_stack == GENERATE_STACK_GPT:
                    effective_provider = "openai"
                    effective_model = OPENAI_DEFAULT_MODEL
                else:
                    effective_provider = "anthropic"
                    effective_model = ai_model.strip()
                    if not effective_model:
                        raise ValueError("Please provide a valid Anthropic model ID for generation.")

            base_step_path = None
            if task_mode == TASK_MODIFY:
                if base_mode == "Upload":
                    if base_file is None:
                        raise ValueError("Please upload the base STEP to modify.")
                    base_step_path = _save_uploaded(base_file, f"base_{base_file.name}")
                else:
                    base_step_path = base_path_text.strip()
                if not base_step_path:
                    raise ValueError("Base STEP path is required for modify mode.")
                _check_step_only(base_step_path)

            with st.spinner("Generating CAD and exporting STEP/STL..."):
                generation_info = generate_and_export(
                    prompt=ai_prompt.strip(),
                    provider=effective_provider,
                    model=effective_model,
                    run_dir=str(run_dir),
                    api_key=ai_api_key.strip() or None,
                    generation_mode=task_mode,
                    base_step_path=base_step_path,
                )

            gen_path = generation_info["generated_step_path"]
            _check_supported(gen_path)

            overrides = json.loads(threshold_blob) if threshold_blob.strip() else {}
            ref_mesh_path = ensure_mesh_path(
                input_path=ref_path,
                run_dir=str(run_dir),
                prefix="reference",
            )
            gen_mesh_path = ensure_mesh_path(
                input_path=gen_path,
                run_dir=str(run_dir),
                prefix="generated",
            )

            (run_dir / "inputs.json").write_text(
                json.dumps(
                    {
                        "task_mode": task_mode,
                        "reference_path": ref_path,
                        "generated_path": gen_path,
                        "reference_mesh_path": ref_mesh_path,
                        "generated_mesh_path": gen_mesh_path,
                        "sample_points": sample_points,
                        "voxel_pitch_mm": float(voxel_pitch),
                        "fast_mode": fast_mode,
                        "grading_profile": grading_profile,
                        "threshold_overrides": overrides,
                        "generation": {
                            "enabled": True,
                            "provider": effective_provider,
                            "model": effective_model,
                            "generate_stack": generate_stack if task_mode == TASK_GENERATE else None,
                            "prompt": ai_prompt,
                            "prompt_input": {
                                "mode": prompt_input_mode if task_mode == TASK_GENERATE else "Raw Prompt",
                                "builder_fields": prompt_builder_fields
                                if task_mode == TASK_GENERATE and prompt_input_mode == "Step Builder"
                                else None,
                                "collated_prompt": prompt_builder_collated
                                if task_mode == TASK_GENERATE and prompt_input_mode == "Step Builder"
                                else None,
                            },
                            "artifacts": generation_info,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with st.spinner("Computing 44 metrics..."):
                result = compare_models(
                    reference_path=ref_mesh_path,
                    generated_path=gen_mesh_path,
                    sample_points=sample_points,
                    voxel_pitch_mm=float(voxel_pitch),
                    thresholds=overrides,
                    fast_mode=fast_mode,
                    grading_profile=grading_profile,
                    alignment_method=alignment_method,
                )

            if not result.get("ok"):
                raise RuntimeError(result.get("error") or "Comparison failed for unknown reason.")

            st.session_state.last_result = result
            st.session_state.last_generation = generation_info
            st.session_state.last_compare_mesh_paths = {
                "reference_mesh_path": ref_mesh_path,
                "generated_mesh_path": gen_mesh_path,
            }
            (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    except json.JSONDecodeError as exc:
        st.session_state.last_error = f"Invalid threshold override JSON: {exc}"
        (run_dir / "error.txt").write_text(st.session_state.last_error, encoding="utf-8")
    except (GenerationError, StepConversionError, StepQAError) as exc:
        st.session_state.last_error = str(exc)
        (run_dir / "error.txt").write_text(st.session_state.last_error, encoding="utf-8")
    except Exception as exc:
        st.session_state.last_error = str(exc)
        (run_dir / "error.txt").write_text(st.session_state.last_error, encoding="utf-8")


if st.session_state.last_error:
    st.error(st.session_state.last_error)
    if st.session_state.last_run_dir:
        st.code(f"Run folder: {st.session_state.last_run_dir}")

if st.session_state.last_task_mode == TASK_QA and st.session_state.last_qa:
    if st.session_state.last_run_dir:
        st.success(f"Saved Q&A result to: {st.session_state.last_run_dir}/qa_result.json")
    _render_qa_result(st.session_state.last_qa)
elif st.session_state.last_result:
    if st.session_state.last_run_dir:
        st.success(f"Saved run result to: {st.session_state.last_run_dir}/result.json")
    if st.session_state.last_generation:
        with st.expander("AI generation artifacts", expanded=False):
            st.json(
                {
                    "generation_mode": st.session_state.last_generation.get("generation_mode"),
                    "provider": st.session_state.last_generation.get("provider"),
                    "model": st.session_state.last_generation.get("model"),
                    "base_step_path": st.session_state.last_generation.get("base_step_path"),
                    "generated_step_path": st.session_state.last_generation.get("generated_step_path"),
                    "generated_stl_path": st.session_state.last_generation.get("generated_stl_path"),
                    "generated_code_path": st.session_state.last_generation.get("generated_code_path"),
                }
            )
    mesh_paths = st.session_state.last_compare_mesh_paths or {}
    ref_view_path = mesh_paths.get("reference_mesh_path")
    gen_view_path = mesh_paths.get("generated_mesh_path")
    show_views = st.checkbox(
        "Show interactive 3D views (extra memory)",
        value=False,
        key="show_compare_views",
    )
    if show_views and ref_view_path and gen_view_path:
        _render_compare_3d_views(ref_view_path, gen_view_path)
    _render_compare_result(st.session_state.last_result)
else:
    st.info("Choose a task type in the sidebar, provide inputs, and run.")


with st.expander("Threshold sets and grading profiles"):
    for set_name, set_data in THRESHOLD_SETS.items():
        st.subheader(set_name)
        st.caption(set_data["description"])
        st.json(set_data["thresholds"])
    st.subheader("Grading profiles")
    st.json(GRADING_PROFILES)
