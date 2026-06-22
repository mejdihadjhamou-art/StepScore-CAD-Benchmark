from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Dict, Optional


class GenerationError(RuntimeError):
    pass


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except Exception:
        return f"SIG{signum}"


def _resource_limited_preexec(memory_mb: int):
    """
    Build a preexec function that applies soft resource limits for child CAD export.
    Unix-only best effort.
    """

    def _apply_limits():
        try:
            import resource

            mem_bytes = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            # Best-effort only; continue without hard limits if unsupported.
            pass

    return _apply_limits


def _extract_python_code(text: str) -> str:
    if not text or not text.strip():
        raise GenerationError("Model returned empty output.")

    fenced = re.findall(r"```(?:python)?\n([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        return fenced[0].strip()

    # Fallback: raw text as code.
    return text.strip()


def _build_system_prompt(generation_mode: str = "generate") -> str:
    if generation_mode == "modify":
        return (
            "You are an expert CadQuery engineer. "
            "Return only valid Python CadQuery code. No markdown, no explanations. "
            "You are modifying an existing CAD model loaded from STEP. "
            "A runtime variable named base_step_path is provided and points to the input STEP/STP. "
            "Import the base with cq.importers.importStep(base_step_path), apply requested edits, "
            "and define final variable 'part'. "
            "By default produce one connected solid, unless the user specification explicitly requires "
            "multiple disconnected solids/components. "
            "CadQuery robustness rules: never call union/cut/intersect on an empty Workplane stack. "
            "If patterning repeated solids, initialize from a real solid first, then union additional solids. "
            "If multiple disconnected solids are explicitly required by the spec, return "
            "part = cq.Compound.makeCompound([solid1.val(), solid2.val(), ...]) "
            "instead of boolean-unioning them together."
        )
    return (
        "You are an expert CadQuery engineer. "
        "Return only valid Python CadQuery code. No markdown, no explanations. "
        "Define a final variable named 'part' (CadQuery Workplane/Solid). "
        "By default produce one connected solid, unless the user specification explicitly requires "
        "multiple disconnected solids/components. "
        "Do not write files directly; only build the geometry in code. "
        "CadQuery robustness rules: never call union/cut/intersect on an empty Workplane stack. "
        "If patterning repeated solids, initialize from a real solid first, then union additional solids. "
        "If multiple disconnected solids are explicitly required by the spec, return "
        "part = cq.Compound.makeCompound([solid1.val(), solid2.val(), ...]) "
        "instead of boolean-unioning them together."
    )


def _build_user_prompt(user_prompt: str, generation_mode: str = "generate") -> str:
    if generation_mode == "modify":
        return textwrap.dedent(
            f"""
            Modify the existing CAD model based on this specification:

            {user_prompt}

            Requirements:
            - Units: mm
            - Use CadQuery only
            - Import source model from variable `base_step_path` using cq.importers.importStep(base_step_path)
            - Apply modifications to the imported geometry
            - Define final variable: part
            - Produce one connected solid by default unless prompt explicitly asks for multiple components
            - Never perform union/cut/intersect on an empty Workplane stack
            - For repeated solids, initialize from first solid then union others
            """
        ).strip()
    return textwrap.dedent(
        f"""
        Create CadQuery Python code for this part specification:

        {user_prompt}

        Requirements:
        - Units: mm
        - Use CadQuery only
        - Define final variable: part
        - Produce one connected solid by default unless prompt explicitly asks for multiple components
        - Never perform union/cut/intersect on an empty Workplane stack
        - For repeated solids, initialize from first solid then union others
        """
    ).strip()


def _call_openai(model: str, prompt: str, api_key: Optional[str], generation_mode: str = "generate") -> str:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise GenerationError("openai package not installed. Install with: pip install openai") from exc

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise GenerationError("Missing OPENAI_API_KEY (or provide API key in dashboard).")

    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": _build_system_prompt(generation_mode=generation_mode)},
            {"role": "user", "content": _build_user_prompt(prompt, generation_mode=generation_mode)},
        ],
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(model: str, prompt: str, api_key: Optional[str], generation_mode: str = "generate") -> str:
    try:
        import anthropic
    except Exception as exc:
        raise GenerationError("anthropic package not installed. Install with: pip install anthropic") from exc

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise GenerationError("Missing ANTHROPIC_API_KEY (or provide API key in dashboard).")

    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0.1,
        system=_build_system_prompt(generation_mode=generation_mode),
        messages=[{"role": "user", "content": _build_user_prompt(prompt, generation_mode=generation_mode)}],
    )

    parts = []
    for block in resp.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def generate_code_from_prompt(
    prompt: str,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    generation_mode: str = "generate",
) -> Dict[str, str]:
    if generation_mode not in {"generate", "modify"}:
        raise GenerationError(f"Unsupported generation mode '{generation_mode}'.")

    provider_norm = provider.strip().lower()
    if provider_norm == "openai":
        raw = _call_openai(model=model, prompt=prompt, api_key=api_key, generation_mode=generation_mode)
    elif provider_norm == "anthropic":
        raw = _call_anthropic(model=model, prompt=prompt, api_key=api_key, generation_mode=generation_mode)
    else:
        raise GenerationError(f"Unsupported provider '{provider}'. Use 'openai' or 'anthropic'.")

    code = _extract_python_code(raw)
    if "part" not in code:
        # Not a hard parser, just guardrail.
        raise GenerationError("Generated code does not define required variable 'part'.")

    return {"raw_response": raw, "code": code}


def _build_runner_script(
    code_path: Path,
    out_step: Path,
    out_stl: Path,
    base_step_path: Optional[str] = None,
) -> str:
    ns_init = '{"cq": cq}'
    if base_step_path:
        ns_init = '{"cq": cq, "base_step_path": ' + repr(str(base_step_path)) + "}"

    return textwrap.dedent(
        f"""
        import sys
        import traceback
        import cadquery as cq

        ns = {ns_init}
        code = open(r"{code_path}", "r", encoding="utf-8").read()

        try:
            exec(code, ns)
        except Exception:
            traceback.print_exc()
            sys.exit(2)

        part = ns.get("part")
        if part is None:
            print("ERROR: generated code did not define 'part'")
            sys.exit(3)

        # Normalize to CadQuery Workplane for exporters.
        if hasattr(part, "val"):
            shape = part
        else:
            try:
                shape = cq.Workplane(obj=part)
            except Exception:
                print("ERROR: could not normalize 'part' for export")
                sys.exit(4)

        try:
            cq.exporters.export(shape, r"{out_step}")
            cq.exporters.export(shape, r"{out_stl}")
        except Exception:
            traceback.print_exc()
            sys.exit(5)

        print("OK")
        """
    ).strip()


def execute_cadquery_and_export(
    code: str,
    run_dir: str,
    timeout_seconds: int = 180,
    base_step_path: Optional[str] = None,
    out_prefix: str = "generated",
) -> Dict[str, str]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    code_path = run_path / "generated_code.py"
    raw_path = run_path / "generated_raw.txt"
    runner_path = run_path / "_runner_export.py"
    runner_stdout_path = run_path / "runner_stdout.txt"
    runner_stderr_path = run_path / "runner_stderr.txt"
    out_step = run_path / f"{out_prefix}.step"
    out_stl = run_path / f"{out_prefix}.stl"

    code_path.write_text(code, encoding="utf-8")
    runner_path.write_text(
        _build_runner_script(code_path, out_step, out_stl, base_step_path=base_step_path),
        encoding="utf-8",
    )

    max_mem_env = os.environ.get("CADQUERY_RUN_MAX_MEM_MB")
    max_mem_mb = None
    if max_mem_env is not None and str(max_mem_env).strip():
        try:
            parsed = int(max_mem_env)
            if parsed > 0:
                max_mem_mb = parsed
        except Exception:
            max_mem_mb = None
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    python_exec = sys.executable or "python3"

    run_kwargs = dict(
        args=[python_exec, str(runner_path)],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )
    if os.name == "posix" and max_mem_mb is not None:
        run_kwargs["preexec_fn"] = _resource_limited_preexec(memory_mb=max_mem_mb)

    proc = subprocess.run(**run_kwargs)
    runner_stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    runner_stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        signal_hint = ""
        if proc.returncode < 0:
            sig_num = -proc.returncode
            sig_label = _signal_name(sig_num)
            signal_hint = f"\nterminated_by_signal={sig_label} ({sig_num})"
            if sig_num == 9:
                signal_hint += (
                    "\nLikely cause: process killed due to memory pressure (OOM) "
                    "or external kill. Try a simpler prompt/geometry, lower model complexity, "
                    "or increase memory."
                )
        raise GenerationError(
            "CadQuery execution/export failed. "
            f"exit={proc.returncode}{signal_hint}\n"
            f"python_exec={python_exec}\n"
            f"runner_stdout_file={runner_stdout_path}\n"
            f"runner_stderr_file={runner_stderr_path}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    if not out_step.exists() or not out_stl.exists():
        raise GenerationError("Expected generated.step/generated.stl not found after export.")

    return {
        "generated_code_path": str(code_path),
        "generated_step_path": str(out_step),
        "generated_stl_path": str(out_stl),
        "runner_stdout": proc.stdout,
        "runner_stderr": proc.stderr,
    }


def generate_and_export(
    prompt: str,
    provider: str,
    model: str,
    run_dir: str,
    api_key: Optional[str] = None,
    generation_mode: str = "generate",
    base_step_path: Optional[str] = None,
) -> Dict[str, str]:
    if generation_mode not in {"generate", "modify"}:
        raise GenerationError(f"Unsupported generation mode '{generation_mode}'.")
    if generation_mode == "modify":
        if not base_step_path:
            raise GenerationError("Modify mode requires base_step_path.")
        if not Path(base_step_path).exists():
            raise GenerationError(f"Base STEP file not found: {base_step_path}")

    gen = generate_code_from_prompt(
        prompt=prompt,
        provider=provider,
        model=model,
        api_key=api_key,
        generation_mode=generation_mode,
    )

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "prompt.txt").write_text(prompt, encoding="utf-8")
    (run_path / "generated_raw.txt").write_text(gen["raw_response"], encoding="utf-8")

    exports = execute_cadquery_and_export(
        code=gen["code"],
        run_dir=run_dir,
        base_step_path=base_step_path,
        out_prefix="generated",
    )
    return {
        "provider": provider,
        "model": model,
        "generation_mode": generation_mode,
        "base_step_path": base_step_path,
        "generated_code": gen["code"],
        "generated_raw_response": gen["raw_response"],
        **exports,
    }
