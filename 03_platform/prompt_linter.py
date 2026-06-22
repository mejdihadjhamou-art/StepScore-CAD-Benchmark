from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class LintResult:
    errors: List[str]
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return not self.errors


REQUIRED_SECTIONS = {
    "part identity": ["part identity"],
    "units and frame": ["units and coordinate frame", "units and frame"],
    "geometry": ["base geometry", "geometry"],
    "topology": ["topology/output constraints", "topology output constraints"],
}

PLACEHOLDER_RE = re.compile(r"\b(TBD|TBA|N/?A|TODO)\b", re.IGNORECASE)
DIM_RE = re.compile(r"(\b\d+(\.\d+)?\b)|([Ø⌀]\s*\d+(\.\d+)?)")
AXIS_RE = re.compile(r"(\+?x|\+?y|\+?z)\s*(axis|direction|is|=)", re.IGNORECASE)
ORIGIN_RE = re.compile(r"\borigin\b", re.IGNORECASE)
UNITS_MM_RE = re.compile(r"\b(mm|millimeter|millimetre)s?\b", re.IGNORECASE)
UNITS_IN_RE = re.compile(r"\b(inch|inches|in)\b", re.IGNORECASE)
PLACEHOLDER_REF_RE = re.compile(
    r"\b(same as above|as above|see image|see picture|similar to previous|as before)\b",
    re.IGNORECASE,
)
CONTRADICT_RE = re.compile(
    r"\b(height|width|length|diameter|radius|thickness|depth)\b\s*=?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
FORBIDDEN_FEATURES = [
    "color",
    "colour",
    "texture",
    "render",
    "appearance",
    "logo",
    "engrave",
    "engraving",
    "text",
]
VAGUE_PHRASES = [
    "approximately",
    "approx",
    "around",
    "roughly",
    "somewhat",
    "about",
    "etc",
    "and so on",
    "as needed",
    "as required",
    "standard",
    "typical",
    "normal",
    "reasonable",
    "medium",
    "large",
    "small",
    "similar to",
    "like",
    "should be",
    "make it look",
    "make it nice",
    "to taste",
    "generic",
    "any shape",
    "whatever",
    "anything you want",
]

def lint_prompt(prompt: str, min_numeric_warn: int = 5, min_numeric_error: int = 3) -> LintResult:
    p = (prompt or "").strip()
    errors: List[str] = []
    warnings: List[str] = []

    if not p:
        return LintResult(errors=["Prompt is empty."], warnings=[])

    lower = p.lower()

    # Required sections
    for label, variants in REQUIRED_SECTIONS.items():
        if not any(v in lower for v in variants):
            errors.append(f"Missing required section: {label}.")

    # Units check
    if not UNITS_MM_RE.search(p):
        errors.append("Units not specified (mm).")
    if UNITS_MM_RE.search(p) and UNITS_IN_RE.search(p):
        errors.append("Mixed units detected (mm and inches). Use one unit system.")

    # Coordinate frame hints
    if not ORIGIN_RE.search(p):
        errors.append("Origin not specified (e.g., origin at center of bottom face).")
    if not AXIS_RE.search(p):
        errors.append("Axis orientation not specified (e.g., +Z up).")

    # Dimensions
    dims = DIM_RE.findall(p)
    dim_count = len(dims)
    if dim_count == 0:
        errors.append("No numeric dimensions found.")
    if dim_count < min_numeric_error:
        errors.append(f"Too few numeric dimensions (found {dim_count}).")
    elif dim_count < min_numeric_warn:
        warnings.append(f"Low numeric detail (found {dim_count}; recommended >= {min_numeric_warn}).")

    # Placeholders
    if PLACEHOLDER_RE.search(p):
        errors.append("Prompt contains placeholders (TBD/TBA/N/A/TODO).")

    # Ambiguous references
    if PLACEHOLDER_REF_RE.search(p):
        errors.append("Prompt contains ambiguous references (e.g., 'as above', 'see image').")

    # Vague phrases
    vague_hits = [phrase for phrase in VAGUE_PHRASES if phrase in lower]
    if vague_hits:
        errors.append(
            "Prompt contains vague terms: " + ", ".join(sorted(set(vague_hits)))
        )

    # Forbidden non-geometric features
    forbidden_hits = [phrase for phrase in FORBIDDEN_FEATURES if phrase in lower]
    if forbidden_hits:
        warnings.append(
            "Prompt contains non-geometric terms: " + ", ".join(sorted(set(forbidden_hits)))
        )

    # Feature completeness checks
    if "hole" in lower or "bore" in lower:
        if "diameter" not in lower and "dia" not in lower and "ø" not in lower and "⌀" not in lower:
            errors.append("Holes mentioned but no diameter specified.")
        if "through" not in lower and "depth" not in lower:
            errors.append("Holes mentioned but no depth/through specification.")
    if "slot" in lower:
        if "width" not in lower or "length" not in lower:
            errors.append("Slots mentioned but width/length not specified.")
    if "pocket" in lower:
        if not ("depth" in lower and ("width" in lower or "length" in lower)):
            errors.append("Pocket mentioned but missing depth and size.")
    if "fillet" in lower or "chamfer" in lower:
        if "radius" not in lower and "r=" not in lower and "size" not in lower and "mm" not in lower:
            warnings.append("Fillet/chamfer mentioned without explicit size.")

    # Tolerance/fit requirements
    if any(k in lower for k in ["tolerance", "clearance", "press fit", "slip fit", "interference fit", "fit"]):
        tol_match = re.search(r"(tolerance|clearance|fit)[^\n\r]{0,60}(\d+(?:\.\d+)?)", lower)
        if not tol_match:
            errors.append("Fit/tolerance mentioned without numeric value.")

    # Contradiction check (same key with multiple values)
    key_values = {}
    for key, val in CONTRADICT_RE.findall(p):
        k = key.lower()
        key_values.setdefault(k, set()).add(val)
    contradictions = [k for k, vals in key_values.items() if len(vals) > 1]
    if contradictions:
        errors.append("Conflicting dimensions detected for: " + ", ".join(contradictions))

    # Topology constraint explicitness
    if "single connected" not in lower and "one connected" not in lower and "single solid" not in lower:
        warnings.append("No explicit single-connected-solid requirement found.")

    return LintResult(errors=errors, warnings=warnings)
