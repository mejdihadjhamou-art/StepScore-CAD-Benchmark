from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

REVIEWER_VERSION = "1.0.0"
RUBRIC_VERSION = "cad_prompt_rubric_v1"

REQUIRED_SECTIONS: Tuple[str, ...] = (
    "Part identity",
    "Units and frame",
    "Base geometry",
    "Additive features",
    "Subtractive features",
    "Constraints",
    "Topology/output constraints",
)

AMBIGUOUS_TERMS: Tuple[str, ...] = (
    "nice",
    "smooth",
    "ergonomic",
    "roughly",
    "approximately",
    "about",
    "around",
    "some",
    "etc",
    "and so on",
)

RUBRIC_WEIGHTS: Dict[str, float] = {
    "structure_compliance": 0.20,
    "dimensional_completeness": 0.20,
    "positional_clarity": 0.20,
    "constraint_clarity": 0.15,
    "measurability": 0.15,
    "ambiguity_risk": 0.10,
}

EXPECTED_SUBSCORES: Tuple[str, ...] = tuple(RUBRIC_WEIGHTS.keys())

# Each subscore is 0..5. Weighted aggregate converts to 0..100.
SUBSCORE_MAX = 5.0


class LLMReviewerClient(Protocol):
    """Interface for pluggable AI reviewer clients."""

    def review_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Return a parsed JSON object matching the expected AI review schema."""


class OpenAIReviewerClient:
    """
    OpenAI adapter for AI prompt review.

    Notes:
    - Uses chat.completions JSON mode.
    - Keep temperature low for consistency.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        client: Any = None,
        temperature: float = 0.0,
        timeout_seconds: int = 60,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                "openai package not installed. Install with `pip install openai`."
            ) from exc

        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._client = client or OpenAI(api_key=api_key)

    def review_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        completion = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=self.timeout_seconds,
        )

        content = completion.choices[0].message.content or "{}"
        return json.loads(content)


@dataclass
class LintIssue:
    code: str
    message: str
    severity: str = "error"
    suggestion: Optional[str] = None


@dataclass
class LintResult:
    passed: bool
    score: float
    issues: List[LintIssue] = field(default_factory=list)


@dataclass
class AIReviewResult:
    status: str  # passed | failed | skipped | error
    score: Optional[float]
    subscores: Dict[str, int] = field(default_factory=dict)
    blocking_issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    suggested_rewrite: Optional[str] = None
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class PromptReviewResult:
    prompt_hash: str
    linter: LintResult
    ai: AIReviewResult
    final_pass: bool
    final_score: float
    decision_reason: str
    reviewer_version: str = REVIEWER_VERSION
    rubric_version: str = RUBRIC_VERSION


class PromptReviewer:
    """
    Hybrid reviewer:
    1) deterministic linter (hard quality checks)
    2) AI semantic reviewer (rubric scoring)
    """

    def __init__(
        self,
        llm_client: Optional[LLMReviewerClient] = None,
        ai_threshold: float = 80.0,
        hard_fail_on_lint: bool = True,
        run_ai_when_lint_fails: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.ai_threshold = ai_threshold
        self.hard_fail_on_lint = hard_fail_on_lint
        self.run_ai_when_lint_fails = run_ai_when_lint_fails

    def review_prompt(self, prompt_text: str) -> PromptReviewResult:
        text = (prompt_text or "").strip()
        prompt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        lint = lint_prompt(text)

        if self.hard_fail_on_lint and not lint.passed and not self.run_ai_when_lint_fails:
            ai = AIReviewResult(status="skipped", score=None)
            return PromptReviewResult(
                prompt_hash=prompt_hash,
                linter=lint,
                ai=ai,
                final_pass=False,
                final_score=lint.score,
                decision_reason="Blocked by linter failures.",
            )

        ai = self._run_ai_review(text)
        final_pass, final_score, reason = self._decide(lint, ai)

        return PromptReviewResult(
            prompt_hash=prompt_hash,
            linter=lint,
            ai=ai,
            final_pass=final_pass,
            final_score=final_score,
            decision_reason=reason,
        )

    def _run_ai_review(self, prompt_text: str) -> AIReviewResult:
        if self.llm_client is None:
            return AIReviewResult(
                status="skipped",
                score=None,
                error="No AI reviewer client configured.",
            )

        system_prompt, user_prompt = build_ai_reviewer_messages(prompt_text)

        try:
            raw = self.llm_client.review_json(system_prompt=system_prompt, user_prompt=user_prompt)
            normalized = validate_and_normalize_ai_response(raw)
            return AIReviewResult(
                status="passed" if normalized["pass"] else "failed",
                score=normalized["score"],
                subscores=normalized["subscores"],
                blocking_issues=normalized["blocking_issues"],
                improvement_suggestions=normalized["improvement_suggestions"],
                suggested_rewrite=normalized.get("suggested_rewrite"),
                raw_response=raw,
            )
        except Exception as exc:
            return AIReviewResult(
                status="error",
                score=None,
                error=f"AI reviewer error: {exc}",
            )

    def _decide(self, lint: LintResult, ai: AIReviewResult) -> Tuple[bool, float, str]:
        if self.hard_fail_on_lint and not lint.passed:
            return False, lint.score, "Linter failed."

        if ai.status == "skipped":
            return lint.passed, lint.score, "AI review skipped; decision based on linter."

        if ai.status == "error":
            return False, lint.score, "AI reviewer errored."

        ai_score = ai.score if ai.score is not None else 0.0
        composite = round((0.35 * lint.score) + (0.65 * ai_score), 2)

        if ai.blocking_issues:
            return False, composite, "AI review found blocking issues."

        if ai_score < self.ai_threshold:
            return False, composite, f"AI score below threshold ({ai_score:.2f} < {self.ai_threshold:.2f})."

        return True, composite, "Passed linter and AI review."


def lint_prompt(prompt_text: str) -> LintResult:
    text = (prompt_text or "").strip()
    lower = text.lower()
    issues: List[LintIssue] = []

    if not text:
        issues.append(
            LintIssue(
                code="EMPTY_PROMPT",
                message="Prompt is empty.",
                suggestion="Provide a fully specified CAD prompt.",
            )
        )
        return LintResult(passed=False, score=0.0, issues=issues)

    for section in REQUIRED_SECTIONS:
        if section.lower() not in lower:
            issues.append(
                LintIssue(
                    code="MISSING_SECTION",
                    message=f"Missing required section: '{section}'.",
                    suggestion=f"Add a '{section}' section following the guide template.",
                )
            )

    if not re.search(r"\b(mm|millimeter|millimetre|cm|m|in|inch|inches)\b", lower):
        issues.append(
            LintIssue(
                code="MISSING_UNITS",
                message="Units are not explicitly stated.",
                suggestion="Specify units explicitly (recommended: mm).",
            )
        )

    if "origin" not in lower:
        issues.append(
            LintIssue(
                code="MISSING_ORIGIN",
                message="Origin is not explicitly defined.",
                suggestion="Add exact origin definition, e.g. origin at [0,0,0] on a named feature.",
            )
        )

    if "coordinate system" not in lower and "xyz" not in lower:
        issues.append(
            LintIssue(
                code="MISSING_COORD_FRAME",
                message="Coordinate frame is not explicitly defined.",
                suggestion="Specify right-handed XYZ and axis orientation.",
            )
        )

    numeric_tokens = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(numeric_tokens) < 8:
        issues.append(
            LintIssue(
                code="LOW_NUMERIC_DETAIL",
                message="Prompt contains very few numeric values; likely underspecified.",
                suggestion="Add exact dimensions, coordinates, and depths for all critical features.",
            )
        )

    found_ambiguous = [term for term in AMBIGUOUS_TERMS if term in lower]
    if found_ambiguous:
        issues.append(
            LintIssue(
                code="AMBIGUOUS_LANGUAGE",
                message=f"Ambiguous terms detected: {', '.join(found_ambiguous)}.",
                suggestion="Replace ambiguous language with measurable constraints.",
            )
        )

    has_cut_feature = any(token in lower for token in ("hole", "slot", "pocket", "cut"))
    has_cut_depth_rule = any(token in lower for token in ("through", "blind", "depth"))
    if has_cut_feature and not has_cut_depth_rule:
        issues.append(
            LintIssue(
                code="UNSPECIFIED_CUT_DEPTH",
                message="Subtractive features exist but depth behavior (through/blind) is not clear.",
                suggestion="Specify through or blind depth for each subtractive feature.",
            )
        )

    if "single connected" not in lower and "single component" not in lower:
        issues.append(
            LintIssue(
                code="MISSING_TOPOLOGY_REQUIREMENT",
                message="Single-component topology requirement is missing.",
                suggestion="State explicitly: one connected solid component.",
            )
        )

    score = max(0.0, 100.0 - (len(issues) * 12.5))
    return LintResult(passed=len(issues) == 0, score=round(score, 2), issues=issues)


def build_ai_reviewer_messages(prompt_text: str) -> Tuple[str, str]:
    system_prompt = (
        "You are a senior CAD benchmark prompt reviewer. "
        "Evaluate prompt quality against a strict mechanical CAD rubric. "
        "Return JSON only. No markdown."
    )

    rubric_text = (
        "Score each category from 0 to 5 (integer only): "
        "structure_compliance, dimensional_completeness, positional_clarity, "
        "constraint_clarity, measurability, ambiguity_risk. "
        "Higher ambiguity_risk means less ambiguous language. "
        "A prompt should fail if any critical geometry is underspecified."
    )

    response_contract = {
        "pass": "boolean",
        "score": "number 0..100 (weighted by rubric)",
        "subscores": {
            key: "integer 0..5" for key in EXPECTED_SUBSCORES
        },
        "blocking_issues": ["string"],
        "improvement_suggestions": ["string"],
        "suggested_rewrite": "string"
    }

    user_prompt = (
        f"Rubric: {rubric_text}\n\n"
        f"Return JSON with this exact shape:\n{json.dumps(response_contract, indent=2)}\n\n"
        f"Weights: {json.dumps(RUBRIC_WEIGHTS)}\n"
        f"Rubric version: {RUBRIC_VERSION}\n\n"
        f"Prompt to review:\n{prompt_text}"
    )

    return system_prompt, user_prompt


def validate_and_normalize_ai_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("AI response must be a JSON object.")

    for key in ("subscores", "blocking_issues", "improvement_suggestions"):
        if key not in raw:
            raise ValueError(f"Missing required key in AI response: {key}")

    subscores = raw.get("subscores")
    if not isinstance(subscores, dict):
        raise ValueError("'subscores' must be an object.")

    normalized_subscores: Dict[str, int] = {}
    for metric in EXPECTED_SUBSCORES:
        if metric not in subscores:
            raise ValueError(f"Missing subscore: {metric}")
        value = subscores[metric]
        if not isinstance(value, int):
            raise ValueError(f"Subscore '{metric}' must be integer 0..5.")
        if value < 0 or value > int(SUBSCORE_MAX):
            raise ValueError(f"Subscore '{metric}' out of range: {value}")
        normalized_subscores[metric] = value

    blocking_issues = _validate_string_list(raw.get("blocking_issues"), "blocking_issues")
    improvements = _validate_string_list(raw.get("improvement_suggestions"), "improvement_suggestions")

    suggested_rewrite = raw.get("suggested_rewrite")
    if suggested_rewrite is not None and not isinstance(suggested_rewrite, str):
        raise ValueError("'suggested_rewrite' must be a string when provided.")

    weighted = 0.0
    for metric, weight in RUBRIC_WEIGHTS.items():
        weighted += (normalized_subscores[metric] / SUBSCORE_MAX) * weight
    computed_score = round(weighted * 100.0, 2)

    raw_score = raw.get("score")
    score = computed_score
    if isinstance(raw_score, (int, float)):
        # Trust computed rubric score as source of truth to keep deterministic behavior.
        score = computed_score

    declared_pass = raw.get("pass")
    if not isinstance(declared_pass, bool):
        declared_pass = (score >= 80.0 and len(blocking_issues) == 0)

    return {
        "pass": declared_pass and score >= 80.0 and len(blocking_issues) == 0,
        "score": score,
        "subscores": normalized_subscores,
        "blocking_issues": blocking_issues,
        "improvement_suggestions": improvements,
        "suggested_rewrite": suggested_rewrite,
    }


def _validate_string_list(value: Any, name: str) -> List[str]:
    if not isinstance(value, list):
        raise ValueError(f"'{name}' must be a list of strings.")
    out: List[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"'{name}[{idx}]' must be a string.")
        out.append(item.strip())
    return out


__all__ = [
    "AIReviewResult",
    "LintIssue",
    "LintResult",
    "LLMReviewerClient",
    "OpenAIReviewerClient",
    "PromptReviewResult",
    "PromptReviewer",
    "REQUIRED_SECTIONS",
    "RUBRIC_VERSION",
    "REVIEWER_VERSION",
    "build_ai_reviewer_messages",
    "lint_prompt",
    "validate_and_normalize_ai_response",
]
