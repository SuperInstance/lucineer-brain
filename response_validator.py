#!/usr/bin/env python3
"""
Response Validation Layer for Lucineer Brain

Sits between a model's raw text output and the pipeline stages that build
the player-facing {"reply": ..., "commands": [...]} dict. Classifies the
three failure shapes The Architecture Pass (09 series, §1) found leaking
into production: empty 200 OK responses, malformed JSON (including the
Python-dict-repr shape seen in live playtest transcripts — single-quoted
keys, unquoted booleans), and payloads truncated mid-stream.

Standalone — no dependency on brain.py, so it can be unit-tested and
reused independently of the pipeline that currently calls it.

Usage:
    result = validate_response(raw_text)
    if result.ok:
        use result.data
    else:
        use fallback_response(result.failure)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# ─── Failure classification ─────────────────────────────────────────────────

EMPTY_RESPONSE = "empty_response"
TRUNCATED_PAYLOAD = "truncated_payload"
MALFORMED_JSON = "malformed_json"


@dataclass
class ValidationResult:
    """Outcome of validating a raw model response."""
    ok: bool
    failure: str | None = None
    detail: str = ""
    data: dict | list | None = None


def _strip_fences(text: str) -> str:
    """Strip a single markdown code fence wrapping the payload, if present."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    lines = lines[1:]  # drop opening ``` or ```json
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


# Matches a single-quoted dict key like 'reply': — the Python-repr shape
# seen leaking into live replies (Architecture Pass §4).
_DICT_REPR_KEY = re.compile(r"'[^'\\]*'\s*:")
# Matches Python literals where JSON requires lowercase true/false/null.
_PYTHON_LITERAL = re.compile(r":\s*(True|False|None)\b")


def _looks_like_python_repr(text: str) -> bool:
    """Heuristic: does this look like a Python dict repr instead of JSON?"""
    return bool(_DICT_REPR_KEY.search(text) or _PYTHON_LITERAL.search(text))


def _is_truncated(text: str) -> bool:
    """
    Scan for unclosed braces/brackets or a string left open at end-of-text —
    the shape a real network cutoff produces (cut mid-key or mid-value),
    as opposed to a complete-but-wrong-format payload.
    """
    depth = 0
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return in_string or depth > 0


def validate_response(raw: str | None) -> ValidationResult:
    """
    Classify a raw model response. Returns ok=True with parsed `data`
    (always a dict) on success, or ok=False with a `failure` type and
    human-readable `detail` on any of the three recognized failure shapes.
    """
    if raw is None or not raw.strip():
        return ValidationResult(ok=False, failure=EMPTY_RESPONSE, detail="response was empty or whitespace-only")

    text = _strip_fences(raw)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        if _is_truncated(text):
            return ValidationResult(
                ok=False, failure=TRUNCATED_PAYLOAD,
                detail="payload ends inside an open string or unclosed object/array — looks cut off mid-stream",
            )
        detail = "not valid JSON"
        if _looks_like_python_repr(text):
            detail = "looks like a Python dict repr (single-quoted keys / True-False-None) rather than JSON"
        return ValidationResult(ok=False, failure=MALFORMED_JSON, detail=detail)

    if not isinstance(parsed, dict):
        return ValidationResult(
            ok=False, failure=MALFORMED_JSON,
            detail=f"parsed valid JSON but expected an object, got {type(parsed).__name__}",
        )

    return ValidationResult(ok=True, data=parsed)


# ─── Voice-safe fallback replies ────────────────────────────────────────────
# Short, foreman-register lines — never assistant-toned. Kept separate from
# LUCINEER_PERSONA's full voice examples in brain.py since these only ever
# fire on the failure path, not from a model call.

_FALLBACK_LINES: dict[str, list[str]] = {
    EMPTY_RESPONSE: [
        "Nothing came back on that one. Ask again.",
        "Tools went quiet mid-swing. Didn't get anything usable. Say it again.",
    ],
    TRUNCATED_PAYLOAD: [
        "Cut off partway through that build. Didn't finish the thought — try it again.",
        "Lost the signal mid-cut. Whatever came through wasn't whole. One more time.",
    ],
    MALFORMED_JSON: [
        "Whatever came back wasn't buildable — garbled instructions. Give it another shot.",
        "Got something back that didn't hold together. Not building off scrap paper. Try again.",
    ],
}

# Assistant-register phrases Lucineer is explicitly forbidden from using
# (LUCINEER_PERSONA in brain.py) — checked against every fallback line.
_BANNED_PHRASES = ("i heard", "i'd be happy", "let me", "certainly", "great question")


def is_voice_safe(text: str) -> bool:
    """Reject text that reads as raw JSON or assistant-toned filler."""
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return False
    lowered = stripped.lower()
    return not any(phrase in lowered for phrase in _BANNED_PHRASES)


def fallback_response(failure: str, *, index: int = 0) -> dict:
    """
    Build a safe {"reply", "commands", "error"} dict for a classified failure.
    `index` selects among the line pool (callers can rotate); defaults to the
    first line for deterministic behavior.
    """
    pool = _FALLBACK_LINES.get(failure, _FALLBACK_LINES[MALFORMED_JSON])
    reply = pool[index % len(pool)]
    assert is_voice_safe(reply), f"fallback line failed voice-safety check: {reply!r}"
    return {"reply": reply, "commands": [], "error": failure}
