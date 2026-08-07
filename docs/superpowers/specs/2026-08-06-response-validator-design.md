# Response Validation Layer — Design

*2026-08-06 — implements item 2 of "The Architecture Pass" (09 series), §1*

## Problem

`brain.py` has no layer between a model's raw text output and the pipeline stages
that construct a player-facing `{"reply": ..., "commands": [...]}` dict. When a
model call succeeds at the HTTP level but returns garbage — an empty string, a
JSON payload cut off mid-key by a network hiccup, or a Python-dict-repr string
with single-quoted keys instead of real JSON — the only handling today is a
generic `if not parsed: parsed = {...}` fallback in `stage_commands` and
`run_fast`. One of those fallbacks (`run_fast`, "I heard you want...") is itself
assistant-toned language that breaks Lucineer's character — the exact bug the
architecture pass traced into production via the playtest transcripts (§4).

## Design

**`response_validator.py`** — new, standalone module, no dependency on `brain.py`.

- `validate_response(raw: str) -> ValidationResult` classifies raw model output
  into one of three failure types, in this order:
  1. **`empty_response`** — `raw` is `None` or whitespace-only.
  2. **`truncated_payload`** — a bracket/quote-depth scan shows the payload ends
     inside an open string or with unclosed `{`/`[` — the shape a real network
     cutoff produces (`{"reply": "Dock's in", "comm`), not generic invalid JSON.
  3. **`malformed_json`** — structurally balanced but `json.loads` still fails
     (single-quoted keys, unquoted `True`/`False`/`None`, plain prose), or it
     parses but isn't a JSON object. A dedicated regex flags the specific
     Python-dict-repr shape seen leaking into live replies per §4 of the pass.
- `fallback_response(failure) -> dict` returns a safe `{"reply": ..., "commands":
  [], "error": <failure>}` dict, with the reply drawn from a small in-voice line
  pool per failure type — short, foreman-register, no assistant phrasing.
- `is_voice_safe(text) -> bool` — a minimal voice-integrity check (rejects text
  starting with `{`, or matching a banned assistant-phrase list) used to keep
  the fallback line pool honest.

**Integration into `brain.py`** — two call sites only:
- `stage_commands`: replace the inline `{"reply": f"I tried to build...", ...}`
  fallback with `response_validator.fallback_response(...)`.
- `run_fast`: replace the `"I heard you want..."` fallback the same way.

`stage_intent` and `stage_hermes` are unchanged — `stage_intent`'s fallback
never reaches the player as a `reply`, and `stage_hermes` already discards
unparseable output and preserves the prior safe reply.

## Out of scope

- The typed-exception refactor of `call_model` (item 1 of the same doc section)
  — a separate, larger change from what was asked for here.
- Schema validation of `commands[].params` (item 2's second half — numeric
  position/size, known material strings, etc.) — a distinct validator with its
  own design, not built in this pass.

## Testing

`tests/test_response_validator.py`, standalone (no `brain` import needed):
classification tests for all three failure types plus valid-JSON passthrough,
fallback line voice-safety assertions, and edge cases (nested braces, escaped
quotes inside strings, markdown-fenced JSON, non-dict JSON like a bare array).
