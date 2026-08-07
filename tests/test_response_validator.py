#!/usr/bin/env python3
"""
Tests for response_validator.py — the response validation layer that sits
between a model's raw text output and the pipeline stages that build the
player-facing reply. Covers the three failure classes The Architecture Pass
(09 series, §1) called out: empty 200 OK responses, malformed JSON
(including the Python-dict-repr shape seen leaking into live replies), and
payloads truncated mid-stream.

Standalone — does not import brain.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import response_validator as rv


# ═══════════════════════════════════════════════════════════════════════════════
# EMPTY RESPONSES
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmptyResponse:
    def test_none(self):
        result = rv.validate_response(None)
        assert result.ok is False
        assert result.failure == rv.EMPTY_RESPONSE

    def test_empty_string(self):
        result = rv.validate_response("")
        assert result.ok is False
        assert result.failure == rv.EMPTY_RESPONSE

    def test_whitespace_only(self):
        result = rv.validate_response("   \n\t  ")
        assert result.ok is False
        assert result.failure == rv.EMPTY_RESPONSE


# ═══════════════════════════════════════════════════════════════════════════════
# TRUNCATED PAYLOADS — cut mid-stream, not just "invalid JSON"
# ═══════════════════════════════════════════════════════════════════════════════

class TestTruncatedPayload:
    def test_cut_mid_key(self):
        """The exact shape the architecture pass names: cut off mid-key."""
        raw = '{"reply": "Dock\'s in", "comm'
        result = rv.validate_response(raw)
        assert result.ok is False
        assert result.failure == rv.TRUNCATED_PAYLOAD

    def test_cut_mid_string_value(self):
        raw = '{"reply": "Threw up a tower but didn'
        result = rv.validate_response(raw)
        assert result.ok is False
        assert result.failure == rv.TRUNCATED_PAYLOAD

    def test_unclosed_array(self):
        raw = '{"reply": "ok", "commands": [{"type": "createPart", "params": {"name": "wall"'
        result = rv.validate_response(raw)
        assert result.ok is False
        assert result.failure == rv.TRUNCATED_PAYLOAD

    def test_unclosed_nested_object(self):
        raw = '{"reply": "ok", "commands": [{"type": "createPart"'
        result = rv.validate_response(raw)
        assert result.ok is False
        assert result.failure == rv.TRUNCATED_PAYLOAD

    def test_dangling_key_no_value(self):
        raw = '{"reply": "ok", "commands": [], "transparency":'
        result = rv.validate_response(raw)
        assert result.ok is False
        assert result.failure == rv.TRUNCATED_PAYLOAD

    def test_scanner_handles_escaped_quotes(self):
        """An escaped quote inside a string must not be mistaken for the
        string's closing quote (which would misclassify a complete payload
        as truncated, or vice versa)."""
        # Backslash-quote inside the string, then the string is still open.
        raw = '{"reply": "He said \\"hi\\" and then cut off'
        assert rv._is_truncated(raw) is True
        # Same escaping, but the payload is actually complete.
        complete = '{"reply": "He said \\"hi\\" to me", "commands": []}'
        assert rv._is_truncated(complete) is False


# ═══════════════════════════════════════════════════════════════════════════════
# MALFORMED JSON — complete-looking but not valid JSON, including the
# Python-dict-repr shape found leaking into live replies (§4 of the pass)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMalformedJson:
    def test_single_quoted_dict_repr(self):
        """The exact live bug: Python dict repr instead of JSON."""
        raw = "{'reply': 'ok', 'commands': []}"
        result = rv.validate_response(raw)
        assert result.ok is False
        assert result.failure == rv.MALFORMED_JSON
        assert "repr" in result.detail

    def test_python_boolean_literal(self):
        raw = "{'reply': 'ok', 'transparency': False, 'commands': []}"
        result = rv.validate_response(raw)
        assert result.ok is False
        assert result.failure == rv.MALFORMED_JSON

    def test_unquoted_keys(self):
        raw = "{reply: ok, commands: []}"
        result = rv.validate_response(raw)
        assert result.ok is False
        assert result.failure == rv.MALFORMED_JSON

    def test_plain_prose(self):
        raw = "I cannot help with that."
        result = rv.validate_response(raw)
        assert result.ok is False
        assert result.failure == rv.MALFORMED_JSON

    def test_balanced_but_broken_braces(self):
        """Balanced brackets but not valid JSON — malformed, not truncated,
        since nothing is left open."""
        raw = "{{{{broken nested braces}}}}"
        result = rv.validate_response(raw)
        assert result.ok is False
        assert result.failure == rv.MALFORMED_JSON

    def test_valid_json_but_not_an_object(self):
        """A bare JSON array parses fine but isn't the expected schema."""
        raw = "[1, 2, 3]"
        result = rv.validate_response(raw)
        assert result.ok is False
        assert result.failure == rv.MALFORMED_JSON
        assert "object" in result.detail

    def test_binary_garbage(self):
        raw = "\x00\x01binary garbage here"
        result = rv.validate_response(raw)
        assert result.ok is False
        assert result.failure == rv.MALFORMED_JSON


# ═══════════════════════════════════════════════════════════════════════════════
# VALID PASSTHROUGH — well-formed responses should validate cleanly
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidPassthrough:
    def test_clean_json(self):
        raw = '{"reply": "Built it.", "commands": []}'
        result = rv.validate_response(raw)
        assert result.ok is True
        assert result.failure is None
        assert result.data == {"reply": "Built it.", "commands": []}

    def test_markdown_fenced_json(self):
        raw = '```json\n{"reply": "Built it.", "commands": []}\n```'
        result = rv.validate_response(raw)
        assert result.ok is True
        assert result.data["reply"] == "Built it."

    def test_markdown_fenced_no_language_tag(self):
        raw = '```\n{"reply": "ok", "commands": []}\n```'
        result = rv.validate_response(raw)
        assert result.ok is True

    def test_escaped_quotes_inside_string(self):
        raw = '{"reply": "He said \\"hi\\" to me", "commands": []}'
        result = rv.validate_response(raw)
        assert result.ok is True

    def test_pretty_printed_json(self):
        raw = '{\n  "reply": "Built it.",\n  "commands": [\n    {"type": "createPart"}\n  ]\n}'
        result = rv.validate_response(raw)
        assert result.ok is True
        assert len(result.data["commands"]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK RESPONSES — must be voice-safe, never leak raw JSON or assistant tone
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallbackResponse:
    @pytest.mark.parametrize("failure", [rv.EMPTY_RESPONSE, rv.TRUNCATED_PAYLOAD, rv.MALFORMED_JSON])
    def test_fallback_has_required_shape(self, failure):
        result = rv.fallback_response(failure)
        assert "reply" in result
        assert result["commands"] == []
        assert result["error"] == failure

    @pytest.mark.parametrize("failure", [rv.EMPTY_RESPONSE, rv.TRUNCATED_PAYLOAD, rv.MALFORMED_JSON])
    def test_fallback_reply_is_voice_safe(self, failure):
        result = rv.fallback_response(failure)
        assert rv.is_voice_safe(result["reply"])

    def test_unknown_failure_falls_back_to_malformed_pool(self):
        result = rv.fallback_response("some_unrecognized_failure")
        assert result["error"] == "some_unrecognized_failure"
        assert rv.is_voice_safe(result["reply"])

    def test_index_rotates_through_pool(self):
        first = rv.fallback_response(rv.MALFORMED_JSON, index=0)
        second = rv.fallback_response(rv.MALFORMED_JSON, index=1)
        assert first["reply"] != second["reply"]
        # Wraps around
        wrapped = rv.fallback_response(rv.MALFORMED_JSON, index=2)
        assert wrapped["reply"] == first["reply"]


class TestVoiceSafety:
    def test_rejects_raw_json_leak(self):
        assert rv.is_voice_safe('{"reply": "ok", "commands": []}') is False

    def test_rejects_raw_array_leak(self):
        assert rv.is_voice_safe('["reply", "commands"]') is False

    def test_rejects_assistant_toned_fallback(self):
        """This is the exact bug the architecture pass flagged in run_fast."""
        assert rv.is_voice_safe("I heard you want: a castle, but I had trouble.") is False

    def test_rejects_other_assistant_phrases(self):
        assert rv.is_voice_safe("I'd be happy to help with that!") is False
        assert rv.is_voice_safe("Let me build that for you.") is False
        assert rv.is_voice_safe("Certainly, right away.") is False

    def test_accepts_foreman_voice(self):
        assert rv.is_voice_safe("Cut off partway through that build. Try it again.") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
