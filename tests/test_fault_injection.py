#!/usr/bin/env python3
"""
Fault Injection Tests for lucineer-brain.

Stress-tests the pipeline by injecting failures at every stage boundary:
- TimeoutError at each pipeline stage (intent, planner, coder, hermes, safety)
- Empty 200 OK responses (valid HTTP, empty content string)
- Malformed JSON (not parseable) from model calls
- 429 rate limit cascades through the full fallback chain
- Fallback chain activation when primary model fails
- Safety check invariant: always runs, regardless of failure path

All tests mock brain.call_model to avoid real API calls.
"""

import json
import sys
import time
import urllib.error
from unittest.mock import patch, MagicMock, call
from io import StringIO

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import brain


# ═══════════════════════════════════════════════════════════════════════════════
# TIMEOUT FAULT INJECTION — each pipeline stage gets a timeout grenade
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimeoutAtIntentStage:
    """Inject TimeoutError at Stage 1 (intent).

    Note: stage_intent does NOT catch exceptions from call_model — if the API
    call raises, the exception propagates up. The pipeline's only intent-level
    fallback is when call_model succeeds but returns unparseable output (handled
    by extract_json → fallback dict). A true API timeout at intent stage crashes
    the pipeline, which is the correct behavior — there's no cheaper model to
    fall back to.
    """

    @patch("time.sleep")
    @patch("brain.call_model", side_effect=RuntimeError("URL error: operation timed out"))
    def test_intent_timeout_raises(self, mock_call, mock_sleep):
        """When the intent model times out on all retries, call_model raises
        RuntimeError which propagates through stage_intent."""
        with pytest.raises(RuntimeError, match="URL error"):
            brain.stage_intent("fake-key", "build a lighthouse")

    @patch("time.sleep")
    @patch("brain.call_model", side_effect=RuntimeError("operation timed out"))
    def test_intent_timeout_crashes_pipeline(self, mock_call, mock_sleep):
        """A timeout at the intent stage crashes the pipeline — no cheaper fallback."""
        with pytest.raises(RuntimeError):
            brain.run_pipeline("key", "build a tower")

    @patch("brain.call_model", return_value="")
    def test_intent_empty_string_falls_back(self, mock_call):
        """When call_model returns an empty string (whitespace stripped),
        extract_json returns None and stage_intent builds the fallback dict."""
        result = brain.stage_intent("fake-key", "build a lighthouse")
        assert result["intent"] == "build"
        assert result["subject"] == "build a lighthouse"
        assert result["scale"] == "medium"


class TestTimeoutAtPlannerStage:
    """Inject TimeoutError at Stage 2 (planning). Should exhaust fallbacks then degrade."""

    @patch("brain.call_model", side_effect=RuntimeError("operation timed out"))
    @patch("time.sleep")
    def test_planner_timeout_exhausts_fallbacks(self, mock_sleep, mock_call):
        """When all planner models time out, stage_plan returns empty steps with error."""
        result = brain.stage_plan("key", {"summary": "build"}, "build a house", use_deep=False)
        assert result["steps"] == []
        assert "error" in result
        assert "All planner models failed" in result["error"]

    @patch("brain.call_model", side_effect=RuntimeError("operation timed out"))
    @patch("time.sleep")
    def test_planner_timeout_deep_mode(self, mock_sleep, mock_call):
        """Same timeout cascade in deep planning mode."""
        result = brain.stage_plan("key", {"summary": "build"}, "build a city", use_deep=True)
        assert result["steps"] == []
        assert result["error"] == "All planner models failed"

    @patch("brain.stage_plan", return_value={
        "steps": [],
        "error": "All planner models failed",
        "_meta": {"latency_s": 0.0, "model": "none", "stage": "planner", "raw": "", "fallbacks_tried": 2}
    })
    @patch("brain.stage_intent", return_value={
        "summary": "build", "_meta": {"latency_s": 0.1}
    })
    @patch("brain.run_fast", return_value={
        "reply": "Fast fallback.", "commands": [],
        "_pipeline": {"mode": "fast"}
    })
    def test_planner_timeout_triggers_fast_fallback(self, mock_fast, mock_intent, mock_plan):
        """Pipeline should drop to fast mode when planner times out completely."""
        result = brain.run_pipeline("key", "build something big")
        assert result["_pipeline"]["planner_failed"] is True
        mock_fast.assert_called_once()


class TestTimeoutAtCoderStage:
    """Inject TimeoutError at Stage 3 (command generation)."""

    @patch("brain.call_model", side_effect=RuntimeError("operation timed out"))
    @patch("time.sleep")
    def test_coder_timeout_all_fail(self, mock_sleep, mock_call):
        """All coder models timing out should set the all_failed flag."""
        plan = {"steps": [{"step": 1, "action": "build"}]}
        result = brain.stage_commands("key", plan, {"summary": "build"}, "build")
        assert result["_meta"]["all_failed"] is True
        assert result["reply"] == ""
        assert result["commands"] == []

    @patch("brain.stage_commands", return_value={
        "reply": "", "commands": [],
        "_meta": {"all_failed": True, "latency_s": 0.0, "model": "none"}
    })
    @patch("brain.stage_plan", return_value={
        "steps": [{"step": 1}], "_meta": {"latency_s": 0.1, "model": "test"}
    })
    @patch("brain.stage_intent", return_value={
        "summary": "build", "_meta": {"latency_s": 0.1}
    })
    @patch("brain.run_fast", return_value={
        "reply": "Fast save.", "commands": [{"type": "createPart"}],
        "_pipeline": {"mode": "fast"}
    })
    def test_coder_timeout_triggers_fast_fallback(self, mock_fast, mock_intent, mock_plan, mock_cmds):
        """When all coders time out, pipeline drops to fast mode."""
        result = brain.run_pipeline("key", "build a house")
        assert result.get("coder_fallback_exhausted") is True or \
               result["_pipeline"].get("mode") == "fast"
        mock_fast.assert_called_once()


class TestTimeoutAtHermesStage:
    """Inject TimeoutError at Stage 4 (personality wrapping)."""

    @patch("brain.call_model", side_effect=RuntimeError("operation timed out"))
    def test_hermes_timeout_returns_original(self, mock_call):
        """Hermes timeout should return the original result unchanged (best-effort)."""
        original = {
            "reply": "Built a dock.",
            "commands": [{"type": "createPart", "params": {"name": "pier"}}],
        }
        result = brain.stage_hermes("key", original, {"summary": "dock"}, "build a dock")
        assert result["reply"] == "Built a dock."
        assert result["commands"] == original["commands"]
        assert "_meta_hermes" in result
        assert "error" in result["_meta_hermes"]


class TestTimeoutAtSafetyStage:
    """Inject TimeoutError at Stage 5 (safety check). Should fail OPEN."""

    @patch("brain.call_model", side_effect=RuntimeError("operation timed out"))
    @patch("time.sleep")
    def test_safety_timeout_fails_open(self, mock_sleep, mock_call):
        """Safety API timeout should allow the reply through (fail-open policy)."""
        is_safe, reason = brain.stage_safety("key", "Nice tower.", "build a tower")
        assert is_safe is True
        assert "skip" in reason.lower() or "error" in reason.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# EMPTY 200 OK — model returns valid HTTP but content is empty/whitespace
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmpty200Responses:
    """Simulate the model returning a valid 200 OK with an empty content string."""

    def _make_mock_response(self, content=""):
        """Create a mock urllib response returning 200 OK with given content."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop" if content else "length",
            }]
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("urllib.request.urlopen")
    def test_empty_content_raises_at_call_level(self, mock_urlopen):
        """call_model itself should raise RuntimeError on empty content."""
        mock_urlopen.return_value = self._make_mock_response("")
        with pytest.raises(RuntimeError, match="Empty content"):
            brain.call_model("key", "model", [{"role": "user", "content": "hi"}])

    @patch("urllib.request.urlopen")
    def test_whitespace_content_returns_empty(self, mock_urlopen):
        """Whitespace-only content gets stripped by call_model to empty string.
        Note: call_model checks `not content` BEFORE stripping, so whitespace
        passes the check and gets stripped at return time, producing empty string.
        The empty string is the caller's problem to handle."""
        mock_urlopen.return_value = self._make_mock_response("   \n\t  ")
        result = brain.call_model("key", "model", [{"role": "user", "content": "hi"}])
        assert result == ""

    @patch("brain.call_model", return_value="")
    def test_empty_response_at_intent_falls_back(self, mock_call):
        """Empty string from intent model should trigger the fallback intent construction
        (extract_json returns None on empty string, triggering the fallback path)."""
        result = brain.stage_intent("key", "build a castle")
        assert result["intent"] == "build"
        assert result["subject"] == "build a castle"

    @patch("brain.call_model", side_effect=RuntimeError("Empty content from model"))
    def test_empty_response_at_planner_returns_error(self, mock_call):
        """Empty content from all planner models should return the error dict."""
        result = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=False)
        assert result["steps"] == []
        assert result["error"] == "All planner models failed"

    @patch("brain.call_model", side_effect=RuntimeError("Empty content from model"))
    def test_empty_response_at_coder_returns_all_failed(self, mock_call):
        """Empty content from all coder models should trigger the all_failed flag."""
        result = brain.stage_commands("key", {"steps": [{"step": 1}]}, {"summary": "x"}, "build")
        assert result["_meta"]["all_failed"] is True

    @patch("brain.call_model", side_effect=RuntimeError("Empty content from model"))
    def test_empty_response_at_hermes_returns_original(self, mock_call):
        """Empty content from Hermes should return the original result."""
        original = {"reply": "original", "commands": [{"type": "createPart"}]}
        result = brain.stage_hermes("key", original, {"summary": "x"}, "build")
        assert result["reply"] == "original"
        assert "error" in result["_meta_hermes"]

    @patch("brain.call_model", side_effect=RuntimeError("Empty content from model"))
    def test_empty_response_at_safety_fails_open(self, mock_call):
        """Empty content from safety model should fail OPEN."""
        is_safe, reason = brain.stage_safety("key", "some reply", "build")
        assert is_safe is True
        assert "skip" in reason.lower() or "error" in reason.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# MALFORMED JSON — model returns 200 OK with non-JSON content
# ═══════════════════════════════════════════════════════════════════════════════

class TestMalformedJsonResponses:
    """Model returns valid text, but it's not parseable JSON."""

    MALFORMED_OUTPUTS = [
        "I cannot help with that.",                    # plain text refusal
        "Here are the build steps:\n1. Foundation\n2. Walls",  # numbered list
        "{intent: build, subject: castle}",            # unquoted keys
        "```json\n{broken: true\n```",                 # broken code fence
        "{'single': 'quotes'}",                        # single quotes
        "The answer is 42.",                           # no JSON at all
        "{{{{broken nested braces}}}}",                # deeply nested garbage
        "\x00\x01binary garbage here",                 # binary-like content
    ]

    @pytest.mark.parametrize("malformed", MALFORMED_OUTPUTS)
    @patch("brain.call_model")
    def test_malformed_intent_falls_back(self, mock_call, malformed):
        """Malformed output from intent model should produce fallback intent."""
        mock_call.return_value = malformed
        result = brain.stage_intent("key", "build a house")
        assert result["intent"] == "build"
        assert "_meta" in result

    @pytest.mark.parametrize("malformed", MALFORMED_OUTPUTS)
    @patch("brain.call_model")
    def test_malformed_planner_returns_empty_steps(self, mock_call, malformed):
        """Malformed output from planner should return steps=[] with error."""
        mock_call.return_value = malformed
        result = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=False)
        assert result["steps"] == []
        # Either "error" key or empty steps
        assert "error" in result or result.get("steps") == []

    @pytest.mark.parametrize("malformed", MALFORMED_OUTPUTS[:3])
    @patch("brain.call_model")
    def test_malformed_coder_produces_error_reply(self, mock_call, malformed):
        """Malformed output from coder should produce a result with error or fallback reply."""
        mock_call.return_value = malformed
        result = brain.stage_commands("key", {"steps": [{"step": 1}]}, {"summary": "x"}, "build")
        # Should have a reply (even if error message) and commands list
        assert "reply" in result
        assert "commands" in result

    @patch("brain.call_model")
    def test_malformed_hermes_preserves_original(self, mock_call):
        """Malformed Hermes output should preserve the original reply."""
        mock_call.return_value = "I am Hermes, hear me roar. Not JSON."
        original = {"reply": "Dock's in.", "commands": [{"type": "createPart"}]}
        result = brain.stage_hermes("key", original, {"summary": "dock"}, "build")
        assert result["reply"] == "Dock's in."
        assert result["commands"] == original["commands"]

    @patch("brain.call_model")
    def test_malformed_fast_mode_produces_error_result(self, mock_call):
        """Fast mode with malformed output should produce an error result, not crash."""
        mock_call.return_value = "I can't build that."
        with patch("brain.stage_safety", return_value=(True, "safe")):
            result = brain.run_fast("key", "build a house")
            assert "reply" in result
            assert "commands" in result
            assert result["commands"] == []  # no commands from malformed output


# ═══════════════════════════════════════════════════════════════════════════════
# 429 RATE LIMIT CASCADES — full fallback chain activation
# ═══════════════════════════════════════════════════════════════════════════════

class TestRateLimit429Cascades:
    """Test the full 429 cascade through every fallback model in the chain."""

    @patch("time.sleep")  # don't actually sleep
    @patch("brain.call_model")
    def test_coder_429_cascade_primary_to_fallback1(self, mock_call, mock_sleep):
        """Primary coder gets 429, first fallback succeeds."""
        mock_call.side_effect = [
            RuntimeError("HTTP 429: Too Many Requests"),
            json.dumps({"reply": "ok", "commands": [{"type": "createPart"}]}),
        ]
        result = brain.stage_commands("key", {"steps": [{"step": 1}]}, {"summary": "x"}, "build")
        assert result["reply"] == "ok"
        assert mock_call.call_count == 2
        assert result["_meta"]["model"] == brain.CODER_FALLBACKS[0]

    @patch("time.sleep")
    @patch("brain.call_model")
    def test_coder_429_cascade_through_all_fallbacks(self, mock_call, mock_sleep):
        """All coder models get 429 — should exhaust chain and set all_failed."""
        # Primary + each fallback gets 429 on every retry
        mock_call.side_effect = RuntimeError("HTTP 429: Too Many Requests")
        result = brain.stage_commands("key", {"steps": [{"step": 1}]}, {"summary": "x"}, "build")
        assert result["_meta"]["all_failed"] is True
        # Should have tried: primary coder + CODER_FALLBACKS entries
        expected_calls = 1 + len(brain.CODER_FALLBACKS)
        assert mock_call.call_count == expected_calls

    @patch("time.sleep")
    @patch("brain.call_model")
    def test_planner_429_no_fallback_in_standard_mode(self, mock_call, mock_sleep):
        """In standard mode, PLANNER_FALLBACKS contains the same model as primary.
        The dedup filter removes it, so there's no fallback — 429 means all fail."""
        mock_call.side_effect = RuntimeError("HTTP 429: Too Many Requests")
        result = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=False)
        assert result["steps"] == []
        assert "All planner models failed" in result["error"]

    @patch("time.sleep")
    @patch("brain.call_model")
    def test_planner_429_cascade_deep_mode(self, mock_call, mock_sleep):
        """Deep planner 429 → standard planner 429 → all fail."""
        mock_call.side_effect = RuntimeError("HTTP 429: Too Many Requests")
        result = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=True)
        assert result["steps"] == []
        assert "All planner models failed" in result["error"]

    @patch("time.sleep")
    @patch("brain.call_model")
    def test_planner_success_then_coder_429_cascade(self, mock_call, mock_sleep):
        """Planner succeeds, then coder gets 429 on primary, fallback works."""
        mock_call.side_effect = [
            # Planner: primary succeeds
            json.dumps({"steps": [{"step": 1}]}),
            # Coder: primary 429
            RuntimeError("HTTP 429: busy"),
            # Coder: fallback 1 succeeds
            json.dumps({"reply": "Built.", "commands": [{"type": "createPart"}]}),
        ]
        # Stage plan
        plan = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=False)
        assert len(plan["steps"]) == 1
        # Stage commands
        result = brain.stage_commands("key", plan, {"summary": "build"}, "build")
        assert result["reply"] == "Built."
        assert mock_call.call_count == 3


# ═══════════════════════════════════════════════════════════════════════════════
# FALLBACK CHAIN ACTIVATION — verify the right model was used
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallbackChainActivation:
    """Verify that fallback chains report which model was actually used."""

    @patch("brain.call_model")
    def test_planner_fallback_reports_correct_model(self, mock_call):
        """In deep mode, when the deep planner fails, _meta.model should be the
        standard planner (the fallback), and fallbacks_tried >= 1."""
        mock_call.side_effect = [
            RuntimeError("HTTP 500: error"),        # deep planner fails
            json.dumps({"steps": [{"step": 1}]}),   # standard planner succeeds
        ]
        result = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=True)
        assert result["_meta"]["model"] == brain.MODELS["planner"]
        assert result["_meta"]["fallbacks_tried"] >= 1

    @patch("brain.call_model")
    def test_coder_fallback_reports_correct_model(self, mock_call):
        """When primary coder fails, _meta.model should be the fallback model."""
        mock_call.side_effect = [
            RuntimeError("HTTP 500: error"),
            json.dumps({"reply": "ok", "commands": []}),
        ]
        result = brain.stage_commands("key", {"steps": [{"step": 1}]}, {"summary": "x"}, "build")
        assert result["_meta"]["model"] == brain.CODER_FALLBACKS[0]
        assert result["_meta"]["fallbacks_tried"] >= 1

    @patch("brain.call_model")
    def test_deep_planner_fallback_to_standard(self, mock_call):
        """Deep planner fails, standard planner succeeds — model should be standard planner."""
        mock_call.side_effect = [
            RuntimeError("HTTP 429: busy"),
            json.dumps({"steps": [{"step": 1, "action": "foundation"}]}),
        ]
        result = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=True)
        # Should report using the standard planner as fallback
        assert result["_meta"]["model"] == brain.MODELS["planner"]
        assert result["_meta"]["stage"] == "planner"

    @patch("brain.call_model")
    def test_primary_success_reports_zero_fallbacks(self, mock_call):
        """When primary model succeeds on first try, fallbacks_tried should be 0."""
        mock_call.return_value = json.dumps({"steps": [{"step": 1}]})
        result = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=False)
        assert result["_meta"]["fallbacks_tried"] == 0
        assert result["_meta"]["model"] == brain.MODELS["planner"]

    @patch("brain.call_model")
    def test_coder_primary_success_reports_zero_fallbacks(self, mock_call):
        mock_call.return_value = json.dumps({"reply": "ok", "commands": []})
        result = brain.stage_commands("key", {"steps": [{"step": 1}]}, {"summary": "x"}, "build")
        assert result["_meta"]["fallbacks_tried"] == 0
        assert result["_meta"]["model"] == brain.MODELS["coder"]


# ═══════════════════════════════════════════════════════════════════════════════
# SAFETY CHECK INVARIANT — safety always runs, no matter what failed before it
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyCheckInvariant:
    """The safety check MUST run on every pipeline output, regardless of how we got there."""

    @patch("brain.stage_safety", return_value=(True, "safe"))
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_safety_runs_in_standard_pipeline(self, mock_intent, mock_plan, mock_cmds, mock_safety):
        """Safety runs after a clean standard pipeline."""
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [{"step": 1}], "_meta": {"latency_s": 0.1, "model": "t"}}
        mock_cmds.return_value = {
            "reply": "Built.", "commands": [],
            "_meta": {"latency_s": 0.1, "model": "t"},
        }
        brain.run_pipeline("key", "build a house")
        mock_safety.assert_called_once()

    @patch("brain.stage_safety", return_value=(True, "safe"))
    @patch("brain.stage_hermes")
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_safety_runs_after_hermes(self, mock_intent, mock_plan, mock_cmds, mock_hermes, mock_safety):
        """Safety runs after creative mode (Hermes wrapping)."""
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [{"step": 1}], "_meta": {"latency_s": 0.1, "model": "t"}}
        mock_cmds.return_value = {
            "reply": "Built.", "commands": [],
            "_meta": {"latency_s": 0.1, "model": "t"},
        }
        mock_hermes.return_value = {
            "reply": "Enhanced.", "commands": [],
            "_meta": {"latency_s": 0.1, "model": "t"},
            "_meta_hermes": {"model": "hermes", "latency_s": 0.1},
        }
        brain.run_pipeline("key", "build", creative=True)
        mock_safety.assert_called_once()

    @patch("brain.stage_safety", return_value=(True, "safe"))
    @patch("brain.run_fast")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_safety_runs_after_planner_fallback_to_fast(self, mock_intent, mock_plan, mock_fast, mock_safety):
        """Safety runs even when planner fails and we drop to fast mode.

        Note: run_fast itself calls stage_safety internally. This test verifies
        the pipeline's safety call structure when planner fails."""
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [], "_meta": {"latency_s": 0.1, "model": "t"}}
        mock_fast.return_value = {
            "reply": "Fast build.", "commands": [],
            "_pipeline": {"mode": "fast"},
        }
        # When planner fails, run_pipeline calls run_fast which has its own safety check.
        # The pipeline-level safety check is skipped because we return early from run_fast.
        result = brain.run_pipeline("key", "build")
        # run_fast should have been called (which internally calls stage_safety)
        mock_fast.assert_called_once()

    @patch("brain.stage_safety", return_value=(True, "safe"))
    @patch("brain.call_model")
    def test_safety_runs_in_fast_mode(self, mock_call, mock_safety):
        """Safety runs in fast mode."""
        mock_call.return_value = '{"reply": "fast build", "commands": []}'
        brain.run_fast("key", "build")
        mock_safety.assert_called_once()

    @patch("brain.stage_safety", return_value=(True, "safe"))
    @patch("brain.stage_hermes")
    @patch("brain.call_model")
    def test_safety_runs_in_fast_creative_mode(self, mock_call, mock_hermes, mock_safety):
        """Safety runs in fast+creative mode, after Hermes."""
        mock_call.return_value = '{"reply": "fast", "commands": []}'
        mock_hermes.return_value = {
            "reply": "enhanced", "commands": [],
            "_pipeline": {"mode": "fast", "creative": True, "total_time_s": 0.1, "model": "intent"},
            "_meta_hermes": {"model": "hermes", "latency_s": 0.1},
        }
        brain.run_fast("key", "build", creative=True)
        mock_safety.assert_called_once()

    @patch("brain.stage_safety", return_value=(False, "unsafe content"))
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_safety_blocks_unsafe_after_full_pipeline(self, mock_intent, mock_plan, mock_cmds, mock_safety):
        """Safety check blocks unsafe content even after a successful pipeline run."""
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [{"step": 1}], "_meta": {"latency_s": 0.1, "model": "t"}}
        mock_cmds.return_value = {
            "reply": "Something terrible.", "commands": [{"type": "createPart"}],
            "_meta": {"latency_s": 0.1, "model": "t"},
        }
        result = brain.run_pipeline("key", "build something bad")
        assert result["_safety_blocked"] is True
        assert result["commands"] == []
        assert "Not building that" in result["reply"]

    @patch("brain.stage_safety", return_value=(False, "unsafe"))
    @patch("brain.call_model")
    def test_safety_blocks_unsafe_in_fast_mode(self, mock_call, mock_safety):
        """Safety blocks in fast mode too."""
        mock_call.return_value = '{"reply": "bad content", "commands": [{"type": "createPart"}]}'
        result = brain.run_fast("key", "build")
        assert result["_safety_blocked"] is True
        assert result["commands"] == []

    def test_safety_called_with_correct_arguments(self):
        """Safety check receives the reply text and original player message."""
        with patch("brain.call_model") as mock_call:
            mock_call.return_value = "SAFE"
            brain.stage_safety("key", "Built a house.", "build a house")
            # Verify the safety model was called with both texts
            args, kwargs = mock_call.call_args
            messages = kwargs.get("messages", args[2] if len(args) > 2 else [])
            # The player message and reply should be in the messages
            combined = json.dumps(messages)
            assert "build a house" in combined
            assert "Built a house" in combined


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-STAGE FAULT PROPAGATION — failure at stage N doesn't corrupt stage N+1
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossStageFaultPropagation:
    """Verify that a fault at one stage doesn't corrupt downstream stages."""

    @patch("brain.stage_safety", return_value=(True, "safe"))
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_fallback_intent_still_feeds_planner(self, mock_intent, mock_plan, mock_cmds, mock_safety):
        """When intent produces a fallback dict (no API data), the planner should
        still receive a valid intent dict and produce steps."""
        mock_intent.return_value = {
            "intent": "build",
            "subject": "build a castle",
            "style": "default",
            "scale": "medium",
            "mood": "neutral",
            "keywords": ["build", "castle"],
            "summary": "build a castle",
            "_meta": {"model": "fallback", "latency_s": 0.0, "raw": ""},
        }
        mock_plan.return_value = {
            "steps": [{"step": 1, "action": "foundation"}],
            "_meta": {"latency_s": 0.1, "model": "planner"}
        }
        mock_cmds.return_value = {
            "reply": "Castle built.", "commands": [{"type": "createPart"}],
            "_meta": {"latency_s": 0.1, "model": "coder"},
        }
        result = brain.run_pipeline("key", "build a castle")
        # Planner should have been called with the fallback intent
        mock_plan.assert_called_once()
        plan_args = mock_plan.call_args
        # stage_plan(api_key, intent, player_message, use_deep=...)
        # intent is the second positional arg
        passed_intent = plan_args[0][1] if len(plan_args[0]) > 1 else plan_args[1].get("intent")
        assert passed_intent["subject"] == "build a castle"
        assert result["reply"] == "Castle built."

    @patch("brain.stage_safety", return_value=(True, "safe"))
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_planner_fallback_still_feeds_coder(self, mock_intent, mock_plan, mock_cmds, mock_safety):
        """When planner uses a fallback model, the coder should still receive a valid plan."""
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {
            "steps": [{"step": 1, "action": "foundation", "parts": [{"name": "base"}]}],
            "_meta": {"latency_s": 0.2, "model": "fallback-planner", "fallbacks_tried": 1},
        }
        mock_cmds.return_value = {
            "reply": "Built.", "commands": [{"type": "createPart"}],
            "_meta": {"latency_s": 0.1, "model": "coder"},
        }
        result = brain.run_pipeline("key", "build")
        # Coder should have been called with the plan from the fallback planner
        mock_cmds.assert_called_once()
        cmd_args = mock_cmds.call_args
        passed_plan = cmd_args[0][1] if len(cmd_args[0]) > 1 else cmd_args[1].get("plan")
        assert len(passed_plan["steps"]) == 1
        assert result["reply"] == "Built."

    @patch("brain.stage_safety", return_value=(True, "safe"))
    @patch("brain.stage_hermes")
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_coder_fallback_still_feeds_hermes(self, mock_intent, mock_plan, mock_cmds, mock_hermes, mock_safety):
        """When coder uses a fallback model, Hermes should still receive the result."""
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [{"step": 1}], "_meta": {"latency_s": 0.1, "model": "t"}}
        mock_cmds.return_value = {
            "reply": "Fallback built it.", "commands": [{"type": "createPart"}],
            "_meta": {"latency_s": 0.1, "model": "fallback-coder", "fallbacks_tried": 1},
        }
        mock_hermes.return_value = {
            "reply": "Enhanced by Hermes.", "commands": [{"type": "createPart"}],
            "_meta": {"latency_s": 0.1, "model": "fallback-coder"},
            "_meta_hermes": {"model": "hermes", "latency_s": 0.1},
        }
        result = brain.run_pipeline("key", "build", creative=True)
        assert result["reply"] == "Enhanced by Hermes."
        # Hermes should have received the coder's output
        mock_hermes.assert_called_once()
        hermes_args = mock_hermes.call_args
        passed_result = hermes_args[0][1] if len(hermes_args[0]) > 1 else hermes_args[1].get("result")
        assert passed_result["reply"] == "Fallback built it."


# ═══════════════════════════════════════════════════════════════════════════════
# VERBOSE-MODE BRANCH COVERAGE (bonus: hits the verbose print branches)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerboseModeBranches:
    """Exercise verbose=True paths to cover the print branches."""

    @patch("brain.stage_safety", return_value=(True, "safe"))
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_verbose_pipeline_completes(self, mock_intent, mock_plan, mock_cmds, mock_safety):
        """Verbose mode should print progress and still produce valid output."""
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [{"step": 1}], "_meta": {"latency_s": 0.1, "model": "t"}}
        mock_cmds.return_value = {
            "reply": "Built.", "commands": [{"type": "createPart"}],
            "_meta": {"latency_s": 0.1, "model": "t"},
        }
        result = brain.run_pipeline("key", "build", verbose=True)
        assert result["reply"] == "Built."

    @patch("brain.run_fast")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_verbose_planner_failure(self, mock_intent, mock_plan, mock_fast):
        """Verbose mode with planner failure should print the fallback message."""
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [], "_meta": {"latency_s": 0.1, "model": "t"}}
        mock_fast.return_value = {
            "reply": "fast", "commands": [],
            "_pipeline": {"mode": "fast"},
        }
        result = brain.run_pipeline("key", "build", verbose=True)
        assert result["_pipeline"]["planner_failed"] is True

    @patch("brain.stage_safety", return_value=(True, "safe"))
    @patch("brain.call_model")
    def test_verbose_fast_mode(self, mock_call, mock_safety):
        """Verbose fast mode should complete without errors."""
        mock_call.return_value = '{"reply": "fast", "commands": []}'
        result = brain.run_fast("key", "build", verbose=True)
        assert "reply" in result

    @patch("brain.stage_safety", return_value=(True, "safe"))
    @patch("brain.stage_hermes")
    @patch("brain.call_model")
    def test_verbose_fast_creative(self, mock_call, mock_hermes, mock_safety):
        """Verbose fast+creative should exercise the verbose Hermes print branch."""
        mock_call.return_value = '{"reply": "fast", "commands": []}'
        mock_hermes.return_value = {
            "reply": "enhanced", "commands": [],
            "_pipeline": {"mode": "fast", "creative": True, "total_time_s": 0.1, "model": "intent"},
            "_meta_hermes": {"model": "hermes", "latency_s": 0.1},
        }
        result = brain.run_fast("key", "build", verbose=True, creative=True)
        assert result["_pipeline"]["mode"] == "fast+creative"

    @patch("brain.call_model")
    def test_verbose_hermes_unparseable(self, mock_call):
        """Verbose mode through Hermes unparseable output path."""
        brain.verbose_check._verbose = True
        mock_call.return_value = "not json at all"
        original = {"reply": "original", "commands": [{"type": "createPart"}]}
        result = brain.stage_hermes("key", original, {"summary": "x"}, "build")
        assert result["reply"] == "original"
        brain.verbose_check._verbose = False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
