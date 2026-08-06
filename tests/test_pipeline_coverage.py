"""
Coverage expansion for brain.py pipeline stages, safety, health check, and CLI.

Targets the large uncovered regions:
- stage_intent (line ~670)
- stage_plan fallback chains (lines ~676-802)
- stage_commands fallback chains (lines ~802-918)
- stage_safety edge cases (lines ~918-940)
- stage_hermes success/failure paths (lines ~922-1012)
- run_pipeline verbose/quiet, planner fallback, coder fallback (lines ~1044-1111)
- run_fast success/failure/safety/creative (lines ~1112-1238)
- health_check (lines ~1212-1238)
- test_models (lines ~1244-1343)
- main / CLI (line ~1347)

All tests mock call_model to avoid real API calls.
"""

import json
import sys
import time
from unittest.mock import patch, MagicMock, call
from io import StringIO

import pytest

# Ensure we can import brain
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import brain


# ─── stage_intent ────────────────────────────────────────────────────────────

class TestStageIntent:
    @patch("brain.call_model")
    def test_successful_intent_parse(self, mock_call):
        mock_call.return_value = '{"intent": "build", "subject": "castle", "style": "medieval", "scale": "large", "mood": "epic", "keywords": ["tower", "wall"], "summary": "build a big castle"}'
        result = brain.stage_intent("key", "build a castle")
        assert result["intent"] == "build"
        assert result["subject"] == "castle"
        assert "_meta" in result
        assert "latency_s" in result["_meta"]
        assert result["_meta"]["model"] == brain.MODELS["intent"]

    @patch("brain.call_model")
    def test_intent_unparseable_fallback(self, mock_call):
        mock_call.return_value = "garbage not json"
        result = brain.stage_intent("key", "build something cool")
        assert result["intent"] == "build"
        assert result["subject"] == "build something cool"
        assert result["scale"] == "medium"
        assert result["mood"] == "neutral"
        assert result["keywords"] == ["build", "something", "cool"]

    @patch("brain.call_model")
    def test_intent_empty_response(self, mock_call):
        mock_call.return_value = ""
        result = brain.stage_intent("key", "make a house")
        assert result["summary"] == "make a house"


# ─── stage_plan ──────────────────────────────────────────────────────────────

class TestStagePlan:
    @patch("brain.call_model")
    def test_plan_success_standard_mode(self, mock_call):
        mock_call.return_value = '{"steps": [{"step": 1, "action": "foundation", "parts": []}]}'
        result = brain.stage_plan("key", {"summary": "build a house"}, "build a house", use_deep=False)
        assert len(result["steps"]) == 1
        assert result["_meta"]["model"] == brain.MODELS["planner"]

    @patch("brain.call_model")
    def test_plan_success_deep_mode(self, mock_call):
        mock_call.return_value = '{"steps": [{"step": 1, "action": "deep foundation"}]}'
        result = brain.stage_plan("key", {"summary": "build a city"}, "build a city", use_deep=True)
        assert result["_meta"]["model"] == brain.MODELS["deep"]

    @patch("brain.call_model")
    def test_plan_all_models_fail(self, mock_call):
        mock_call.side_effect = RuntimeError("HTTP 429: busy")
        result = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=False)
        assert result["steps"] == []
        assert result["error"] == "All planner models failed"
        assert len(result["details"]) > 0

    @patch("brain.call_model")
    def test_plan_all_models_fail_deep_mode(self, mock_call):
        mock_call.side_effect = RuntimeError("HTTP 500: error")
        result = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=True)
        assert result["steps"] == []
        assert result["error"] == "All planner models failed"

    @patch("brain.call_model")
    def test_plan_unparseable_response(self, mock_call):
        mock_call.return_value = "not json at all"
        result = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=False)
        assert result["steps"] == []
        assert "error" in result

    @patch("brain.call_model")
    def test_plan_fallback_success(self, mock_call):
        """Deep planner fails, standard planner succeeds as fallback."""
        mock_call.side_effect = [RuntimeError("429"), '{"steps": [{"step": 1, "action": "build"}]}']
        result = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=True)
        assert mock_call.call_count == 2
        assert result["_meta"]["model"] == brain.MODELS["planner"]

    @patch("brain.call_model")
    def test_plan_deep_fallback_to_standard(self, mock_call):
        """Deep planner fails, standard planner succeeds."""
        mock_call.side_effect = [RuntimeError("429"), '{"steps": [{"step": 1}]}']
        result = brain.stage_plan("key", {"summary": "build"}, "build", use_deep=True)
        assert len(result["steps"]) == 1


# ─── stage_commands ──────────────────────────────────────────────────────────

class TestStageCommands:
    @patch("brain.call_model")
    def test_commands_success(self, mock_call):
        mock_call.return_value = '{"reply": "Done.", "commands": [{"type": "createPart", "params": {"name": "wall"}}]}'
        plan = {"steps": [{"step": 1, "action": "build"}]}
        result = brain.stage_commands("key", plan, {"summary": "build"}, "build a house")
        assert result["reply"] == "Done."
        assert len(result["commands"]) == 1
        assert "_meta" in result

    @patch("brain.call_model")
    def test_commands_all_fail(self, mock_call):
        mock_call.side_effect = RuntimeError("HTTP 429")
        plan = {"steps": [{"step": 1}]}
        result = brain.stage_commands("key", plan, {"summary": "build"}, "build")
        assert result["error"] == "All coder models failed"
        assert result["_meta"]["all_failed"] is True

    @patch("brain.call_model")
    def test_commands_unparseable(self, mock_call):
        mock_call.return_value = "not json"
        plan = {"steps": [{"step": 1}]}
        result = brain.stage_commands("key", plan, {"summary": "build"}, "build")
        assert "error" in result
        assert result["_meta"]["fallbacks_tried"] == 0

    @patch("brain.call_model")
    def test_commands_fallback_success(self, mock_call):
        """Primary coder fails, fallback succeeds."""
        mock_call.side_effect = [RuntimeError("429"), '{"reply": "ok", "commands": []}']
        plan = {"steps": [{"step": 1}]}
        result = brain.stage_commands("key", plan, {"summary": "build"}, "build")
        assert result["reply"] == "ok"
        assert mock_call.call_count == 2


# ─── stage_safety ────────────────────────────────────────────────────────────

class TestStageSafety:
    def test_empty_reply_is_safe(self):
        is_safe, reason = brain.stage_safety("key", "", "build a house")
        assert is_safe is True
        assert reason == "empty reply"

    def test_whitespace_reply_is_safe(self):
        is_safe, reason = brain.stage_safety("key", "   ", "build")
        assert is_safe is True
        assert reason == "empty reply"

    @patch("brain.call_model")
    def test_safe_verdict(self, mock_call):
        mock_call.return_value = "SAFE"
        is_safe, reason = brain.stage_safety("key", "Built a tower.", "build a tower")
        assert is_safe is True
        assert reason == "safe"

    @patch("brain.call_model")
    def test_unsafe_verdict_with_reason(self, mock_call):
        mock_call.return_value = "UNSAFE: violence detected"
        is_safe, reason = brain.stage_safety("key", "Kill everyone", "build")
        assert is_safe is False
        assert "VIOLENCE" in reason.upper()

    @patch("brain.call_model")
    def test_unsafe_verdict_no_colon(self, mock_call):
        mock_call.return_value = "UNSAFE something bad"
        is_safe, reason = brain.stage_safety("key", "bad reply", "build")
        assert is_safe is False
        assert reason == "unspecified"

    @patch("brain.call_model")
    def test_ambiguous_verdict(self, mock_call):
        mock_call.return_value = "MAYBE"
        is_safe, reason = brain.stage_safety("key", "some reply", "build")
        assert is_safe is False
        assert "ambiguous" in reason

    @patch("brain.call_model")
    def test_api_error_fails_open(self, mock_call):
        mock_call.side_effect = Exception("Connection refused")
        is_safe, reason = brain.stage_safety("key", "some reply", "build")
        assert is_safe is True
        assert "skipped" in reason


# ─── stage_hermes ────────────────────────────────────────────────────────────

class TestStageHermes:
    @patch("brain.call_model")
    def test_hermes_success(self, mock_call):
        mock_call.return_value = '{"reply": "Tower\'s up. Left the top open.", "commands": []}'
        result = {"reply": "original", "commands": [{"type": "createPart"}]}
        intent = {"summary": "build", "mood": "neutral"}
        out = brain.stage_hermes("key", result, intent, "build a tower")
        assert "Tower" in out["reply"]
        assert out["_meta_hermes"]["model"] == brain.MODELS["hermes"]
        # Commands should be preserved from original
        assert len(out["commands"]) == 1

    @patch("brain.call_model")
    def test_hermes_unparseable_output(self, mock_call):
        mock_call.return_value = "not json"
        result = {"reply": "original", "commands": [{"type": "createPart"}]}
        intent = {"summary": "build"}
        out = brain.stage_hermes("key", result, intent, "build")
        assert out["reply"] == "original"
        assert "error" in out["_meta_hermes"]

    @patch("brain.call_model")
    def test_hermes_runtime_error(self, mock_call):
        mock_call.side_effect = RuntimeError("API unavailable")
        result = {"reply": "original", "commands": []}
        intent = {"summary": "build"}
        out = brain.stage_hermes("key", result, intent, "build")
        assert out["reply"] == "original"
        assert "error" in out["_meta_hermes"]


# ─── run_pipeline ────────────────────────────────────────────────────────────

class TestRunPipeline:
    @patch("brain.run_fast")
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_full_pipeline_standard(self, mock_intent, mock_plan, mock_cmds, _):
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [{"step": 1}], "_meta": {"latency_s": 0.2, "model": "test"}}
        mock_cmds.return_value = {
            "reply": "Built it.",
            "commands": [{"type": "createPart"}],
            "_meta": {"latency_s": 0.3, "model": "coder"},
        }
        result = brain.run_pipeline("key", "build a house")
        assert result["reply"] == "Built it."
        assert result["_pipeline"]["mode"] == "standard"
        assert "stage_times_s" in result["_pipeline"]

    @patch("brain.stage_safety")
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_pipeline_creative_mode(self, mock_intent, mock_plan, mock_cmds, mock_safety):
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [{"step": 1}], "_meta": {"latency_s": 0.2, "model": "test"}}
        mock_cmds.return_value = {
            "reply": "Built it.",
            "commands": [],
            "_meta": {"latency_s": 0.3, "model": "coder"},
        }
        mock_safety.return_value = (True, "safe")
        with patch("brain.stage_hermes") as mock_hermes:
            mock_hermes.return_value = {
                "reply": "Tower's up.",
                "commands": [],
                "_meta": {"latency_s": 0.3, "model": "coder"},
                "_meta_hermes": {"model": brain.MODELS["hermes"], "latency_s": 0.2},
            }
            result = brain.run_pipeline("key", "build", creative=True)
            assert result["_pipeline"]["creative"] is True

    @patch("brain.stage_safety")
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_pipeline_unsafe_reply_blocked(self, mock_intent, mock_plan, mock_cmds, mock_safety):
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [{"step": 1}], "_meta": {"latency_s": 0.2, "model": "test"}}
        mock_cmds.return_value = {
            "reply": "Bad content.",
            "commands": [{"type": "createPart"}],
            "_meta": {"latency_s": 0.3, "model": "coder"},
        }
        mock_safety.return_value = (False, "unsafe content")
        result = brain.run_pipeline("key", "build something bad")
        assert result["_safety_blocked"] is True
        assert result["commands"] == []
        assert "Not building that" in result["reply"]

    @patch("brain.run_fast")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_pipeline_planner_no_steps_fallback(self, mock_intent, mock_plan, mock_fast):
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [], "_meta": {"latency_s": 0.2, "model": "test"}}
        mock_fast.return_value = {"reply": "fast fallback", "commands": [], "_pipeline": {"mode": "fast"}}
        result = brain.run_pipeline("key", "build")
        assert result["_pipeline"]["planner_failed"] is True
        assert "fast" in result["_pipeline"]["mode"]

    @patch("brain.run_fast")
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_pipeline_coder_all_failed_fallback(self, mock_intent, mock_plan, mock_cmds, mock_fast):
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [{"step": 1}], "_meta": {"latency_s": 0.2, "model": "test"}}
        mock_cmds.return_value = {
            "reply": "",
            "commands": [],
            "_meta": {"all_failed": True, "latency_s": 0.3},
        }
        mock_fast.return_value = {"reply": "fast", "commands": [], "_pipeline": {"mode": "fast"}}
        result = brain.run_pipeline("key", "build")
        assert result.get("coder_fallback_exhausted") is True or result["_pipeline"].get("mode") == "fast"

    @patch("brain.stage_safety")
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_pipeline_verbose_mode(self, mock_intent, mock_plan, mock_cmds, mock_safety):
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [{"step": 1}], "_meta": {"latency_s": 0.2, "model": "test"}}
        mock_cmds.return_value = {
            "reply": "Built.",
            "commands": [],
            "_meta": {"latency_s": 0.3, "model": "coder"},
        }
        mock_safety.return_value = (True, "safe")
        # Just verify it doesn't crash with verbose=True
        result = brain.run_pipeline("key", "build", verbose=True)
        assert "reply" in result

    @patch("brain.stage_safety")
    @patch("brain.stage_commands")
    @patch("brain.stage_plan")
    @patch("brain.stage_intent")
    def test_pipeline_deep_mode(self, mock_intent, mock_plan, mock_cmds, mock_safety):
        mock_intent.return_value = {"summary": "build", "_meta": {"latency_s": 0.1}}
        mock_plan.return_value = {"steps": [{"step": 1}], "_meta": {"latency_s": 0.2, "model": "deep"}}
        mock_cmds.return_value = {
            "reply": "Built deep.",
            "commands": [],
            "_meta": {"latency_s": 0.3, "model": "coder"},
        }
        mock_safety.return_value = (True, "safe")
        result = brain.run_pipeline("key", "build", use_deep=True)
        assert result["_pipeline"]["mode"] == "deep"


# ─── run_fast ────────────────────────────────────────────────────────────────

class TestRunFast:
    @patch("brain.stage_safety")
    @patch("brain.call_model")
    def test_fast_success(self, mock_call, mock_safety):
        mock_call.return_value = '{"reply": "Fast build.", "commands": [{"type": "createPart"}]}'
        mock_safety.return_value = (True, "safe")
        result = brain.run_fast("key", "build a house")
        assert result["reply"] == "Fast build."
        assert result["_pipeline"]["mode"] == "fast"

    @patch("brain.stage_safety")
    @patch("brain.call_model")
    def test_fast_unparseable(self, mock_call, mock_safety):
        mock_call.return_value = "not json"
        mock_safety.return_value = (True, "safe")
        result = brain.run_fast("key", "build")
        assert "error" in result
        assert result["commands"] == []

    @patch("brain.stage_safety")
    @patch("brain.call_model")
    def test_fast_unsafe_blocked(self, mock_call, mock_safety):
        mock_call.return_value = '{"reply": "bad", "commands": [{"type": "createPart"}]}'
        mock_safety.return_value = (False, "unsafe")
        result = brain.run_fast("key", "build")
        assert result["_safety_blocked"] is True
        assert result["commands"] == []

    @patch("brain.stage_safety")
    @patch("brain.stage_hermes")
    @patch("brain.call_model")
    def test_fast_creative_mode(self, mock_call, mock_hermes, mock_safety):
        mock_call.return_value = '{"reply": "Fast build.", "commands": []}'
        # stage_hermes returns a dict; run_fast mutates parsed["_pipeline"]["mode"]
        # so the returned dict must contain _pipeline
        mock_hermes.return_value = {
            "reply": "Enhanced reply.",
            "commands": [],
            "_pipeline": {"mode": "fast", "creative": True, "total_time_s": 0.1, "model": brain.MODELS["intent"]},
            "_meta_hermes": {"model": brain.MODELS["hermes"], "latency_s": 0.1},
        }
        mock_safety.return_value = (True, "safe")
        result = brain.run_fast("key", "build", creative=True)
        assert result["_pipeline"]["mode"] == "fast+creative"
        assert result["reply"] == "Enhanced reply."

    @patch("brain.stage_safety")
    @patch("brain.call_model")
    def test_fast_verbose(self, mock_call, mock_safety):
        mock_call.return_value = '{"reply": "ok", "commands": []}'
        mock_safety.return_value = (True, "safe")
        result = brain.run_fast("key", "build", verbose=True)
        assert "reply" in result


# ─── health_check ────────────────────────────────────────────────────────────

class TestHealthCheck:
    @patch("brain.load_api_key")
    def test_no_api_key(self, mock_load):
        mock_load.side_effect = RuntimeError("No key")
        result = brain.health_check()
        assert result["api_key_loaded"] is False
        assert result["can_reach_api"] is False
        assert "timestamp" in result
        assert len(result["models_configured"]) > 0

    @patch("brain.call_model")
    @patch("brain.load_api_key")
    def test_api_reachable(self, mock_load, mock_call):
        mock_load.return_value = "fake-key"
        mock_call.return_value = "ok"
        result = brain.health_check()
        assert result["api_key_loaded"] is True
        assert result["can_reach_api"] is True

    @patch("brain.call_model")
    @patch("brain.load_api_key")
    def test_api_unreachable(self, mock_load, mock_call):
        mock_load.return_value = "fake-key"
        mock_call.side_effect = Exception("Connection refused")
        result = brain.health_check()
        assert result["api_key_loaded"] is True
        assert result["can_reach_api"] is False

    def test_health_check_with_explicit_key(self):
        with patch("brain.call_model") as mock_call:
            mock_call.side_effect = Exception("fail")
            result = brain.health_check(api_key="explicit-key")
            assert result["api_key_loaded"] is True
            assert result["can_reach_api"] is False


# ─── test_models ─────────────────────────────────────────────────────────────

class TestTestModels:
    @patch("brain.call_model")
    def test_all_models_succeed(self, mock_call, capsys):
        mock_call.return_value = "Hello"
        brain.test_models("key")
        captured = capsys.readouterr()
        assert "Testing DeepInfra" in captured.out
        assert "✓" in captured.out

    @patch("brain.call_model")
    def test_all_models_fail(self, mock_call, capsys):
        mock_call.side_effect = RuntimeError("HTTP 500")
        brain.test_models("key")
        captured = capsys.readouterr()
        assert "✗" in captured.out


# ─── CLI / main ──────────────────────────────────────────────────────────────

class TestCli:
    def test_no_args_prints_help_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["brain.py"]):
                brain.main()
        assert exc_info.value.code == 1

    def test_no_api_key_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["brain.py", "build", "something"]):
                with patch("brain.load_api_key", side_effect=RuntimeError("no key")):
                    brain.main()
        assert exc_info.value.code == 1

    @patch("brain.run_pipeline")
    @patch("brain.load_api_key")
    def test_cli_default_pipeline(self, mock_key, mock_pipeline, capsys):
        mock_key.return_value = "fake"
        mock_pipeline.return_value = {"reply": "Built it.", "commands": []}
        with patch("sys.argv", ["brain.py", "build", "a", "house"]):
            brain.main()
        captured = capsys.readouterr()
        result = json.loads(captured.out.strip())
        assert result["reply"] == "Built it."

    @patch("brain.run_fast")
    @patch("brain.load_api_key")
    def test_cli_fast_mode(self, mock_key, mock_fast, capsys):
        mock_key.return_value = "fake"
        mock_fast.return_value = {"reply": "Fast.", "commands": []}
        with patch("sys.argv", ["brain.py", "--fast", "build"]):
            brain.main()
        captured = capsys.readouterr()
        result = json.loads(captured.out.strip())
        assert result["reply"] == "Fast."

    @patch("brain.run_pipeline")
    @patch("brain.load_api_key")
    def test_cli_pretty_output(self, mock_key, mock_pipeline, capsys):
        mock_key.return_value = "fake"
        mock_pipeline.return_value = {"reply": "Built.", "commands": []}
        with patch("sys.argv", ["brain.py", "--pretty", "build"]):
            brain.main()
        captured = capsys.readouterr()
        # Pretty output has indentation
        assert "  " in captured.out

    @patch("brain.run_pipeline")
    @patch("brain.load_api_key")
    def test_cli_pipeline_error(self, mock_key, mock_pipeline, capsys):
        mock_key.return_value = "fake"
        mock_pipeline.side_effect = Exception("Pipeline failed")
        with patch("sys.argv", ["brain.py", "build"]):
            with pytest.raises(SystemExit) as exc_info:
                brain.main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        result = json.loads(captured.out.strip())
        assert "Build failed" in result["reply"]

    @patch("brain.test_models")
    @patch("brain.load_api_key")
    def test_cli_test_mode(self, mock_key, mock_test):
        mock_key.return_value = "fake"
        with patch("sys.argv", ["brain.py", "--test"]):
            brain.main()
        mock_test.assert_called_once()

    @patch("brain.run_pipeline")
    @patch("brain.load_api_key")
    def test_cli_deep_and_creative_flags(self, mock_key, mock_pipeline):
        mock_key.return_value = "fake"
        mock_pipeline.return_value = {"reply": "Built.", "commands": []}
        with patch("sys.argv", ["brain.py", "--deep", "--creative", "build"]):
            brain.main()
        args, kwargs = mock_pipeline.call_args
        assert kwargs.get("use_deep") is True
        assert kwargs.get("creative") is True


# ─── call_model edge cases ───────────────────────────────────────────────────

class TestCallModelEdgeCases:
    @patch("brain.urllib.request.urlopen")
    def test_no_choices_returned(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"choices": []}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        with pytest.raises(RuntimeError, match="No choices"):
            brain.call_model("key", "model", [])

    @patch("brain.urllib.request.urlopen")
    def test_empty_content_with_reasoning(self, mock_urlopen):
        """Reasoning models put output in reasoning_content."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "", "reasoning_content": "from reasoning"}, "finish_reason": "stop"}]
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        result = brain.call_model("key", "model", [])
        assert result == "from reasoning"

    @patch("brain.urllib.request.urlopen")
    def test_empty_content_no_reasoning(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}]
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        with pytest.raises(RuntimeError, match="Empty content"):
            brain.call_model("key", "model", [])

    @patch("brain.urllib.request.urlopen")
    def test_http_error_non_429(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 500, "Server Error", {},
            MagicMock(read=MagicMock(return_value=b"internal server error"))
        )
        with pytest.raises(RuntimeError, match="HTTP 500"):
            brain.call_model("key", "model", [], max_retries=1)

    @patch("brain.urllib.request.urlopen")
    def test_url_error_non_timeout(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        with pytest.raises(RuntimeError, match="URL error"):
            brain.call_model("key", "model", [], max_retries=1)

    @patch("brain.urllib.request.urlopen")
    def test_successful_call_strips_whitespace(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "  trimmed  "}, "finish_reason": "stop"}]
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp
        result = brain.call_model("key", "model", [])
        assert result == "trimmed"


# ─── extract_json edge cases ─────────────────────────────────────────────────

class TestExtractJsonEdgeCases:
    def test_json_embedded_in_text(self):
        text = 'Here is the result: {"reply": "ok", "commands": []} done'
        result = brain.extract_json(text)
        assert result == {"reply": "ok", "commands": []}

    def test_json_array_embedded_in_text(self):
        text = 'Result: [1, 2, 3] trailing'
        result = brain.extract_json(text)
        assert result == [1, 2, 3]

    def test_no_json_at_all(self):
        assert brain.extract_json("just text") is None

    def test_empty_string(self):
        assert brain.extract_json("") is None

    def test_markdown_fence_with_language(self):
        text = '```json\n{"a": 1}\n```'
        assert brain.extract_json(text) == {"a": 1}

    def test_malformed_json_returns_none(self):
        assert brain.extract_json("{broken json}}}") is None

    def test_nested_braces(self):
        text = '{"a": {"b": {"c": 1}}}'
        assert brain.extract_json(text) == {"a": {"b": {"c": 1}}}


# ─── persona_for edge cases ──────────────────────────────────────────────────

class TestPersonaFor:
    def test_tier_0_low_bond(self):
        p = brain.persona_for(0)
        assert "RELATIONSHIP" not in p

    def test_tier_boundaries(self):
        # Tier 0: < 10
        assert "RELATIONSHIP" not in brain.persona_for(9)
        # Tier 1: 10-29
        assert "RELATIONSHIP" in brain.persona_for(10)
        assert "RELATIONSHIP" in brain.persona_for(29)
        # Tier 2: 30-69
        assert "RELATIONSHIP" in brain.persona_for(30)
        assert "RELATIONSHIP" in brain.persona_for(69)
        # Tier 3: 70-149
        assert "RELATIONSHIP" in brain.persona_for(70)
        assert "RELATIONSHIP" in brain.persona_for(149)
        # Tier 4: 150+
        assert "RELATIONSHIP" in brain.persona_for(150)
        assert "RELATIONSHIP" in brain.persona_for(999)

    def test_tier_content_differs(self):
        p1 = brain.persona_for(10)
        p4 = brain.persona_for(150)
        assert p1 != p4


# ─── cache_stats / cache_clear ───────────────────────────────────────────────

class TestCacheUtilities:
    def test_cache_stats_empty(self):
        brain._CACHE.clear()
        stats = brain.cache_stats()
        assert stats["entries"] == 0
        assert stats["max_entries"] == brain._CACHE_MAX

    def test_cache_clear_returns_count(self):
        brain._CACHE.clear()
        brain._CACHE["a"] = {"response": {}, "_time": 0}
        brain._CACHE["b"] = {"response": {}, "_time": 0}
        count = brain.cache_clear()
        assert count == 2
        assert len(brain._CACHE) == 0

    def test_cache_eviction_on_full(self):
        brain._CACHE.clear()
        # Fill to max
        for i in range(brain._CACHE_MAX):
            brain._CACHE[f"key{i}"] = {"response": {}, "_time": float(i)}
        # Adding one more should evict the oldest
        brain._cache_set("new_key", {"data": "new"})
        assert len(brain._CACHE) == brain._CACHE_MAX
        assert "key0" not in brain._CACHE
        assert "new_key" in brain._CACHE

    def test_cache_ttl_expiry(self):
        brain._CACHE.clear()
        # Insert with old timestamp
        brain._CACHE["old"] = {"response": {"old": True}, "_time": time.time() - brain._CACHE_TTL - 1}
        assert brain._cache_get("old") is None
        assert "old" not in brain._CACHE
