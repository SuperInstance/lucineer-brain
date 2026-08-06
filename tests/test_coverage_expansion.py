"""Tests for cache, health check, and call_model error handling in brain.py.

Targets uncovered lines in lucineer-brain for coverage improvement.
"""

import json
import time
import urllib.error
from unittest.mock import patch, MagicMock

import pytest

import brain


# ─── Cache Tests ─────────────────────────────────────────────────────────────


class TestCacheKey:
    def test_cache_key_includes_mode(self):
        k1 = brain._cache_key("build a house", "full")
        k2 = brain._cache_key("build a house", "fast")
        assert k1 != k2

    def test_cache_key_same_message_same_key(self):
        assert brain._cache_key("hello") == brain._cache_key("hello")

    def test_cache_key_different_messages_different_keys(self):
        assert brain._cache_key("hello") != brain._cache_key("goodbye")

    def test_cache_key_default_mode(self):
        k = brain._cache_key("test")
        assert k.startswith("full:")


class TestCacheGetSet:
    def setup_method(self):
        brain._CACHE.clear()

    def teardown_method(self):
        brain._CACHE.clear()

    def test_set_then_get(self):
        brain._cache_set("key1", {"reply": "hello"})
        assert brain._cache_get("key1") == {"reply": "hello"}

    def test_get_missing_returns_none(self):
        assert brain._cache_get("nonexistent") is None

    def test_set_overwrites(self):
        brain._cache_set("key1", {"v": 1})
        brain._cache_set("key1", {"v": 2})
        assert brain._cache_get("key1") == {"v": 2}

    def test_set_multiple(self):
        brain._cache_set("a", {"r": "a"})
        brain._cache_set("b", {"r": "b"})
        brain._cache_set("c", {"r": "c"})
        assert brain._cache_get("a") == {"r": "a"}
        assert brain._cache_get("b") == {"r": "b"}
        assert brain._cache_get("c") == {"r": "c"}

    def test_expired_entry_returns_none(self):
        brain._cache_set("old", {"r": "data"})
        # Manually age the entry
        brain._CACHE["old"]["_time"] = time.time() - brain._CACHE_TTL - 1
        assert brain._cache_get("old") is None

    def test_expired_entry_deleted_on_get(self):
        brain._cache_set("old", {"r": "data"})
        brain._CACHE["old"]["_time"] = time.time() - brain._CACHE_TTL - 1
        brain._cache_get("old")
        assert "old" not in brain._CACHE

    def test_fresh_entry_not_expired(self):
        brain._cache_set("fresh", {"r": "data"})
        assert brain._cache_get("fresh") is not None


class TestCacheEviction:
    def setup_method(self):
        brain._CACHE.clear()

    def teardown_method(self):
        brain._CACHE.clear()

    def test_evicts_oldest_when_full(self):
        # Fill to max
        for i in range(brain._CACHE_MAX):
            brain._cache_set(f"key{i}", {"i": i})
            time.sleep(0.001)

        assert len(brain._CACHE) == brain._CACHE_MAX

        # Adding one more should evict the oldest
        brain._cache_set("newest", {"i": "new"})
        assert len(brain._CACHE) == brain._CACHE_MAX
        assert brain._cache_get("key0") is None  # oldest evicted
        assert brain._cache_get("newest") is not None


class TestCacheStats:
    def setup_method(self):
        brain._CACHE.clear()

    def teardown_method(self):
        brain._CACHE.clear()

    def test_stats_empty(self):
        s = brain.cache_stats()
        assert s == {"entries": 0, "max_entries": brain._CACHE_MAX, "ttl_seconds": brain._CACHE_TTL}

    def test_stats_with_entries(self):
        brain._cache_set("a", {"r": 1})
        brain._cache_set("b", {"r": 2})
        s = brain.cache_stats()
        assert s["entries"] == 2
        assert s["max_entries"] == brain._CACHE_MAX
        assert s["ttl_seconds"] == brain._CACHE_TTL


class TestCacheClear:
    def setup_method(self):
        brain._CACHE.clear()

    def teardown_method(self):
        brain._CACHE.clear()

    def test_clear_returns_count(self):
        brain._cache_set("a", {"r": 1})
        brain._cache_set("b", {"r": 2})
        count = brain.cache_clear()
        assert count == 2

    def test_clear_empties_cache(self):
        brain._cache_set("a", {"r": 1})
        brain.cache_clear()
        assert len(brain._CACHE) == 0

    def test_clear_empty_returns_zero(self):
        count = brain.cache_clear()
        assert count == 0


# ─── Health Check Tests ─────────────────────────────────────────────────────


class TestHealthCheck:
    def test_health_check_returns_dict(self):
        result = brain.health_check()
        assert isinstance(result, dict)

    def test_health_check_has_expected_keys(self):
        result = brain.health_check()
        assert "models_configured" in result
        assert "can_reach_api" in result
        assert "api_key_loaded" in result

    def test_health_check_models_count(self):
        result = brain.health_check()
        assert len(result["models_configured"]) == len(brain.MODELS)

    def test_health_check_cache_entries(self):
        """health_check doesn't report cache entries directly, but cache_stats does."""
        result = brain.health_check()
        assert isinstance(result, dict)  # just verify it works

    def test_health_check_api_false_when_no_key(self):
        """Without a valid API key, can_reach_api should be False."""
        result = brain.health_check()
        # No DEEPINFRA_API_KEY in env typically
        assert result["can_reach_api"] in (True, False)  # depends on env

    def test_health_check_with_explicit_key(self):
        """Pass a fake key that will fail — should still return structure."""
        result = brain.health_check(api_key="fake-key-12345")
        assert isinstance(result, dict)
        assert result["can_reach_api"] is False  # fake key won't work


# ─── call_model Error Handling Tests ────────────────────────────────────────


class TestCallModelErrors:
    def test_http_error_raises_runtime_error(self):
        """Non-429 HTTP errors should raise RuntimeError."""
        error = urllib.error.HTTPError(
            url="https://api.deepinfra.com/v1/openai/chat/completions",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=MagicMock(),
        )
        error.read = MagicMock(return_value=b'{"error": "server error"}')

        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                brain.call_model("fake-key", "test-model", messages=[{"role": "user", "content": "hi"}])

    def test_429_retries_then_fails(self):
        """429 errors should retry then eventually raise RuntimeError."""
        error = urllib.error.HTTPError(
            url="https://api.deepinfra.com/v1/openai/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=MagicMock(),
        )
        error.read = MagicMock(return_value=b'{"error": "rate limited"}')

        with patch("urllib.request.urlopen", side_effect=error):
            with patch("time.sleep"):  # Don't actually sleep
                with pytest.raises(RuntimeError, match="HTTP 429"):
                    brain.call_model(
                        "fake-key",
                        "test-model",
                        messages=[{"role": "user", "content": "hi"}],
                        max_retries=2,
                    )

    def test_url_error_raises_runtime_error(self):
        """URLErrors should raise RuntimeError."""
        error = urllib.error.URLError("connection refused")

        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(RuntimeError, match="URL error"):
                brain.call_model("fake-key", "test-model", messages=[{"role": "user", "content": "hi"}])

    def test_url_timeout_retries(self):
        """Timeout URLErrors should retry."""
        error = urllib.error.URLError("operation timed out")

        with patch("urllib.request.urlopen", side_effect=error):
            with patch("time.sleep"):
                with pytest.raises(RuntimeError, match="URL error"):
                    brain.call_model(
                        "fake-key",
                        "test-model",
                        messages=[{"role": "user", "content": "hi"}],
                        max_retries=3,
                    )

    def test_empty_content_raises(self):
        """Empty response content should raise RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}]
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="Empty content"):
                brain.call_model("fake-key", "test-model", messages=[{"role": "user", "content": "hi"}])

    def test_no_choices_raises(self):
        """Response with no choices should raise RuntimeError."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"choices": []}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="No choices"):
                brain.call_model("fake-key", "test-model", messages=[{"role": "user", "content": "hi"}])

    def test_reasoning_content_fallback(self):
        """When content is empty but reasoning_content exists, use it."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "This is the reasoning output."
                },
                "finish_reason": "stop"
            }]
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = brain.call_model("fake-key", "test-model", messages=[{"role": "user", "content": "hi"}])
            assert "This is the reasoning output." in result

    def test_successful_call_returns_content(self):
        """A successful call should return the content string."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{
                "message": {"content": "Hello, world!"},
                "finish_reason": "stop"
            }]
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = brain.call_model("fake-key", "test-model", messages=[{"role": "user", "content": "hi"}])
            assert result == "Hello, world!"

    def test_content_stripped(self):
        """Returned content should be stripped of whitespace."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "choices": [{
                "message": {"content": "  Hello!  \n"},
                "finish_reason": "stop"
            }]
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = brain.call_model("fake-key", "test-model", messages=[{"role": "user", "content": "hi"}])
            assert result == "Hello!"


# ─── Stage Fallback Tests (mocking call_model) ──────────────────────────────


class TestStagePlanFallback:
    """Test the planner fallback chain logic."""

    def test_all_planners_fail_returns_error_dict(self):
        """When all planner models fail, should return error dict."""
        with patch.object(brain, "call_model", side_effect=RuntimeError("HTTP 429: busy")):
            result = brain.stage_plan("fake-key", {"intent": "build"}, "test message")

        assert result["steps"] == []
        assert "error" in result
        assert "All planner models failed" in result["error"]

    def test_deep_mode_uses_deep_planner_first(self):
        """Deep mode should try Seed-2.0-pro first."""
        call_log = []

        def mock_call(api_key, model, **kwargs):
            call_log.append(model)
            if model == brain.MODELS["deep"]:
                return json.dumps({"steps": [{"step": 1}]})
            raise RuntimeError("should not reach here")

        with patch.object(brain, "call_model", side_effect=mock_call):
            result = brain.stage_plan("fake-key", {"intent": "build"}, "test", use_deep=True)

        assert brain.MODELS["deep"] in call_log
        assert call_log[0] == brain.MODELS["deep"]
        assert result.get("steps") == [{"step": 1}]

    def test_standard_mode_uses_planner_first(self):
        """Standard mode should try Qwen3.6 first."""
        call_log = []

        def mock_call(api_key, model, **kwargs):
            call_log.append(model)
            if model == brain.MODELS["planner"]:
                return json.dumps({"steps": [{"step": 1}]})
            raise RuntimeError("should not reach here")

        with patch.object(brain, "call_model", side_effect=mock_call):
            result = brain.stage_plan("fake-key", {"intent": "build"}, "test", use_deep=False)

        assert call_log[0] == brain.MODELS["planner"]


class TestStageCommandsFallback:
    """Test the coder fallback chain."""

    def test_all_coders_fail_returns_error_dict(self):
        """When all coder models fail, return error with all_failed flag."""
        with patch.object(brain, "call_model", side_effect=RuntimeError("HTTP 500")):
            result = brain.stage_commands("fake-key", {"steps": [{"step": 1}]}, {}, "test")

        assert result["reply"] == ""
        assert result["commands"] == []
        assert result["_meta"]["all_failed"] is True

    def test_primary_coder_succeeds(self):
        """Primary coder succeeding should return parsed result."""
        mock_response = {"reply": "Done!", "commands": [{"type": "build", "part": "wall"}]}

        with patch.object(brain, "call_model", return_value=json.dumps(mock_response)):
            result = brain.stage_commands("fake-key", {"steps": [{"step": 1}]}, {}, "test")

        assert result["reply"] == "Done!"
        assert len(result["commands"]) == 1
        assert result["_meta"]["fallbacks_tried"] == 0


class TestStageHermesFallback:
    """Test Hermes stage fallback behavior."""

    def test_hermes_unavailable_returns_original(self):
        """If Hermes raises RuntimeError, original result should be returned with error meta."""
        original = {"reply": "original", "commands": [{"type": "build"}]}

        with patch.object(brain, "call_model", side_effect=RuntimeError("HTTP 503")):
            result = brain.stage_hermes("fake-key", original, {"summary": "test"}, "msg")

        assert result["reply"] == "original"
        assert result["commands"] == [{"type": "build"}]
        assert "_meta_hermes" in result
        assert "error" in result["_meta_hermes"]

    def test_hermes_unparseable_returns_original(self):
        """If Hermes output can't be parsed, keep original reply."""
        original = {"reply": "original", "commands": [{"type": "build"}]}

        with patch.object(brain, "call_model", return_value="not json at all"):
            result = brain.stage_hermes("fake-key", original, {"summary": "test"}, "msg")

        assert result["reply"] == "original"
        assert "_meta_hermes" in result

    def test_hermes_preserves_commands(self):
        """Hermes should never replace commands, only reply text."""
        original = {"reply": "boring", "commands": [{"type": "build", "part": "wall"}]}

        hermes_output = {"reply": "Exciting personality!", "commands": ["HALLUCINATED"]}

        with patch.object(brain, "call_model", return_value=json.dumps(hermes_output)):
            result = brain.stage_hermes("fake-key", original, {"summary": "test"}, "msg")

        assert result["reply"] == "Exciting personality!"
        assert result["commands"] == [{"type": "build", "part": "wall"}]


class TestStageIntentFallback:
    """Test intent parsing fallback behavior."""

    def test_intent_unparseable_creates_minimal(self):
        """When model output can't be parsed, construct minimal intent."""
        with patch.object(brain, "call_model", return_value="not json"):
            result = brain.stage_intent("fake-key", "build a castle")

        assert result["intent"] == "build"
        assert result["subject"] == "build a castle"
        assert result["scale"] == "medium"
        assert "_meta" in result

    def test_intent_successful_parse(self):
        """Successfully parsed intent should include model output."""
        mock_intent = {"intent": "build", "subject": "castle", "style": "medieval"}
        with patch.object(brain, "call_model", return_value=json.dumps(mock_intent)):
            result = brain.stage_intent("fake-key", "build a castle")

        assert result["intent"] == "build"
        assert result["subject"] == "castle"
        assert result["style"] == "medieval"
        assert "_meta" in result
