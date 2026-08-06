"""Tests for health_check, response cache, and related utilities."""

import time
import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import brain


# ── Health check tests ────────────────────────────────────────────

class TestHealthCheck:
    def test_returns_dict(self):
        """health_check should always return a dict with expected keys."""
        with patch("brain.load_api_key", side_effect=RuntimeError("no key")):
            result = brain.health_check()
        assert isinstance(result, dict)
        assert "api_key_loaded" in result
        assert "models_configured" in result
        assert "env_file_exists" in result
        assert "can_reach_api" in result
        assert "timestamp" in result

    def test_no_api_key(self):
        """Without an API key, should report api_key_loaded=False."""
        with patch("brain.load_api_key", side_effect=RuntimeError("no key")):
            result = brain.health_check()
        assert result["api_key_loaded"] is False
        assert result["can_reach_api"] is False

    @patch("brain.call_model")
    @patch("brain.load_api_key", return_value="fake-key")
    def test_api_reachable(self, mock_key, mock_call):
        """When call_model succeeds, can_reach_api should be True."""
        mock_call.return_value = "ok"
        result = brain.health_check()
        assert result["api_key_loaded"] is True
        assert result["can_reach_api"] is True

    @patch("brain.call_model", side_effect=RuntimeError("connection failed"))
    @patch("brain.load_api_key", return_value="fake-key")
    def test_api_unreachable(self, mock_key, mock_call):
        """When call_model fails, can_reach_api should be False."""
        result = brain.health_check()
        assert result["api_key_loaded"] is True
        assert result["can_reach_api"] is False

    def test_models_listed(self):
        """health_check should list configured models."""
        with patch("brain.load_api_key", side_effect=RuntimeError("no key")):
            result = brain.health_check()
        assert len(result["models_configured"]) >= 5  # intent, planner, deep, coder, hermes


# ── Cache tests ───────────────────────────────────────────────────

class TestResponseCache:
    def setup_method(self):
        brain.cache_clear()

    def teardown_method(self):
        brain.cache_clear()

    def test_cache_empty_initially(self):
        stats = brain.cache_stats()
        assert stats["entries"] == 0

    def test_cache_set_and_get(self):
        key = "test-key"
        response = {"reply": "hello", "commands": []}
        brain._cache_set(key, response)
        result = brain._cache_get(key)
        assert result == response

    def test_cache_miss(self):
        result = brain._cache_get("nonexistent")
        assert result is None

    def test_cache_key_generation(self):
        """Same message + mode should produce same key."""
        key1 = brain._cache_key("build a castle", "full")
        key2 = brain._cache_key("build a castle", "full")
        assert key1 == key2

    def test_cache_key_different_messages(self):
        key1 = brain._cache_key("build a castle", "full")
        key2 = brain._cache_key("build a forest", "full")
        assert key1 != key2

    def test_cache_key_different_modes(self):
        key1 = brain._cache_key("build a castle", "full")
        key2 = brain._cache_key("build a castle", "fast")
        assert key1 != key2

    def test_cache_clear(self):
        brain._cache_set("key1", {"data": 1})
        brain._cache_set("key2", {"data": 2})
        assert brain.cache_stats()["entries"] == 2
        removed = brain.cache_clear()
        assert removed == 2
        assert brain.cache_stats()["entries"] == 0

    def test_cache_eviction(self):
        """When cache is full, oldest entry should be evicted."""
        brain._CACHE_MAX = 3
        brain._cache_set("k1", {"d": 1})
        time.sleep(0.01)
        brain._cache_set("k2", {"d": 2})
        time.sleep(0.01)
        brain._cache_set("k3", {"d": 3})
        time.sleep(0.01)
        brain._cache_set("k4", {"d": 4})  # should evict k1
        assert brain._cache_get("k1") is None
        assert brain._cache_get("k4") is not None
        assert brain.cache_stats()["entries"] == 3

    def test_cache_ttl_expiry(self):
        """Cached entries should expire after TTL."""
        brain._CACHE_TTL = 0.1  # 100ms
        brain._cache_set("short", {"data": 1})
        time.sleep(0.15)
        result = brain._cache_get("short")
        assert result is None

    def test_cache_stats_structure(self):
        stats = brain.cache_stats()
        assert "entries" in stats
        assert "max_entries" in stats
        assert "ttl_seconds" in stats
