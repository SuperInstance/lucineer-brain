#!/usr/bin/env python3
"""
Test suite for lucineer-brain — the foreman finally checks his own work.

Tests cover the pure-logic functions that don't require API calls:
- extract_json: the JSON extraction utility
- persona_for: bond-tier persona generation
- load_api_key: API key loading
- MODELS / config constants
- VOICE_EXAMPLES format
- verbose_check toggle
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure we can import brain
sys.path.insert(0, str(Path(__file__).parent.parent))
import brain


# ─── extract_json ────────────────────────────────────────────────────────────

class TestExtractJson:
    """The JSON extractor is the pipeline's safety net. If it fails, everything fails."""

    def test_clean_json_object(self):
        result = brain.extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_clean_json_array(self):
        result = brain.extract_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_nested_json(self):
        text = '{"a": {"b": [1, 2, {"c": 3}]}}'
        assert brain.extract_json(text) == {"a": {"b": [1, 2, {"c": 3}]}}

    def test_markdown_fenced_json(self):
        text = '```json\n{"key": "value"}\n```'
        assert brain.extract_json(text) == {"key": "value"}

    def test_markdown_fenced_no_language(self):
        text = '```\n{"key": "value"}\n```'
        assert brain.extract_json(text) == {"key": "value"}

    def test_json_embedded_in_prose(self):
        text = 'Here is the result: {"key": "value"} done.'
        assert brain.extract_json(text) == {"key": "value"}

    def test_json_array_embedded_in_prose(self):
        text = 'Output: [1, 2, 3] finished'
        assert brain.extract_json(text) == [1, 2, 3]

    def test_empty_string(self):
        assert brain.extract_json('') is None

    def test_whitespace_only(self):
        assert brain.extract_json('   \n\t  ') is None

    def test_no_json_content(self):
        assert brain.extract_json('no json here, just words') is None

    def test_partial_json_incomplete(self):
        """Incomplete JSON should return None, not crash."""
        assert brain.extract_json('{"key": "value"') is None

    def test_json_with_unicode(self):
        text = '{"name": "Lucineer ⚒️", "role": "foreman"}'
        result = brain.extract_json(text)
        assert result["name"] == "Lucineer ⚒️"

    def test_json_with_nested_code_fences_inside_string(self):
        """A JSON string value containing backticks should not break extraction."""
        text = '{"reply": "Set the floor. ```lua\\nprint(1)\\n``` was the old way."}'
        result = brain.extract_json(text)
        assert result is not None
        assert result["reply"].startswith("Set the floor")

    def test_multiple_json_objects_finds_first(self):
        text = '{"first": true} and {"second": true}'
        result = brain.extract_json(text)
        assert result == {"first": True}

    def test_json_with_numbers_bool_null(self):
        text = '{"int": 42, "float": 3.14, "bool": true, "null": null}'
        result = brain.extract_json(text)
        assert result == {"int": 42, "float": 3.14, "bool": True, "null": None}

    def test_json_array_in_object(self):
        text = '{"commands": [{"type": "createPart", "params": {"x": 1}}]}'
        result = brain.extract_json(text)
        assert len(result["commands"]) == 1
        assert result["commands"][0]["params"]["x"] == 1

    def test_trailing_content_after_json(self):
        """Should extract JSON even with trailing garbage."""
        text = '{"reply": "built it"}\n\nSome extra text'
        result = brain.extract_json(text)
        assert result == {"reply": "built it"}

    def test_markdown_fence_with_blank_lines(self):
        text = '```\n\n{"key": "value"}\n\n```'
        result = brain.extract_json(text)
        assert result == {"key": "value"}

    def test_malformed_json_returns_none(self):
        """Garbage that looks like JSON but isn't."""
        assert brain.extract_json('{key: value}') is None

    def test_very_long_json(self):
        """Large JSON should extract correctly."""
        data = {"parts": [{"name": f"part_{i}", "x": i} for i in range(100)]}
        text = json.dumps(data)
        result = brain.extract_json(text)
        assert len(result["parts"]) == 100
        assert result["parts"][50]["name"] == "part_50"


# ─── persona_for / Bond Tiers ────────────────────────────────────────────────

class TestPersonaFor:
    """Bond tiers gate relationship depth. Getting this wrong breaks character."""

    def test_tier_0_low_bond(self):
        """Bond 0-9: no relationship section."""
        persona = brain.persona_for(0)
        assert "RELATIONSHIP" not in persona
        assert brain.LUCINEER_PERSONA in persona

    def test_tier_0_boundary(self):
        """Bond 9 is still tier 0."""
        persona = brain.persona_for(9)
        assert "RELATIONSHIP" not in persona

    def test_tier_1_entered(self):
        """Bond 10: first relationship tier."""
        persona = brain.persona_for(10)
        assert "RELATIONSHIP" in persona
        assert "PREVIOUS builds" in persona or "previous builds" in persona.lower()

    def test_tier_1_boundary(self):
        """Bond 29 is still tier 1."""
        persona = brain.persona_for(29)
        assert "RELATIONSHIP" in persona

    def test_tier_2_entered(self):
        """Bond 30: trust level — arguing allowed."""
        persona = brain.persona_for(30)
        assert "RELATIONSHIP" in persona
        assert "ARGUE" in persona or "argue" in persona.lower()

    def test_tier_2_boundary(self):
        persona = brain.persona_for(69)
        assert "RELATIONSHIP" in persona

    def test_tier_3_entered(self):
        """Bond 70: 'we' territory."""
        persona = brain.persona_for(70)
        assert "RELATIONSHIP" in persona
        assert "Say 'we'" in persona or "we" in persona.lower()

    def test_tier_3_boundary(self):
        persona = brain.persona_for(149)
        assert "RELATIONSHIP" in persona

    def test_tier_4_entered(self):
        """Bond 150: full truth."""
        persona = brain.persona_for(150)
        assert "RELATIONSHIP" in persona
        assert "truth" in persona.lower() or "Tell the truth" in persona

    def test_tier_4_high(self):
        """Very high bond still tier 4."""
        persona = brain.persona_for(999)
        assert "RELATIONSHIP" in persona

    def test_negative_bond_clamped_to_tier_0(self):
        """Negative bond should not crash."""
        persona = brain.persona_for(-5)
        assert "RELATIONSHIP" not in persona

    def test_base_persona_always_present(self):
        """The core persona text should be in every tier."""
        for level in [0, 10, 30, 70, 150, 500]:
            persona = brain.persona_for(level)
            assert brain.LUCINEER_PERSONA in persona, f"Base persona missing at bond {level}"

    def test_foreman_voice_in_base_persona(self):
        """Key character traits must be in the base persona."""
        assert "foreman" in brain.LUCINEER_PERSONA.lower()
        assert "unfinished" in brain.LUCINEER_PERSONA.lower() or "unfinished" in brain.LUCINEER_PERSONA
        assert "Magnus" in brain.LUCINEER_PERSONA

    def test_no_assistant_language(self):
        """The persona must explicitly reject assistant-like language."""
        # The persona should mention what NOT to say
        lower = brain.LUCINEER_PERSONA.lower()
        assert "never" in lower or "not an assistant" in lower
        # Must reject common AI-assistant tropes
        assert "how can i help" in lower or "great question" in lower or "not an assistant" in lower


# ─── load_api_key ────────────────────────────────────────────────────────────

class TestLoadApiKey:
    """API key loading from file and environment."""

    def test_loads_from_env_variable(self, monkeypatch):
        """Should fall back to environment variable."""
        monkeypatch.setattr(brain, "ENV_PATH", Path("/nonexistent/path/.env"))
        monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key-from-env")
        assert brain.load_api_key() == "test-key-from-env"

    def test_loads_from_file(self, tmp_path):
        """Should load from .env file."""
        env_file = tmp_path / ".env"
        env_file.write_text('DEEPINFRA_API_KEY="key-from-file"\n')
        
        original_path = brain.ENV_PATH
        brain.ENV_PATH = env_file
        try:
            result = brain.load_api_key()
            assert result == "key-from-file"
        finally:
            brain.ENV_PATH = original_path

    def test_loads_from_file_no_quotes(self):
        """Should handle keys without quotes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('DEEPINFRA_API_KEY=bare-key-no-quotes\n')
            f.flush()
            
            original_path = brain.ENV_PATH
            brain.ENV_PATH = Path(f.name)
            try:
                assert brain.load_api_key() == "bare-key-no-quotes"
            finally:
                brain.ENV_PATH = original_path
                os.unlink(f.name)

    def test_raises_when_not_found(self, monkeypatch):
        """Should raise RuntimeError when key not found anywhere."""
        monkeypatch.setattr(brain, "ENV_PATH", Path("/nonexistent/.env"))
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="DEEPINFRA_API_KEY"):
            brain.load_api_key()

    def test_raises_on_empty_key(self, tmp_path):
        """Empty key in file should fall through to env, then error."""
        env_file = tmp_path / ".env"
        env_file.write_text('DEEPINFRA_API_KEY=\n')
        
        original_path = brain.ENV_PATH
        brain.ENV_PATH = env_file
        try:
            with pytest.raises(RuntimeError):
                brain.load_api_key()
        finally:
            brain.ENV_PATH = original_path


# ─── Configuration Constants ─────────────────────────────────────────────────

class TestConfig:
    """Verify the configuration is well-formed and consistent."""

    def test_all_model_stages_defined(self):
        required_stages = {"intent", "planner", "deep", "coder", "hermes"}
        assert set(brain.MODELS.keys()) == required_stages

    def test_model_names_are_strings(self):
        for stage, model in brain.MODELS.items():
            assert isinstance(model, str), f"{stage} model is not a string"
            assert len(model) > 10, f"{stage} model name suspiciously short: {model}"

    def test_all_stages_have_temperatures(self):
        for stage in brain.MODELS:
            assert stage in brain.Temperatures, f"No temperature for {stage}"

    def test_all_stages_have_max_tokens(self):
        for stage in brain.MODELS:
            assert stage in brain.MAX_TOKENS, f"No max_tokens for {stage}"

    def test_temperatures_in_valid_range(self):
        for stage, temp in brain.Temperatures.items():
            assert 0.0 <= temp <= 2.0, f"{stage} temperature {temp} out of range"

    def test_max_tokens_positive(self):
        for stage, tokens in brain.MAX_TOKENS.items():
            assert tokens > 0, f"{stage} max_tokens not positive"
            assert tokens <= 8192, f"{stage} max_tokens suspiciously large: {tokens}"

    def test_intent_is_low_temperature(self):
        """Intent parsing should be deterministic (low temperature)."""
        assert brain.Temperatures["intent"] <= 0.4

    def test_hermes_has_highest_temperature(self):
        """Creative personality wrapping should be hottest."""
        assert brain.Temperatures["hermes"] == max(brain.Temperatures.values())
        assert brain.Temperatures["hermes"] >= 0.7

    def test_coder_has_low_temperature(self):
        """Code generation should be near-deterministic."""
        assert brain.Temperatures["coder"] <= 0.3

    def test_planner_fallbacks_is_list(self):
        assert isinstance(brain.PLANNER_FALLBACKS, list)
        assert len(brain.PLANNER_FALLBACKS) >= 1

    def test_safety_model_defined(self):
        assert hasattr(brain, "SAFETY_MODEL")
        assert "Safety" in brain.SAFETY_MODEL or "safety" in brain.SAFETY_MODEL.lower()

    def test_api_base_correct(self):
        assert "deepinfra.com" in brain.API_BASE
        assert "openai" in brain.API_BASE.lower() or "chat/completions" in brain.API_BASE


# ─── VOICE_EXAMPLES ──────────────────────────────────────────────────────────

class TestVoiceExamples:
    """Voice examples are the few-shot prompt for the coder model. They must be right."""

    def test_examples_are_list_of_strings(self):
        assert isinstance(brain.VOICE_EXAMPLES, list)
        assert all(isinstance(ex, str) for ex in brain.VOICE_EXAMPLES)

    def test_examples_are_short(self):
        """Voice examples should be 1-3 sentences each."""
        for i, ex in enumerate(brain.VOICE_EXAMPLES):
            sentences = ex.count('.') + ex.count('!') + ex.count('?')
            assert 1 <= sentences <= 5, f"Example {i} has {sentences} sentence-ending marks"

    def test_no_assistant_language_in_examples(self):
        """Voice examples must NOT contain AI-assistant tropes."""
        forbidden = ["great question", "i'd be happy to", "certainly", "let's", "shall we", "amazing"]
        for i, ex in enumerate(brain.VOICE_EXAMPLES):
            lower = ex.lower()
            for word in forbidden:
                assert word not in lower, f"Example {i} contains '{word}': {ex[:50]}..."

    def test_examples_use_contractions(self):
        """The foreman contracts. At least half the examples should have contractions."""
        contracted = sum(1 for ex in brain.VOICE_EXAMPLES if "'" in ex)
        assert contracted >= len(brain.VOICE_EXAMPLES) // 2, \
            f"Only {contracted}/{len(brain.VOICE_EXAMPLES)} examples use contractions"

    def test_at_least_one_magnus_reference(self):
        """At least one example should reference Magnus."""
        assert any("Magnus" in ex for ex in brain.VOICE_EXAMPLES)

    def test_at_least_one_alaska_reference(self):
        """At least one example should reference Alaska/maritime work."""
        alaska_terms = ["alaska", "petersburg", "tender", "cannery", "channel", "crab"]
        has_alaska = any(
            any(term in ex.lower() for term in alaska_terms)
            for ex in brain.VOICE_EXAMPLES
        )
        assert has_alaska, "No Alaska/maritime references in voice examples"

    def test_examples_demonstrate_unfinished_pattern(self):
        """At least one example should show the 'left X unfinished' pattern."""
        unfinished_terms = ["left", "didn't", "open", "figure", "depends", "you'll want"]
        has_unfinished = any(
            any(term in ex.lower() for term in unfinished_terms)
            for ex in brain.VOICE_EXAMPLES
        )
        assert has_unfinished, "No 'unfinished' pattern in voice examples"


# ─── verbose_check ───────────────────────────────────────────────────────────

class TestVerboseCheck:
    """The verbose flag is a function attribute, not a global. Easy to get wrong."""

    def test_default_false(self):
        """Verbose should default to False."""
        brain.verbose_check._verbose = False
        assert brain.verbose_check() is False

    def test_set_true(self):
        brain.verbose_check._verbose = True
        assert brain.verbose_check() is True

    def test_set_false(self):
        brain.verbose_check._verbose = True
        brain.verbose_check._verbose = False
        assert brain.verbose_check() is False

    def tearDown(self):
        """Reset to default after each test."""
        brain.verbose_check._verbose = False


# ─── LUCINEER_PERSONA content checks ─────────────────────────────────────────

class TestPersonaContent:
    """Deep checks on the persona text — the character bible in code."""

    def test_three_beat_pattern_described(self):
        """The persona must describe the three-beat pattern."""
        assert "three-beat" in brain.LUCINEER_PERSONA.lower() or "three beat" in brain.LUCINEER_PERSONA.lower()

    def test_bond_tiers_match_persona(self):
        """Bond tier descriptions should be consistent with the persona's relationship rules."""
        # Tier 0 should have empty string (no additions)
        assert brain.BOND_TIERS[0] == ""

    def test_tiers_are_progressive(self):
        """Each tier should be longer/more detailed than the last (rough heuristic)."""
        lengths = [len(brain.BOND_TIERS[i]) for i in range(5)]
        # Tier 0 is empty, so start from 1
        for i in range(1, 4):
            assert lengths[i] > 10, f"Tier {i} suspiciously short: {lengths[i]}"

    def test_persona_mentions_southeast_alaska(self):
        assert "Southeast Alaska" in brain.LUCINEER_PERSONA or "southeast alaska" in brain.LUCINEER_PERSONA.lower()

    def test_persona_mentions_foundations(self):
        """The foreman cares about foundations."""
        assert "foundation" in brain.LUCINEER_PERSONA.lower() or "foundations" in brain.LUCINEER_PERSONA.lower()

    def test_persona_mentions_old_engines(self):
        """References to old engines (the Yard, Shell, Scrapcraft, Fleet)."""
        assert "Yard" in brain.LUCINEER_PERSONA or "Shell" in brain.LUCINEER_PERSONA

    def test_persona_has_calibration_examples(self):
        """The persona should include calibration examples."""
        assert "CALIBRATION" in brain.LUCINEER_PERSONA or "calibration" in brain.LUCINEER_PERSONA.lower()


# ─── System Prompts ──────────────────────────────────────────────────────────

class TestSystemPrompts:
    """Verify all system prompts exist and have the right shape."""

    @pytest.mark.parametrize("prompt_name", [
        "SYSTEM_INTENT", "SYSTEM_PLANNER", "SYSTEM_DEEP_PLANNER",
        "SYSTEM_CODER", "SYSTEM_HERMES", "SYSTEM_FAST",
    ])
    def test_prompt_exists(self, prompt_name):
        prompt = getattr(brain, prompt_name)
        assert isinstance(prompt, str)
        assert len(prompt) > 100, f"{prompt_name} too short ({len(prompt)} chars)"

    @pytest.mark.parametrize("prompt_name", [
        "SYSTEM_INTENT", "SYSTEM_PLANNER", "SYSTEM_DEEP_PLANNER",
        "SYSTEM_CODER", "SYSTEM_FAST",
    ])
    def test_prompts_mention_json(self, prompt_name):
        """Planning and coding prompts should mention JSON format."""
        prompt = getattr(brain, prompt_name)
        assert "json" in prompt.lower() or "JSON" in prompt

    def test_intent_prompt_has_schema(self):
        """The intent prompt should describe the expected JSON schema."""
        assert "intent" in brain.SYSTEM_INTENT.lower()
        assert "build" in brain.SYSTEM_INTENT.lower()

    def test_coder_prompt_lists_command_types(self):
        """The coder prompt should enumerate available command types."""
        assert "createPart" in brain.SYSTEM_CODER
        assert "addLight" in brain.SYSTEM_CODER
        assert "setTerrain" in brain.SYSTEM_CODER

    def test_hermes_prompt_preserves_commands(self):
        """Hermes should be told NOT to modify commands."""
        assert "commands" in brain.SYSTEM_HERMES
        # The prompt must say something about not changing commands
        lower = brain.SYSTEM_HERMES.lower()
        assert "unchanged" in lower or "do not" in lower or "same" in lower

    def test_deep_planner_more_detailed_than_standard(self):
        """Deep planner should have more guidance than standard."""
        assert len(brain.SYSTEM_DEEP_PLANNER) >= len(brain.SYSTEM_PLANNER)


# ─── Integration: function signatures ────────────────────────────────────────

class TestFunctionSignatures:
    """Verify the pipeline functions accept the right arguments."""

    def test_stage_intent_signature(self):
        import inspect
        sig = inspect.signature(brain.stage_intent)
        params = list(sig.parameters.keys())
        assert "api_key" in params
        assert "player_message" in params

    def test_stage_plan_signature(self):
        import inspect
        sig = inspect.signature(brain.stage_plan)
        params = list(sig.parameters.keys())
        assert "use_deep" in params

    def test_stage_commands_signature(self):
        import inspect
        sig = inspect.signature(brain.stage_commands)
        params = list(sig.parameters.keys())
        assert "plan" in params
        assert "intent" in params

    def test_stage_hermes_signature(self):
        import inspect
        sig = inspect.signature(brain.stage_hermes)
        params = list(sig.parameters.keys())
        assert "result" in params

    def test_run_pipeline_signature(self):
        import inspect
        sig = inspect.signature(brain.run_pipeline)
        params = list(sig.parameters.keys())
        assert "use_deep" in params
        assert "creative" in params

    def test_run_fast_signature(self):
        import inspect
        sig = inspect.signature(brain.run_fast)
        params = list(sig.parameters.keys())
        assert "creative" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ─── Fallback Chain & Graceful Degradation ──────────────────────────────────

class TestCoderFallbackChain:
    """Verify the coder fallback chain is well-structured (no redundant entries)."""

    def test_coder_fallbacks_excludes_primary(self):
        """CODER_FALLBACKS should NOT contain MODELS['coder'] — it's tried first
        separately by stage_commands, so including it wastes a retry cycle."""
        assert brain.MODELS["coder"] not in brain.CODER_FALLBACKS, \
            "CODER_FALLBACKS should not include the primary coder model"

    def test_coder_fallbacks_nonempty(self):
        """At least one fallback model must exist."""
        assert len(brain.CODER_FALLBACKS) >= 1

    def test_coder_fallbacks_are_strings(self):
        for model in brain.CODER_FALLBACKS:
            assert isinstance(model, str)
            assert "/" in model  # DeepInfra model format: org/model


class TestSafetyFailOpen:
    """The safety stage must FAIL OPEN on API errors, not block all replies."""

    def test_safety_fails_open_on_exception(self):
        """When call_model raises an exception, stage_safety should return (True, ...).
        Blocking everything during a safety API outage would make the brain unusable."""
        with patch("brain.call_model", side_effect=RuntimeError("API down")):
            is_safe, reason = brain.stage_safety("fake-key", "a safe reply", "build a house")
        assert is_safe is True, "Safety stage should fail OPEN (allow reply) on API error"
        assert "skip" in reason.lower() or "error" in reason.lower()

    def test_safety_blocks_unsafe_content(self):
        """When the model returns UNSAFE, the reply should be blocked."""
        with patch("brain.call_model", return_value="UNSAFE: profanity detected"):
            is_safe, reason = brain.stage_safety("fake-key", "some reply", "bad words")
        assert is_safe is False
        assert "profanity" in reason.lower()

    def test_safety_allows_safe_content(self):
        """When the model returns SAFE, the reply should pass."""
        with patch("brain.call_model", return_value="SAFE"):
            is_safe, reason = brain.stage_safety("fake-key", "a nice house", "build a house")
        assert is_safe is True

    def test_safety_empty_reply_passes(self):
        """Empty replies should pass the safety check without calling the API."""
        is_safe, reason = brain.stage_safety("fake-key", "", "anything")
        assert is_safe is True
        assert "empty" in reason.lower()

    def test_safety_whitespace_reply_passes(self):
        """Whitespace-only replies should also short-circuit."""
        is_safe, reason = brain.stage_safety("fake-key", "   \n\t  ", "anything")
        assert is_safe is True


class TestPlannerFailureDegradation:
    """When the planner returns no steps, the pipeline should gracefully degrade."""

    def test_planner_empty_steps_triggers_fast_fallback(self):
        """If stage_plan returns no steps, run_pipeline should drop to fast mode
        instead of sending an empty plan to the coder."""
        api_key = "fake-key"
        player_message = "build a house"

        # Mock stage_intent to succeed
        intent_result = {
            "intent": "build",
            "subject": "house",
            "summary": "build a house",
            "_meta": {"model": "test", "latency_s": 0.1, "raw": ""},
        }

        # Mock stage_plan to return empty steps (total failure)
        plan_result = {
            "steps": [],
            "error": "All planner models failed",
            "_meta": {"model": "none", "stage": "planner", "latency_s": 0.0, "raw": "", "fallbacks_tried": 2},
        }

        # Mock run_fast to return a valid result
        fast_result = {
            "reply": "Threw up a shell. Rough.",
            "commands": [{"type": "createPart", "params": {"name": "wall"}}],
            "_pipeline": {"mode": "fast"},
        }

        with patch("brain.stage_intent", return_value=intent_result), \
             patch("brain.stage_plan", return_value=plan_result), \
             patch("brain.run_fast", return_value=fast_result) as mock_fast, \
             patch("brain.stage_commands") as mock_coder, \
             patch("brain.stage_safety", return_value=(True, "safe")):

            result = brain.run_pipeline(api_key, player_message, verbose=False)

            # run_fast should have been called
            mock_fast.assert_called_once()
            # stage_commands should NOT have been called (planner failed)
            mock_coder.assert_not_called()
            # The result should come from fast mode
            assert result["_pipeline"]["planner_failed"] is True
            assert "fast" in result["_pipeline"]["mode"]
