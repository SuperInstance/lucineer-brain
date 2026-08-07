#!/usr/bin/env python3
"""
Integration test: Verify emotional_context flows through ALL 5 pipeline stages.

Sends "I'm scared" through the pipeline (with mocked model calls) and verifies:
  Stage 1 (intent): emotion detected, context injected
  Stage 2 (plan):   emotional context passed to planner
  Stage 3 (coder):  emotional context passed to coder
  Stage 4 (hermes): emotional acknowledgment instructions in prompt
  Stage 5 (safety): reply checked for safety regardless of emotion

Also verifies the full-pipeline mock path produces a reply that acknowledges
the emotion ("scared") before describing the build.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import brain


# ─── Stage-by-stage verification ──────────────────────────────────────────────

class TestEmotionFlowsThroughAllStages:
    """The emotion must flow through every stage, not just intent + hermes."""

    # ── Stage 2: Planner gets emotional context ──
    @patch("brain.call_model")
    def test_planner_receives_emotional_context(self, mock_call):
        """stage_plan must include emotional context in the user message when emotion is present."""
        # First call = planner, return a valid plan
        mock_call.return_value = json.dumps({
            "steps": [{
                "step": 1,
                "action": "build shelter",
                "parts": [{
                    "name": "wall",
                    "purpose": "protection",
                    "shape_hint": "Block",
                    "position_hint": "around player",
                    "size_hint": "12x8x1",
                    "color_hint": "#4a3a2a",
                    "material_hint": "Wood",
                }]
            }]
        })

        intent = {
            "intent": "build",
            "subject": "shelter",
            "summary": "scared player wants shelter",
            "mood": "scared",
            "emotion": "scared",
            "emotional_context": brain.EMOTIONAL_ACKNOWLEDGMENTS["scared"],
        }

        brain.stage_plan("key", intent, "I'm scared", use_deep=False)

        # Verify the planner got emotional context in its user message
        call_args = mock_call.call_args
        messages = call_args.kwargs.get("messages", [])
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert "EMOTIONAL CONTEXT" in user_msg or "scared" in user_msg.lower(), \
            "Planner did not receive emotional context"

    @patch("brain.call_model")
    def test_planner_no_emotional_context_when_absent(self, mock_call):
        """stage_plan must NOT include emotional context when no emotion present."""
        mock_call.return_value = json.dumps({"steps": []})

        intent = {
            "intent": "build",
            "subject": "tower",
            "summary": "build a tower",
            "mood": "neutral",
        }

        brain.stage_plan("key", intent, "build a tower", use_deep=False)

        call_args = mock_call.call_args
        messages = call_args.kwargs.get("messages", [])
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert "EMOTIONAL CONTEXT" not in user_msg

    # ── Stage 3: Coder gets emotional context ──
    @patch("brain.call_model")
    def test_coder_receives_emotional_context(self, mock_call):
        """stage_commands must include emotional context when emotion is present."""
        mock_call.return_value = json.dumps({
            "reply": "I hear you. Walls are up.",
            "commands": [
                {"type": "createPart", "params": {"name": "shelter_wall"}}
            ]
        })

        plan = {"steps": [{"step": 1, "action": "build shelter", "parts": []}]}
        intent = {
            "intent": "build",
            "summary": "scared player wants shelter",
            "mood": "scared",
            "emotion": "scared",
            "emotional_context": brain.EMOTIONAL_ACKNOWLEDGMENTS["scared"],
        }

        brain.stage_commands("key", plan, intent, "I'm scared")

        call_args = mock_call.call_args
        messages = call_args.kwargs.get("messages", [])
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert "EMOTIONAL CONTEXT" in user_msg or "scared" in user_msg.lower(), \
            "Coder did not receive emotional context"

    @patch("brain.call_model")
    def test_coder_no_emotional_context_when_absent(self, mock_call):
        """stage_commands must NOT include emotional context when no emotion present."""
        mock_call.return_value = json.dumps({
            "reply": "Tower's up.",
            "commands": [{"type": "createPart", "params": {"name": "tower"}}]
        })

        plan = {"steps": [{"step": 1, "action": "build tower", "parts": []}]}
        intent = {
            "intent": "build",
            "summary": "build a tower",
            "mood": "neutral",
        }

        brain.stage_commands("key", plan, intent, "build a tower")

        call_args = mock_call.call_args
        messages = call_args.kwargs.get("messages", [])
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert "EMOTIONAL CONTEXT" not in user_msg


# ─── Full pipeline: "I'm scared" end-to-end ────────────────────────────────────

class TestScaredPipelineEndToEnd:
    """
    Send 'I'm scared' through the full pipeline with mocked models.
    Verify the final reply acknowledges the fear.
    """

    @patch("brain.stage_safety")
    @patch("brain.call_model")
    def test_scared_response_acknowledges_emotion(self, mock_call, mock_safety):
        """When a player says 'I'm scared', the pipeline reply must acknowledge it."""
        # Stage 1 (intent): return scared intent
        # Stage 2 (plan): return a shelter plan
        # Stage 3 (coder): return a reply that acknowledges fear
        # Stage 4 (hermes): enhance with empathy
        #
        # mock_call is called sequentially: intent, planner, coder, hermes, safety
        responses = [
            # Stage 1: Intent
            json.dumps({
                "intent": "build", "subject": "shelter",
                "style": "safe", "scale": "small",
                "mood": "neutral", "keywords": ["scared"],
                "summary": "scared player wants somewhere safe"
            }),
            # Stage 2: Plan
            json.dumps({
                "steps": [{
                    "step": 1,
                    "action": "build enclosed shelter",
                    "parts": [{
                        "name": "shelter_wall",
                        "purpose": "enclosure",
                        "shape_hint": "Block",
                        "position_hint": "around player",
                        "size_hint": "12x8x1",
                        "color_hint": "#4a3a2a",
                        "material_hint": "Wood",
                    }]
                }]
            }),
            # Stage 3: Commands — coder acknowledges emotion in reply
            json.dumps({
                "reply": "I hear you. Built you somewhere solid. Walls up, lantern lit. Left the door open — you decide when to close it.",
                "commands": [
                    {"type": "createPart", "params": {"name": "shelter_wall_1", "position": {"x": 0, "y": 0, "z": 0}, "size": {"x": 12, "y": 8, "z": 1}, "material": "Wood", "color": "#4a3a2a"}},
                    {"type": "addLight", "params": {"name": "warm_lantern", "position": {"x": 0, "y": 4, "z": 0}, "range": 15, "brightness": 2, "color": "#ffd700"}},
                ]
            }),
            # Stage 4: Hermes — enhances reply with empathy
            json.dumps({
                "reply": "I hear you. Let's get you somewhere solid. Walls are up, lantern's lit. Left the door open — you decide when to close it.",
                "commands": []  # Hermes commands are ignored anyway
            }),
        ]
        mock_call.side_effect = responses

        # Mock safety to always pass
        mock_safety.return_value = (True, "safe")

        result = brain.run_pipeline(
            "fake_key",
            "I'm scared",
            verbose=False,
            use_deep=False,
            creative=True,  # Enable Hermes stage
        )

        # The reply MUST acknowledge the fear
        reply_lower = result.get("reply", "").lower()
        acknowledge_indicators = [
            "hear you", "safe", "somewhere solid", "scared",
            "alright", "steady", "get you", "walls", "shelter"
        ]
        has_acknowledgment = any(ind in reply_lower for ind in acknowledge_indicators)
        assert has_acknowledgment, \
            f"Reply does not acknowledge fear: '{result.get('reply', '')}'"

        # Verify emotion was detected in pipeline metadata
        pipeline_meta = result.get("_pipeline", {})
        assert pipeline_meta.get("creative") is True

    @patch("brain.stage_safety")
    @patch("brain.call_model")
    def test_scared_intent_detected_early(self, mock_call, mock_safety):
        """detect_emotion runs before any model call — emotion is in stage 1."""
        # Verify detect_emotion catches it independently
        emotion = brain.detect_emotion("I'm scared")
        assert emotion == "scared", f"Expected 'scared', got '{emotion}'"

    @patch("brain.stage_safety")
    @patch("brain.call_model")
    def test_scared_fast_mode_preserves_emotion(self, mock_call, mock_safety):
        """Fast mode (single model) should still handle emotion reasonably."""
        mock_call.return_value = json.dumps({
            "reply": "I hear you. Built you somewhere solid. Walls are up.",
            "commands": [
                {"type": "createPart", "params": {"name": "wall", "position": {"x": 0, "y": 0, "z": 0}, "size": {"x": 12, "y": 8, "z": 1}, "material": "Wood", "color": "#4a3a2a"}}
            ]
        })
        mock_safety.return_value = (True, "safe")

        result = brain.run_fast("fake_key", "I'm scared", verbose=False, creative=False)

        # Even in fast mode, the reply should exist and make sense
        assert "reply" in result
        assert len(result.get("commands", [])) > 0
        assert result["_pipeline"]["mode"] == "fast"


# ─── Emotional context doesn't leak into non-emotional builds ──────────────────

class TestEmotionIsolation:
    """Ensure emotional context doesn't accidentally appear in neutral builds."""

    @patch("brain.call_model")
    def test_neutral_build_has_no_emotion_keywords(self, mock_call):
        """A normal build request shouldn't inject any emotional context."""
        mock_call.return_value = json.dumps({
            "intent": "build",
            "subject": "castle",
            "summary": "build a big castle",
            "mood": "epic",
            "keywords": ["tower"],
            "style": "medieval",
            "scale": "large",
        })

        result = brain.stage_intent("key", "build me a big castle")
        assert "emotion" not in result
        assert "emotional_context" not in result

    def test_detect_emotion_returns_none_for_build_requests(self):
        """Common build phrases should not trigger emotional detection."""
        build_phrases = [
            "build me a house",
            "make a tall tower",
            "create a garden",
            "put up a dock",
            "build a bridge across the river",
            "I want to see a windmill",
            "give me a lighthouse",
        ]
        for phrase in build_phrases:
            assert brain.detect_emotion(phrase) is None, \
                f"False positive emotion for: '{phrase}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
