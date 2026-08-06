#!/usr/bin/env python3
"""
Tests for emotional detection and emotional_context wiring in brain.py.

Covers:
- detect_emotion: keyword matching, priority order, word boundaries, no match
- EMOTIONAL_KEYWORDS: completeness, no empty lists
- EMOTIONAL_ACKNOWLEDGMENTS: every emotion has an acknowledgment
- stage_intent: emotional injection into intent dict
- stage_hermes: emotional context acknowledgment instructions
- stage_plan: emotional context passthrough (via intent)
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import brain


# ─── detect_emotion ──────────────────────────────────────────────────────────

class TestDetectEmotion:
    """The emotional detector is the first line of empathy. It must be reliable."""

    # ── Single-keyword hits ──
    def test_scared(self):
        assert brain.detect_emotion("I'm scared") == "scared"

    def test_afraid(self):
        assert brain.detect_emotion("I'm afraid of the dark") == "scared"

    def test_lonely(self):
        assert brain.detect_emotion("I feel lonely") == "lonely"

    def test_alone(self):
        assert brain.detect_emotion("I'm all alone here") == "lonely"

    def test_sad(self):
        assert brain.detect_emotion("I'm sad") == "sad"

    def test_unhappy(self):
        assert brain.detect_emotion("I feel unhappy") == "sad"

    def test_happy(self):
        assert brain.detect_emotion("I'm so happy right now") == "happy"

    def test_excited(self):
        assert brain.detect_emotion("I'm excited to build!") == "excited"

    def test_stoked(self):
        assert brain.detect_emotion("I'm so stoked about this") == "excited"

    def test_angry(self):
        assert brain.detect_emotion("I'm angry at this game") == "angry"

    def test_frustrated(self):
        assert brain.detect_emotion("This is so frustrating") == "angry"

    def test_worried(self):
        assert brain.detect_emotion("I'm worried about something") == "scared"

    def test_anxious(self):
        assert brain.detect_emotion("I feel anxious") == "scared"

    # ── No match ──
    def test_no_emotion_neutral_build(self):
        assert brain.detect_emotion("build me a castle") is None

    def test_no_emotion_greeting(self):
        assert brain.detect_emotion("hello there") is None

    def test_empty_string(self):
        assert brain.detect_emotion("") is None

    def test_whitespace_only(self):
        assert brain.detect_emotion("   ") is None

    # ── Case insensitivity ──
    def test_uppercase(self):
        assert brain.detect_emotion("I'M SCARED") == "scared"

    def test_mixed_case(self):
        assert brain.detect_emotion("I'm ScArEd") == "scared"

    def test_all_caps(self):
        assert brain.detect_emotion("LONELY") == "lonely"

    # ── Word boundary (no false positives) ──
    def test_scared_in_scarecrow(self):
        """'scared' should NOT match inside 'scarecrow'."""
        assert brain.detect_emotion("build a scarecrow") is None

    def test_mad_in_madagascar(self):
        """'mad' should NOT match inside 'Madagascar'."""
        assert brain.detect_emotion("build Madagascar") is None

    def test_sad_in_saddlebag(self):
        """'sad' should NOT match inside 'saddlebag'."""
        assert brain.detect_emotion("build a saddlebag") is None

    # ── Priority order: scared > lonely > sad > happy > excited > angry ──
    def test_priority_scared_over_happy(self):
        """If both scared and happy keywords appear, scared wins."""
        result = brain.detect_emotion("I'm scared but also happy")
        assert result == "scared"

    def test_priority_lonely_over_sad(self):
        """If both lonely and sad keywords appear, lonely wins."""
        result = brain.detect_emotion("I'm lonely and sad")
        assert result == "lonely"

    def test_priority_sad_over_happy(self):
        """If both sad and happy keywords appear, sad wins."""
        result = brain.detect_emotion("I'm sad yet somehow happy")
        assert result == "sad"

    def test_priority_happy_over_angry(self):
        """If both happy and angry keywords appear, happy wins."""
        result = brain.detect_emotion("I'm happy but also angry")
        assert result == "happy"


# ─── EMOTIONAL_KEYWORDS & ACKNOWLEDGMENTS ─────────────────────────────────────

class TestEmotionalConfig:
    def test_all_emotions_have_keywords(self):
        for emotion, keywords in brain.EMOTIONAL_KEYWORDS.items():
            assert isinstance(keywords, list), f"{emotion} keywords not a list"
            assert len(keywords) > 0, f"{emotion} has empty keyword list"

    def test_all_emotions_have_acknowledgments(self):
        for emotion in brain.EMOTIONAL_KEYWORDS:
            assert emotion in brain.EMOTIONAL_ACKNOWLEDGMENTS, \
                f"No acknowledgment for '{emotion}'"
            ack = brain.EMOTIONAL_ACKNOWLEDGMENTS[emotion]
            assert isinstance(ack, str) and len(ack) > 10, \
                f"Acknowledgment for '{emotion}' is too short"

    def test_expected_emotion_set(self):
        expected = {"scared", "lonely", "sad", "happy", "excited", "angry"}
        actual = set(brain.EMOTIONAL_KEYWORDS.keys())
        assert actual == expected, f"Unexpected emotions: {actual ^ expected}"

    def test_no_duplicate_keywords_across_emotions(self):
        """A keyword shouldn't appear in two emotion categories."""
        all_kws = []
        for keywords in brain.EMOTIONAL_KEYWORDS.values():
            all_kws.extend(keywords)
        duplicates = [kw for kw in all_kws if all_kws.count(kw) > 1]
        assert not duplicates, f"Duplicate keywords: {set(duplicates)}"


# ─── stage_intent emotional injection ─────────────────────────────────────────

class TestStageIntentEmotion:
    @patch("brain.call_model")
    def test_emotion_injected_on_match(self, mock_call):
        """When player says 'I'm scared', intent dict gets emotion + emotional_context."""
        mock_call.return_value = json.dumps({
            "intent": "build", "subject": "shelter",
            "style": "safe", "scale": "small",
            "mood": "neutral", "keywords": ["scared"],
            "summary": "player is scared and wants shelter"
        })
        result = brain.stage_intent("key", "I'm scared, build me a shelter")
        assert result["emotion"] == "scared"
        assert "emotional_context" in result
        assert result["emotional_context"] == brain.EMOTIONAL_ACKNOWLEDGMENTS["scared"]

    @patch("brain.call_model")
    def test_mood_shifted_from_neutral_to_emotion(self, mock_call):
        """When mood is 'neutral' and emotion detected, mood shifts to the emotion."""
        mock_call.return_value = json.dumps({
            "intent": "chat", "subject": "feelings",
            "style": "default", "scale": "small",
            "mood": "neutral", "keywords": ["lonely"],
            "summary": "lonely player"
        })
        result = brain.stage_intent("key", "I'm so lonely")
        assert result["mood"] == "lonely"
        assert result["emotion"] == "lonely"

    @patch("brain.call_model")
    def test_mood_preserved_when_not_neutral(self, mock_call):
        """If the model already set a mood, don't override it with the emotion."""
        mock_call.return_value = json.dumps({
            "intent": "build", "subject": "house",
            "style": "cozy", "scale": "small",
            "mood": "warm", "keywords": ["happy"],
            "summary": "happy player wants a cozy house"
        })
        result = brain.stage_intent("key", "I'm so happy, build me a house")
        assert result["mood"] == "warm"
        assert result["emotion"] == "happy"

    @patch("brain.call_model")
    def test_no_emotion_when_none_detected(self, mock_call):
        """Normal build requests don't inject emotion."""
        mock_call.return_value = json.dumps({
            "intent": "build", "subject": "castle",
            "style": "medieval", "scale": "large",
            "mood": "epic", "keywords": ["tower"],
            "summary": "build a big castle"
        })
        result = brain.stage_intent("key", "build me a big castle")
        assert "emotion" not in result
        assert "emotional_context" not in result

    @patch("brain.call_model")
    def test_emotion_injected_even_on_fallback(self, mock_call):
        """Even if the model returns garbage, emotion is still detected."""
        mock_call.return_value = "total garbage not json at all"
        result = brain.stage_intent("key", "I'm scared")
        assert result["emotion"] == "scared"
        assert "emotional_context" in result
        # Fallback intent should still work
        assert result["intent"] == "build"


# ─── stage_hermes emotional acknowledgment ────────────────────────────────────

class TestStageHermesEmotion:
    @patch("brain.call_model")
    def test_emotion_instructions_appear_in_hermes_call(self, mock_call):
        """When intent has emotion, Hermes gets acknowledgment instructions."""
        mock_call.return_value = json.dumps({
            "reply": "I hear you. Built you something solid.",
            "commands": [{"type": "createPart", "params": {"name": "wall"}}]
        })
        intent = {
            "summary": "scared player wants shelter",
            "mood": "scared", "style": "safe", "scale": "small",
            "emotion": "scared",
            "emotional_context": brain.EMOTIONAL_ACKNOWLEDGMENTS["scared"],
        }
        result_dict = {
            "reply": "Built a shelter.",
            "commands": [{"type": "createPart", "params": {"name": "wall"}}]
        }

        brain.stage_hermes("key", result_dict, intent, "I'm scared")

        # Check the user message sent to Hermes contains emotion instructions
        call_args = mock_call.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert "EMOTIONAL CONTEXT" in user_msg or "emotion" in user_msg.lower()
        assert "scared" in user_msg

    @patch("brain.call_model")
    def test_no_emotion_instructions_when_emotion_absent(self, mock_call):
        """When intent has no emotion, Hermes call has no emotion instructions."""
        mock_call.return_value = json.dumps({
            "reply": "Threw up a tower.",
            "commands": [{"type": "createPart", "params": {"name": "tower"}}]
        })
        intent = {
            "summary": "build a tower",
            "mood": "neutral", "style": "default", "scale": "medium",
        }
        result_dict = {
            "reply": "Built a tower.",
            "commands": [{"type": "createPart", "params": {"name": "tower"}}]
        }

        brain.stage_hermes("key", result_dict, intent, "build a tower")

        call_args = mock_call.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assert "EMOTIONAL CONTEXT" not in user_msg

    @patch("brain.call_model")
    def test_hermes_preserves_commands_with_emotion(self, mock_call):
        """Commands are never modified by Hermes, even with emotional context."""
        original_commands = [
            {"type": "createPart", "params": {"name": "shelter_wall_1"}},
            {"type": "createPart", "params": {"name": "shelter_wall_2"}},
            {"type": "addLight", "params": {"name": "warm_lantern"}},
        ]
        mock_call.return_value = json.dumps({
            "reply": "I hear you. Let's get you somewhere safe. Walls are up.",
            "commands": [{"type": "createPart", "params": {"name": "FAKE"}}],
        })
        intent = {
            "summary": "scared player",
            "mood": "scared", "style": "safe", "scale": "small",
            "emotion": "scared",
            "emotional_context": brain.EMOTIONAL_ACKNOWLEDGMENTS["scared"],
        }
        result_dict = {
            "reply": "Built a shelter.",
            "commands": original_commands,
        }

        final = brain.stage_hermes("key", result_dict, intent, "I'm scared")
        # Commands must be unchanged — Hermes only touches the reply
        assert final["commands"] == original_commands
        # Reply should be the Hermes-enhanced version
        assert "safe" in final["reply"].lower() or "hear" in final["reply"].lower()


# ─── Integration: full emotion pipeline (mocked) ──────────────────────────────

class TestEmotionPipelineIntegration:
    @patch("brain.call_model")
    def test_emotion_flows_from_detect_through_intent(self, mock_call):
        """End-to-end: detect_emotion → stage_intent → intent dict has emotion."""
        mock_call.return_value = json.dumps({
            "intent": "chat", "subject": "feelings",
            "style": "emotional", "scale": "small",
            "mood": "neutral", "keywords": ["sad"],
            "summary": "sad player"
        })

        # Detect emotion first (as stage_intent does internally)
        emotion = brain.detect_emotion("I'm so sad today")
        assert emotion == "sad"

        # Then run stage_intent — it should find the same emotion
        result = brain.stage_intent("key", "I'm so sad today")
        assert result["emotion"] == "sad"
        assert "something gentle" in result["emotional_context"].lower() or \
               "sad" in result["emotional_context"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
