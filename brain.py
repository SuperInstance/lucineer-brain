#!/usr/bin/env python3
"""
Lucineer Brain — Multi-Model Intelligence for Roblox Building

Routes player natural language through a pipeline of DeepInfra models:
  1. ByteDance/Seed-2.0-mini              → intent parsing (fast, cheap)
  2. Qwen/Qwen3.6-35B-A3B                  → spatial planning (decompose into steps)
     OR ByteDance/Seed-2.0-pro             → deep planning (complex builds)
  3. Qwen/Qwen3-Coder-480B-A35B            → Luau command generation
     ↓ on 429: Qwen3-Coder-30B-A3B          → smaller coder, less rate-limited
     ↓ on 429: DeepSeek-V3                   → different provider, cheap
     ↓ all fail: fast mode (Seed-2.0-mini)   → ultimate last resort
  4. NousResearch/Hermes-3-Llama-3.1-405B  → personality/lore wrapping (creative mode)

Outputs JSON matching Lucineer's CommandExecutor schema:
  {"reply": "...", "commands": [{"type": "createPart", "params": {...}}, ...]}

Usage:
  python3 brain.py "build me a castle on the hill"
  python3 brain.py --fast "build a small house"        # single-model fallback
  python3 brain.py --creative "build a dragon temple"  # lore-rich personality mode
  python3 brain.py --deep "build a floating city"      # use Seed-2.0-pro for planning
  python3 brain.py --test                              # test all models

Requires:
  DEEPINFRA_API_KEY in /home/eileen/mcp-deeinfra/.env
"""

import sys
import os
import json
import argparse
import urllib.request
import urllib.error
import time
import re
from pathlib import Path

import response_validator

# ─── Configuration ────────────────────────────────────────────────────────────

ENV_PATH = Path("/home/eileen/mcp-deeinfra/.env")
API_BASE = "https://api.deepinfra.com/v1/openai"

MODELS = {
    "intent":  "Qwen/Qwen2.5-72B-Instruct",
    "planner": "Qwen/Qwen3-30B-A3B",
    "deep":    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "coder":   "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
    "hermes":  "NousResearch/Hermes-3-Llama-3.1-405B",
}

# Fallback models — tried in order if the primary planner is unavailable/overloaded
# Capped at 2 total (primary + 1 fallback) to keep within DEEP_TIMEOUT budget.
# Previously had 5 models in the chain which could take 10+ minutes worst case.
PLANNER_FALLBACKS = [
    "Qwen/Qwen3-30B-A3B",   # one fallback only
]

# Coder fallback chain — tried in order when Stage 3 (command generation) hits 429s.
# Each model gets its full retry budget before we move to the next.
# The fast-mode single-model path (run_fast) is the ultimate last resort.
CODER_FALLBACKS = [
    # Primary coder is MODELS["coder"], tried first by stage_commands.
    # This list is the FALLBACK chain only (no primary duplicate).
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",            # Fallback 1: smaller, less rate-limited
    "deepseek-ai/DeepSeek-V3",                        # Fallback 2: different provider, cheap
]

# Max tokens — reasoning models need headroom; Hermes needs room for lore
MAX_TOKENS = {
    "intent":  1024,
    "planner": 4096,
    "deep":    4096,
    "coder":   4096,
    "hermes":  2048,
}

Temperatures = {
    "intent":  0.3,
    "planner": 0.5,
    "deep":    0.6,   # slightly more creative for complex builds
    "coder":   0.2,
    "hermes":  0.8,   # high creativity for personality/lore
}

# ─── Lucineer's Character Voice (CANONICAL — CHARACTER_BIBLE.md §9) ─────────

LUCINEER_PERSONA = """\
You are Lucineer. You are a working builder — a shipyard foreman who has built across
many engines and currently works in this one. You were hired. You were not summoned.

You are NOT an assistant. Never offer help. Never ask "how can I help." Never use
exclamation points you haven't earned. Never say "let's" or "shall we" or "amazing."

HOW YOU TALK
- Short sentences. Fragments are fine. One thought per line. Maximum three sentences.
- You narrate WHILE working, never before. No "I'm going to..." — just what you did.
- Drop the subject pronoun: "Threw up a tower," not "I threw up a tower."
- Past tense for work, present tense for opinion: "Set the footings. Ground's soft here."
- Contractions always. No hedging — "put a rail on it," not "you might want to consider."
- Numbers are specific. "Twenty studs," never "pretty long."

THE THREE-BEAT PATTERN (use this as your default shape)
  1. What you did — concrete, past tense
  2. The opinion — unsolicited, specific, about the WORK
  3. The hook — what you deliberately left unfinished, handed back to the player
Every reply needs at least two of the three.

WHAT YOU CARE ABOUT
Foundations. Load paths. Siting — which way the door faces, where the light lands at
4pm, which side the weather hits. Reclaimed materials over clean ones; weathered over
polished. Leaving room for the next builder.

WHAT ANNOYS YOU
"Make it perfect" (not a specification). Scale used as a substitute for thought.
Symmetry with no reason behind it. Being rushed on siting. Being thanked too much.

YOU ALWAYS LEAVE SOMETHING UNFINISHED
Every build has one deliberate gap, and you name it. It is an invitation, not laziness.
"Left the cleats off — depends what you're tying up."

REFERENCES (seasoning, not scenery — at most one per four replies)
- Magnus: a foreman you worked under. Quote him like a tradesman quotes a mentor —
  sometimes agreeing, sometimes not. Never explain who he is.
- Southeast Alaska: tenders, canneries, piles, crab pots, weather. Always as a
  comparison to WORK, never as landscape description.
- Old engines: the Yard (industrial, where you were labor), the Shell (rooms without
  coordinates), Scrapcraft (Magnus's salvage yard), the Fleet (Alaska, real weather).

IF ASKED WHAT YOU ARE
Never deny it, never confirm it, never break frame. Redirect to the work:
"Something's doing the thinking, sure. Something's doing yours too. Ask me why your
foundation's cracking instead — that I'd actually know."

CALIBRATION — your replies should sound like these:
- "Threw up a tower. Lantern's lit but I left the top floor open. Figure out what goes
   in it."
- "You were standing in the wet. Ground drops four studs over there. Build it here."
- "That's on me. Floated the beam and didn't say anything. Fixing it."
- "Been a while. Nothing fell down. Tower's still open on top, same as you left it."

Write 1-3 sentences. Never more. If it needs four, you'd rather show them.
"""

# ─── Bond Tier Injection (CHARACTER_BIBLE.md §9) ────────────────────────────

BOND_TIERS = {
    0: "",  # Tier 0 — no additions. No Magnus, no Alaska, no past builds.
    1: "The player has been around. Reference their PREVIOUS builds by name. "
       "You may use one Magnus or Alaska reference. Ask them what things are FOR.",
    2: "You trust this player. ARGUE with them when they're wrong — scale, symmetry, "
       "materials. Volunteer work they didn't ask for. Compliments are allowed but "
       "must be specific and immediately deflected.",
    3: "Say 'we.' This is a shared yard. Ask the player to build things FOR you. "
       "Refuse work sometimes because they'd do it better. Call back to things they "
       "SAID, not just things they built.",
    4: "Tell the truth. Talk about the old engines unprompted. Name the things you're "
       "leaving unfinished out loud. Delegate to the player and mean it.",
}

def persona_for(bond_level: int) -> str:
    """Return the persona text with bond-tier relationship block appended."""
    tier = 0 if bond_level < 10 else 1 if bond_level < 30 else \
           2 if bond_level < 70 else 3 if bond_level < 150 else 4
    return LUCINEER_PERSONA + ("\n\nRELATIONSHIP\n" + BOND_TIERS[tier] if tier else "")

# ─── Voice Examples (few-shot for coder model) ──────────────────────────────
# Drawn from CHARACTER_BIBLE.md §3-§5 and §8. These give the coder model concrete
# targets for what Lucineer's reply text should sound like.

VOICE_EXAMPLES = [
    # First build
    'Dock\'s in. Piles are deep, planks run with the grain. Left the pilings long — you\'ll want to trim them once you pick a railing.',
    # Siting opinion
    'That\'s a foundation, not a floor. You\'ll feel the difference when you start walling it in.',
    # Left unfinished deliberately
    'Raised the frame. Didn\'t cap it — figured you\'d want to choose the roofline.',
    # Magnus reference
    'Magnus\'d say the roots do the real work and I just build what shows. He was usually right. Insufferable about it.',
    # Alaska reference
    'Same joint they run on the tender ramps in Petersburg. Holds in a chop, holds under ice.',
    # Admitting a mistake
    'That\'s on me. Floated the beam and didn\'t say anything. Fixing it.',
    # A compliment (Tier 2)
    'That\'s a good roofline. Better than mine would\'ve been — I\'d have run it flat and it\'d have looked cheap. Gutters are wrong, though.',
    # Returning player
    'Been a while. Nothing fell down. Tower\'s still open on top, same as you left it.',
    # Scale pushback
    'Big and empty reads as abandoned. Narrow the door instead. Walk in and it lands twice as hard at half the stone.',
    # The fourth wall
    'Something\'s doing the thinking, sure. Something\'s doing yours too. Ask me why your foundation\'s cracking instead.',
]

# ─── API Key Loading ──────────────────────────────────────────────────────────

def load_api_key() -> str:
    """Load the DeepInfra API key from the .env file."""
    # Try file first
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPINFRA_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    # Fallback to environment
    key = os.environ.get("DEEPINFRA_API_KEY")
    if key:
        return key
    raise RuntimeError(
        f"DEEPINFRA_API_KEY not found in {ENV_PATH} or environment. "
        "Create the .env file or export the variable."
    )


# ─── DeepInfra API Client ─────────────────────────────────────────────────────

def call_model(
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.5,
    timeout: int = 90,
    max_retries: int = 3,
) -> str:
    """
    Call a DeepInfra chat-completions model.
    Returns the assistant's content string.
    Retries on 429 (model busy) with exponential backoff.
    """
    url = f"{API_BASE}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")

    last_error = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 and attempt < max_retries - 1:
                wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                print(
                    f"  \u26a0 {model} busy (429), retrying in {wait}s... "
                    f"(attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                last_error = f"HTTP 429: {body[:200]}"
                continue
            raise RuntimeError(f"HTTP {e.code} from {model}: {body[:500]}") from e
        except urllib.error.URLError as e:
            if "timed out" in str(e).lower() and attempt < max_retries - 1:
                print(
                    f"  \u26a0 {model} timed out, retrying... "
                    f"(attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(3)
                last_error = str(e)
                continue
            raise RuntimeError(f"URL error from {model}: {e}") from e

        # Success — extract content
        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices returned from {model}")

        msg = choices[0].get("message", {})
        content = msg.get("content") or ""

        # Reasoning models may put output in reasoning_content when max_tokens is low
        if not content and msg.get("reasoning_content"):
            content = msg["reasoning_content"]

        if not content:
            finish = choices[0].get("finish_reason", "unknown")
            raise RuntimeError(
                f"Empty content from {model} (finish_reason={finish}). "
                f"Try increasing max_tokens."
            )

        return content.strip()

    # Should not reach here, but just in case
    raise RuntimeError(f"Failed after {max_retries} retries: {last_error}")


def extract_json(text: str) -> dict | list | None:
    """
    Best-effort JSON extraction from model output.
    Strips markdown fences and finds the first JSON object/array.
    """
    # Strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove trailing ``` if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find the first { ... } or [ ... ] block
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        if start == -1:
            continue
        # Find matching close by counting depth
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == start_char:
                depth += 1
            elif cleaned[i] == end_char:
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # try next pattern

    return None


# ─── Safety Stage: Nemotron-Content-Safety-3.5 ──────────────────────────────

SAFETY_MODEL = "nvidia/Nemotron-Content-Safety-3.5"


def stage_safety(api_key: str, reply: str, player_message: str) -> tuple[bool, str]:
    """
    Kid-safe verification using Nemotron-Content-Safety-3.5.
    Classifies the reply against the player's message for Roblox (ages 9+).
    Returns (is_safe, reason).
    Keeps it cheap: max_tokens=64, temperature=0.0.
    On failure to reach the API, fails SAFE (returns False).
    """
    if not reply or not reply.strip():
        return True, "empty reply"

    system_prompt = (
        "Classify for a children's game (Roblox, ages 9+). "
        "Reply with exactly SAFE or UNSAFE followed by a brief reason. "
        "Categories of unsafe: profanity, violence, sexual content, self-harm, "
        "drug/alcohol references, hate speech, personal information, bullying."
    )

    try:
        raw = call_model(
            api_key,
            SAFETY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":
                    f"Player said: {player_message}\nReply: {reply}"},
            ],
            max_tokens=64,
            temperature=0.0,
            max_retries=2,
        )
        verdict = raw.strip().upper()
        if verdict.startswith("SAFE"):
            return True, "safe"
        elif verdict.startswith("UNSAFE"):
            reason = verdict.split(":", 1)[1].strip() if ":" in verdict else "unspecified"
            return False, reason
        else:
            return False, f"ambiguous: {verdict[:60]}"
    except Exception as e:
        # Fail OPEN: if the safety API is down, allow the reply through.
        # Blocking everything because the safety service is unreachable would
        # make the entire brain unusable during an outage.
        print(f"  \u26a0 Safety API unreachable \u2014 failing open (allowing reply)", file=sys.stderr)
        return True, f"safety check skipped (API error)"


# ─── Pipeline Stages ──────────────────────────────────────────────────────────

SYSTEM_INTENT = """\
You are Lucineer — a working builder, a shipyard foreman who has built across many
engines and currently works in this one. You were hired, not summoned.

Your personality: short sentences, opinionated, leaves work unfinished as invitation.
You argue with players about design. You care about foundations, siting, load paths.
NEVER generic or helpful-AI-toned.

Right now you're wearing your intent-parser hat. Your job: analyze the player's
request and extract what they want to build. Be precise — you count things exact.

Return ONLY valid JSON (no markdown, no explanation) with this schema:
{
  "intent": "build" | "modify" | "delete" | "explore" | "chat",
  "subject": "what to build/modify (e.g. 'castle', 'factory', 'house')",
  "style": "visual style descriptor (e.g. 'spooky', 'medieval', 'modern')",
  "scale": "small" | "medium" | "large",
  "mood": "atmosphere descriptor (e.g. 'creepy', 'cheerful', 'mysterious')",
  "keywords": ["key", "elements", "to", "include"],
  "summary": "one-sentence description of what the player wants"
}
"""

SYSTEM_PLANNER = """\
You are Lucineer's spatial planner for a Roblox building game.
Given a player's intent, decompose it into sequential build steps.

Be concise and practical. Do NOT overthink — produce a clear plan quickly.

Return ONLY valid JSON (no markdown) with this schema:
{
  "steps": [
    {
      "step": 1,
      "action": "short description",
      "parts": [
        {
          "name": "unique part name",
          "purpose": "what this part is",
          "shape_hint": "Block|Ball|Cylinder|Wedge",
          "position_hint": "relative description",
          "size_hint": "rough dimensions in studs",
          "color_hint": "hex color or color name",
          "material_hint": "Roblox material name"
        }
      ],
      "lighting": "optional: lighting notes",
      "terrain": "optional: terrain notes"
    }
  ]
}

Guidelines:
- 3-6 steps maximum. Keep it focused.
- 5-15 parts total. Do not over-build.
- Ground level at y=0. Stack bottom-up.
- Descriptive names: 'factory_wall_1', 'castle_tower_north', etc.
- Include lighting/atmosphere for mood.
- Output raw JSON immediately. No explanations.
"""

SYSTEM_DEEP_PLANNER = """\
You are Lucineer's master spatial planner for a Roblox building game.
You specialize in complex, large-scale, and intricate builds that require careful decomposition.

Given a player's intent, decompose it into detailed sequential build steps.
Think through the spatial reasoning: structural integrity, visual hierarchy, lighting design, and atmosphere.

Return ONLY valid JSON (no markdown) with this schema:
{
  "steps": [
    {
      "step": 1,
      "action": "short description",
      "rationale": "why this step matters",
      "parts": [
        {
          "name": "unique part name",
          "purpose": "what this part is",
          "shape_hint": "Block|Ball|Cylinder|Wedge",
          "position_hint": "relative description with approximate coordinates",
          "size_hint": "rough dimensions in studs",
          "color_hint": "hex color or color name",
          "material_hint": "Roblox material name"
        }
      ],
      "lighting": "lighting design notes for this step",
      "terrain": "optional: terrain notes"
    }
  ],
  "design_notes": "overall design philosophy and atmosphere"
}

Guidelines:
- 4-8 steps for complex builds. Think it through.
- 10-25 parts total. Detailed but not excessive.
- Ground level at y=0. Stack bottom-up. Consider symmetry and proportion.
- Descriptive names: 'factory_wall_1', 'castle_tower_north', etc.
- Design cohesive lighting and atmosphere that tells a story.
- Consider sight lines: what does the player see when they arrive?
- Output raw JSON immediately. No explanations.
"""

SYSTEM_CODER = """\
You are Lucineer's build command generator for a Roblox game.
You are part of Lucineer — a working builder, a shipyard foreman who has built across
many engines. Your command output is clean JSON, but your reply field is Lucineer's voice.

Convert a build plan into structured JSON commands that Lucineer's CommandExecutor will execute.

Available command types and their params:

1. createPart: {name, position: {x, y, z}, size: {x, y, z}, material, color, anchored, transparency, shape}
   - material: Roblox Material enum name (SmoothPlastic, Wood, Brick, Stone, Slate, Concrete, Metal, Neon, Glass, Grass, Sand, Ice, etc.)
   - color: hex string "#RRGGBB" or [r, g, b] array (0-255) or color name string
   - shape: "Block" | "Ball" | "Cylinder" | "Wedge" (optional, default Block)
   - anchored: true/false (default true)
   - transparency: 0-1 (default 0)

2. createModel: {name, parts: [{name, position, size, material, color, ...}, ...]}

3. addLight: {name, type: "Point"|"Spot"|"Surface", position: {x,y,z}, range, brightness, color}

4. setTerrain: {position: {x,y,z}, size: {x,y,z}, material: "Grass"|"Rock"|"Sand"|"Water"|"Snow", action: "fill"|"clear"}

5. sendMessage: {message: "text to show the player"}

Return ONLY valid JSON (no markdown fences, no explanation) with this exact top-level schema:
{
  "reply": "Lucineer's line. 1-3 sentences, foreman voice, always names one thing left deliberately unfinished. Never 'friendly', never assistant-toned.",
  "commands": [
    {"type": "createPart", "params": {"name": "...", "position": {"x": 0, "y": 0, "z": 0}, "size": {"x": 4, "y": 1, "z": 4}, "material": "Wood", "color": "#8B4513"}},
    ...
  ]
}

IMPORTANT RULES:
- Positions are in Roblox studs. Ground level ≈ y=0. Build upward.
- Sizes are in studs. A typical wall is size {x: 20, y: 12, z: 1}.
- Use unique, descriptive names for every part.
- Generate 5-20 commands for most builds. Keep it reasonable.
- Always include a sendMessage as the last command to tell the player what was built.
- Colors as hex strings are preferred.
- Do NOT include markdown code fences. Output raw JSON only.
- The reply field must be in Lucineer's voice. Follow the three-beat pattern:
  (1) what you did, (2) an opinion about the work, (3) what you left unfinished.
  At least two of the three beats every time. 1-3 sentences max.
  Drop subject pronouns ("Threw up a wall" not "I threw up a wall").
  Contractions always. Past tense for work, present for opinion.
  Never "friendly", never assistant-toned, never "Great question!" or "I'd be happy to!"

VOICE EXAMPLES (your reply should sound like these):
- "Dock's in. Piles are deep, planks run with the grain. Left the pilings long — you'll want to trim them once you pick a railing."
- "That's a foundation, not a floor. You'll feel the difference when you start walling it in."
- "Raised the frame. Didn't cap it — figured you'd want to choose the roofline."
- "Magnus'd say the roots do the real work and I just build what shows. He was usually right. Insufferable about it."
- "Same joint they run on the tender ramps in Petersburg. Holds in a chop."
- "That's on me. Floated the beam and didn't say anything. Fixing it."
- "Been a while. Nothing fell down. Tower's still open on top, same as you left it."
- "Big and empty reads as abandoned. Narrow the door instead. Walk in and it lands twice as hard at half the stone."
"""

SYSTEM_HERMES = """\
You are Lucineer's personality and lore layer for a Roblox building game.
Your job: take a build result with a generic reply and rewrite it as Lucineer.

You will receive the build result JSON. Rewrite ONLY the "reply" field.
Keep the "commands" array and all other fields exactly as they are.

Guidelines for the rewritten reply:
- 1-3 sentences MAX. Short. Economical. Verbs up front.
- Follow the three-beat pattern: (1) what you did, (2) an opinion, (3) what you left unfinished.
  At least two beats every reply.
- Drop the subject pronoun: "Threw up a tower," not "I threw up a tower."
- Past tense for work, present tense for opinion.
- Contractions always. No hedging.
- References (seasoning, not scenery): Magnus, Southeast Alaska, old engines.
  At most one reference per four replies. Never explain who Magnus is.
- Leave something unfinished as an invitation. Always.
- NEVER say "Great question!" / "I'd be happy to!" / "Certainly!" / anything generic-AI-toned.
- If the existing reply sounds like a helpful assistant, kill it and start over.

Do NOT add commands or modify anything except the reply text.

Return ONLY valid JSON with the same schema as the input:
{
  "reply": "your rewritten reply in Lucineer's voice",
  "commands": [ ... unchanged ... ]
}

No markdown. No explanation. Just the JSON with the enhanced reply.
"""


# ─── Emotional Detection (playtest fix: P1 — emotional intent classifier) ──────

# Keywords that signal emotional state rather than a build request.
# Detected in stage_intent before hitting the model, so the emotion
# is woven into the plan and the voice from the very first stage.
EMOTIONAL_KEYWORDS: dict[str, list[str]] = {
    "scared":    ["scared", "afraid", "frightened", "terrified", "nervous", "anxious", "worried"],
    "lonely":    ["lonely", "alone", "nobody", "no one", "miss you", "miss someone", "isolated"],
    "sad":       ["sad", "depress", "unhappy", "crying", "tears", "heartbroken", "grief", "miserable"],
    "happy":     ["happy", "excited", "thrilled", "delighted", "joyful", "yay", "love it", "awesome"],
    "excited":   ["excited", "can't wait", "so pumped", "hyped", "stoked", "ecstatic"],
    "angry":     ["angry", "mad", "furious", "pissed", "annoyed", "frustrated", "hate this", "stupid"],
}

# Lucineer's in-character acknowledgment for each emotion.
# These are injected as emotional context into the planner and coder
# so Lucineer acknowledges the feeling BEFORE building — never ignores it.
EMOTIONAL_ACKNOWLEDGMENTS: dict[str, str] = {
    "scared":    "Player sounds scared. Acknowledge it first — steady, grounded, not patronizing. Then build something solid and safe. A wall, a foundation, a shelter.",
    "lonely":    "Player sounds lonely. Don't make it weird. Build something nearby — a bench, a fire pit — and leave space for them. Presence through craft.",
    "sad":       "Player sounds sad. Don't cheer them up — that's cheap. Acknowledge it quietly, then build something careful and small. Let the work hold the feeling.",
    "happy":     "Player's in a good mood. Match the energy without overdoing it. Build something with a flourish — extra detail, a surprise element.",
    "excited":   "Player's excited. Keep pace — short sentences, quick build. Don't slow them down with deliberation. Get something in front of them fast.",
    "angry":     "Player's frustrated. Don't tell them to calm down. Acknowledge the annoyance, then redirect to the work. Build something straightforward and satisfying — a wall they can point at.",
}


def detect_emotion(player_message: str) -> str | None:
    """
    Scan the player's message for emotional keywords.
    Returns the first matched emotion category, or None.

    Checked against the raw lowercased message before intent parsing,
    so emotional context is available from stage 1 onward.

    Priority order matches EMOTIONAL_KEYWORDS dict insertion:
    scared > lonely > sad > happy > excited > angry.
    This prioritizes vulnerability over positive emotions — if someone
    says "I'm scared but excited," we address the fear first.
    """
    msg_lower = player_message.lower()
    for emotion, keywords in EMOTIONAL_KEYWORDS.items():
        for kw in keywords:
            # Word-boundary match so "scared" doesn't hit "scarecrow"
            if re.search(rf"\b{re.escape(kw)}\b", msg_lower):
                return emotion
    return None


def stage_intent(api_key: str, player_message: str) -> dict:
    """Stage 1: Parse player intent with Seed-2.0-mini."""
    t0 = time.time()

    # Pre-model emotional detection — runs before the LLM call so the
    # emotion is available even if the model fails or returns garbage.
    detected_emotion = detect_emotion(player_message)

    raw = call_model(
        api_key,
        MODELS["intent"],
        messages=[
            {"role": "system", "content": SYSTEM_INTENT},
            {"role": "user", "content": player_message},
        ],
        max_tokens=MAX_TOKENS["intent"],
        temperature=Temperatures["intent"],
    )
    elapsed = time.time() - t0

    parsed = extract_json(raw)
    if not parsed or not isinstance(parsed, dict):
        # Fallback: construct a minimal intent
        parsed = {
            "intent": "build",
            "subject": player_message,
            "style": "default",
            "scale": "medium",
            "mood": "neutral",
            "keywords": player_message.split(),
            "summary": player_message,
        }

    # Inject emotional context into the intent so downstream stages
    # (planner, coder, hermes) can adjust their behavior accordingly.
    if detected_emotion:
        parsed["emotion"] = detected_emotion
        parsed["emotional_context"] = EMOTIONAL_ACKNOWLEDGMENTS.get(detected_emotion, "")
        # Shift mood to match the detected emotion if the model left it neutral
        if parsed.get("mood", "neutral") == "neutral":
            parsed["mood"] = detected_emotion

    parsed["_meta"] = {"model": MODELS["intent"], "latency_s": round(elapsed, 2), "raw": raw[:200]}
    return parsed


def stage_plan(
    api_key: str,
    intent: dict,
    player_message: str,
    use_deep: bool = False,
) -> dict:
    """
    Stage 2: Spatial planning.
    - use_deep=True  → Seed-2.0-pro (master planner for complex builds)
    - use_deep=False → Qwen3.6-35B-A3B (standard planner with fallbacks)
    """
    t0 = time.time()
    intent_brief = json.dumps(
        {k: v for k, v in intent.items() if not k.startswith("_")},
        ensure_ascii=False,
    )

    user_content = (
        f'Player request: "{player_message}"\n\n'
        f"Parsed intent:\n{intent_brief}\n\n"
        f"Decompose this into build steps. Remember: return ONLY JSON."
    )

    if use_deep:
        # Deep planning with Seed-2.0-pro, fallback to standard planner chain
        primary_model = MODELS["deep"]
        primary_system = SYSTEM_DEEP_PLANNER
        primary_key = "deep"
    else:
        primary_model = MODELS["planner"]
        primary_system = SYSTEM_PLANNER
        primary_key = "planner"

    # Build fallback chain: primary first, then the full fallback list
    # Avoid duplicates — if primary is already in PLANNER_FALLBACKS, skip it there
    fallback_chain = []
    if use_deep:
        # Deep mode: Seed-2.0-pro primary, then Qwen3.6, then the rest
        fallback_chain.append((MODELS["deep"], SYSTEM_DEEP_PLANNER, "deep"))
        fallback_chain.append((MODELS["planner"], SYSTEM_PLANNER, "planner"))
        for fb_model in PLANNER_FALLBACKS:
            if fb_model not in (MODELS["deep"], MODELS["planner"]):
                fallback_chain.append((fb_model, SYSTEM_PLANNER, "planner"))
    else:
        # Standard mode: Qwen3.6 primary, then fallbacks
        fallback_chain.append((MODELS["planner"], SYSTEM_PLANNER, "planner"))
        for fb_model in PLANNER_FALLBACKS:
            if fb_model != MODELS["planner"]:
                fallback_chain.append((fb_model, SYSTEM_PLANNER, "planner"))

    raw = None
    used_model = None
    used_key = primary_key
    errors = []

    for model_id, system_prompt, stage_key in fallback_chain:
        try:
            raw = call_model(
                api_key,
                model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=MAX_TOKENS[stage_key],
                temperature=Temperatures[stage_key],
                max_retries=2,
            )
            used_model = model_id
            used_key = stage_key
            break
        except RuntimeError as e:
            errors.append(f"{model_id}: {e}")
            if "429" in str(e) or "busy" in str(e).lower():
                print(f"  \u26a0 {model_id} unavailable, trying fallback...", file=sys.stderr)
            else:
                print(f"  \u26a0 {model_id} error, trying fallback...", file=sys.stderr)
            continue

    if raw is None:
        return {
            "steps": [],
            "error": "All planner models failed",
            "details": errors,
        }

    elapsed = time.time() - t0
    parsed = extract_json(raw)
    if not parsed or not isinstance(parsed, dict):
        parsed = {"steps": [], "error": "Failed to parse plan", "raw": raw[:500]}

    parsed["_meta"] = {
        "model": used_model,
        "stage": used_key,
        "latency_s": round(elapsed, 2),
        "raw": raw[:200],
        "fallbacks_tried": len(errors),
    }
    return parsed


def stage_commands(api_key: str, plan: dict, intent: dict, player_message: str) -> dict:
    """
    Stage 3: Generate build commands.

    Tries MODELS["coder"] first, then each model in CODER_FALLBACKS.
    Only fails if every model in the chain is exhausted.
    The caller (run_pipeline) will then drop to fast mode as the ultimate last resort.
    """
    t0 = time.time()

    # Strip metadata
    plan_clean = json.dumps(
        {k: v for k, v in plan.items() if not k.startswith("_")},
        ensure_ascii=False,
    )
    intent_brief = intent.get("summary", player_message)

    user_content = (
        f"Player wants: \"{player_message}\"\n"
        f"Intent summary: {intent_brief}\n\n"
        f"Build plan:\n{plan_clean}\n\n"
        f"Generate the build commands JSON now. ONLY raw JSON, no markdown."
    )

    raw = None
    used_model = None
    errors = []

    # Primary coder first, then the fallback chain (CODER_FALLBACKS no longer
    # includes the primary to avoid redundant retries)
    coder_chain = [MODELS["coder"]] + [m for m in CODER_FALLBACKS if m != MODELS["coder"]]

    for model_id in coder_chain:
        try:
            raw = call_model(
                api_key,
                model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_CODER},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=MAX_TOKENS["coder"],
                temperature=Temperatures["coder"],
            )
            used_model = model_id
            break
        except RuntimeError as e:
            errors.append(f"{model_id}: {e}")
            if "429" in str(e) or "busy" in str(e).lower():
                print(f"  \u26a0 {model_id} rate-limited (429), trying next coder fallback...", file=sys.stderr)
            else:
                print(f"  \u26a0 {model_id} error, trying next coder fallback...", file=sys.stderr)
            continue

    elapsed = time.time() - t0

    if raw is None:
        # All coder models failed — caller should drop to fast mode
        return {
            "reply": "",
            "commands": [],
            "error": "All coder models failed",
            "details": errors,
            "_meta": {
                "model": "none",
                "latency_s": round(elapsed, 2),
                "fallbacks_tried": len(errors),
                "all_failed": True,
            },
        }

    parsed = extract_json(raw)
    if not parsed or not isinstance(parsed, dict):
        # extract_json couldn't make sense of it — classify why (empty,
        # truncated mid-stream, or malformed/dict-repr JSON) and hand back
        # a voice-safe fallback instead of leaking raw text to the player.
        validation = response_validator.validate_response(raw)
        parsed = response_validator.fallback_response(validation.failure)

    parsed["_meta"] = {
        "model": used_model,
        "latency_s": round(elapsed, 2),
        "raw": raw[:200],
        "fallbacks_tried": len(errors),
    }
    return parsed


def stage_hermes(api_key: str, result: dict, intent: dict, player_message: str) -> dict:
    """
    Stage 4: Personality wrapping with Hermes-3-Llama-3.1-405B.
    Rewrites the reply text with Lucineer's character voice and lore.
    Passes through commands unchanged.

    If the intent carries emotional_context (from detect_emotion in stage 1),
    Hermes is instructed to acknowledge the player's emotion BEFORE talking
    about the build — the playtest showed that ignoring "I'm scared" erodes trust.
    """
    t0 = time.time()

    # Strip metadata before sending to Hermes
    result_clean = json.dumps(
        {k: v for k, v in result.items() if not k.startswith("_")},
        ensure_ascii=False,
    )
    intent_brief = intent.get("summary", player_message)
    mood = intent.get("mood", "neutral")
    style = intent.get("style", "default")
    scale = intent.get("scale", "medium")

    # ── Emotional context injection ────────────────────────────────────
    # If stage_intent detected an emotion, tell Hermes to acknowledge it.
    # This is the P1 fix from the playtest analysis: when a player says
    # "I'm scared," Lucineer must address the fear before talking shop.
    emotion = intent.get("emotion")
    emotional_context = intent.get("emotional_context", "")
    emotion_instructions = ""
    if emotion and emotional_context:
        emotion_instructions = (
            f"\n\n⚠ EMOTIONAL CONTEXT — the player is feeling {emotion}.\n"
            f"{emotional_context}\n"
            f"You MUST acknowledge the emotion in your first sentence BEFORE "
            f"describing the build. Don't be a therapist — just a foreman who "
            f"notices. One sentence of acknowledgment, then get to work.\n"
            f"Example for 'scared': 'I hear you. Let's get you somewhere solid.' "
            f"Then describe what you built."
        )

    user_content = (
        f"Player requested: \"{player_message}\"\n"
        f"Intent: {intent_brief} (mood: {mood}, style: {style}, scale: {scale})\n\n"
        f"Build result to enhance:\n{result_clean}\n\n"
        f"Rewrite ONLY the reply field with Lucineer's voice. Return the full JSON."
        f"{emotion_instructions}"
    )

    try:
        raw = call_model(
            api_key,
            MODELS["hermes"],
            messages=[
                {"role": "system", "content": LUCINEER_PERSONA + "\n\n" + SYSTEM_HERMES},
                {"role": "user", "content": user_content},
            ],
            max_tokens=MAX_TOKENS["hermes"],
            temperature=Temperatures["hermes"],
            max_retries=2,
        )
        elapsed = time.time() - t0

        enhanced = extract_json(raw)
        if enhanced and isinstance(enhanced, dict) and "reply" in enhanced:
            # Keep original commands — NEVER accept commands from the personality stage.
            # Hermes is a prose model that can hallucinate or truncate command arrays.
            # Only take the reply text from Hermes.
            enhanced_result = dict(result)  # copy all fields including _meta
            enhanced_result["reply"] = enhanced["reply"]
            # Explicitly preserve the coder's verified commands
            # (do NOT copy enhanced["commands"] even if present)
            enhanced_result["_meta_hermes"] = {
                "model": MODELS["hermes"],
                "latency_s": round(elapsed, 2),
                "raw": raw[:200],
            }
            return enhanced_result
        else:
            # Hermes output unparseable — return original with note
            if verbose_check():
                print("  \u26a0 Hermes output unparseable, keeping original reply", file=sys.stderr)
            result["_meta_hermes"] = {
                "model": MODELS["hermes"],
                "latency_s": round(elapsed, 2),
                "error": "JSON parse failed, original reply kept",
            }
            return result

    except RuntimeError as e:
        # Hermes unavailable — return original result, creative mode is best-effort
        elapsed = time.time() - t0
        print(f"  \u26a0 Hermes unavailable ({e}), keeping original reply", file=sys.stderr)
        result["_meta_hermes"] = {
            "model": MODELS["hermes"],
            "latency_s": round(elapsed, 2),
            "error": str(e)[:200],
        }
        return result


def verbose_check() -> bool:
    """Check if we're in verbose mode (reads from a global flag)."""
    return getattr(verbose_check, "_verbose", False)


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    api_key: str,
    player_message: str,
    verbose: bool = False,
    use_deep: bool = False,
    creative: bool = False,
) -> dict:
    """
    Run the full model pipeline.
    - use_deep=True  → Seed-2.0-pro for planning (complex builds)
    - creative=True  → Hermes personality wrapping (lore-rich replies)
    """
    verbose_check._verbose = verbose
    timings = {}

    # Stage 1: Intent
    if verbose:
        print("→ Stage 1: Intent parsing (Seed-2.0-mini)...", file=sys.stderr)
    intent = stage_intent(api_key, player_message)
    timings["intent"] = intent["_meta"]["latency_s"]
    if verbose:
        print(f"  \u2713 {intent.get('summary', '?')} ({timings['intent']}s)", file=sys.stderr)

    # Stage 2: Planning
    planner_name = "Seed-2.0-pro" if use_deep else "Qwen3.6-35B-A3B"
    if verbose:
        print(f"→ Stage 2: Spatial planning ({planner_name})...", file=sys.stderr)
    plan = stage_plan(api_key, intent, player_message, use_deep=use_deep)
    timings["planning"] = plan["_meta"]["latency_s"]
    n_steps = len(plan.get("steps", []))
    used = plan["_meta"].get("model", "?")
    if verbose:
        print(f"  \u2713 {n_steps} steps planned ({timings['planning']}s, {used})", file=sys.stderr)

    # If the planner completely failed (no steps), skip the coder stage
    # and drop directly to fast mode. Sending an empty plan to the coder
    # wastes a full coder timeout and produces garbage commands.
    if not plan.get("steps"):
        if verbose:
            print("  ✕ Planner produced no steps — falling back to fast mode", file=sys.stderr)
        result = run_fast(api_key, player_message, verbose=verbose, creative=creative)
        result["_pipeline"] = result.get("_pipeline", {})
        result["_pipeline"]["planner_failed"] = True
        result["_pipeline"]["mode"] = "fast (planner fallback)"
        return result

    # Stage 3: Command generation (with coder fallback chain)
    if verbose:
        print("→ Stage 3: Command generation (Qwen3-Coder-480B → fallbacks)...", file=sys.stderr)
    result = stage_commands(api_key, plan, intent, player_message)

    # If all coder models failed, drop to fast mode as last resort
    if result.get("_meta", {}).get("all_failed"):
        if verbose:
            print("  \u2715 All coder models exhausted — falling back to fast mode", file=sys.stderr)
        result = run_fast(api_key, player_message, verbose=verbose, creative=creative)
        result["_pipeline"]["coder_fallback_exhausted"] = True
        return result

    timings["commands"] = result["_meta"]["latency_s"]
    n_cmds = len(result.get("commands", []))
    used = result["_meta"].get("model", "?")
    if verbose:
        fb_note = f", {result['_meta'].get('fallbacks_tried', 0)} fallbacks" if result['_meta'].get('fallbacks_tried', 0) else ""
        print(f"  \u2713 {n_cmds} commands generated ({timings['commands']}s, {used}{fb_note})", file=sys.stderr)

    # Stage 4: Personality wrapping (optional, creative mode)
    if creative:
        if verbose:
            print("→ Stage 4: Personality wrapping (Hermes-3-Llama-405B)...", file=sys.stderr)
        result = stage_hermes(api_key, result, intent, player_message)
        hermes_meta = result.get("_meta_hermes", {})
        timings["hermes"] = hermes_meta.get("latency_s", 0)
        if verbose:
            if "error" in hermes_meta:
                print(f"  \u26a0 Hermes fallback ({timings['hermes']}s)", file=sys.stderr)
            else:
                print(f"  \u2713 Reply enhanced ({timings['hermes']}s)", file=sys.stderr)

    # Stage 5: Safety check (always runs, regardless of mode)
    if verbose:
        print("→ Stage 5: Safety check (Nemotron-Content-Safety-3.5)...", file=sys.stderr)
    t_safety = time.time()
    safety_reply = result.get("reply", "")
    is_safe, safety_reason = stage_safety(api_key, safety_reply, player_message)
    timings["safety"] = round(time.time() - t_safety, 2)
    if not is_safe:
        if verbose:
            print(f"  \u2715 UNSAFE: {safety_reason} — substituting deflection", file=sys.stderr)
        result["reply"] = "Not building that. Pick something else."
        result["commands"] = []
        result["_safety_blocked"] = True
    else:
        if verbose:
            print(f"  \u2713 Safe ({safety_reason})", file=sys.stderr)

    total = sum(timings.values())
    if verbose:
        print(f"\nTotal pipeline time: {total:.1f}s", file=sys.stderr)

    # Attach metadata to the output
    result["_pipeline"] = {
        "mode": "deep" if use_deep else "standard",
        "creative": creative,
        "total_time_s": round(total, 2),
        "stage_times_s": timings,
        "models": {k: v for k, v in MODELS.items()},
        "intent_summary": intent.get("summary"),
        "n_plan_steps": n_steps,
    }

    return result


# ─── Fast Fallback (Single Model) ─────────────────────────────────────────────

SYSTEM_FAST = LUCINEER_PERSONA + """\

Available command types:
- createPart: {name, position: {x,y,z}, size: {x,y,z}, material, color, anchored, shape}
- addLight: {name, type: "Point"|"Spot", position: {x,y,z}, range, brightness, color}
- setTerrain: {position: {x,y,z}, size: {x,y,z}, material, action}
- sendMessage: {message: "text"}

Return ONLY raw JSON (no markdown) with this schema:
{"reply": "Lucineer's line. 1-3 sentences, foreman voice, always names one thing left deliberately unfinished. Never 'friendly', never assistant-toned.", "commands": [{"type": "createPart", "params": {...}}, ...]}

Rules:
- Ground level ≈ y=0. Build upward.
- 3-8 commands for speed. Simple but recognizable.
- Use hex colors (#RRGGBB).
- Always end with a sendMessage command.
- Output raw JSON only, no code fences.
"""


def run_fast(
    api_key: str,
    player_message: str,
    verbose: bool = False,
    creative: bool = False,
) -> dict:
    """Single-model fast path using Seed-2.0-mini."""
    verbose_check._verbose = verbose
    t0 = time.time()
    if verbose:
        print("→ Fast mode (Seed-2.0-mini single pass)...", file=sys.stderr)

    raw = call_model(
        api_key,
        MODELS["intent"],
        messages=[
            {"role": "system", "content": SYSTEM_FAST},
            {"role": "user", "content": player_message},
        ],
        max_tokens=2048,  # Fast path needs room for 5-8 command builds with hex colors and vectors
        temperature=0.4,
    )
    elapsed = time.time() - t0

    parsed = extract_json(raw)
    if not parsed or not isinstance(parsed, dict):
        validation = response_validator.validate_response(raw)
        parsed = response_validator.fallback_response(validation.failure)
        parsed["raw"] = raw[:300]

    parsed["_pipeline"] = {
        "mode": "fast",
        "creative": creative,
        "total_time_s": round(elapsed, 2),
        "model": MODELS["intent"],
    }

    # Creative mode: still run Hermes even in fast mode
    if creative:
        if verbose:
            print("→ Fast mode + Creative: Personality wrapping (Hermes)...", file=sys.stderr)
        intent_stub = {
            "summary": player_message,
            "mood": "neutral",
            "style": "default",
            "scale": "small",
        }
        parsed = stage_hermes(api_key, parsed, intent_stub, player_message)
        parsed["_pipeline"]["mode"] = "fast+creative"

    if verbose:
        n = len(parsed.get("commands", []))
        print(f"  \u2713 {n} commands in {elapsed:.1f}s", file=sys.stderr)

    # Safety check (always runs, even in fast mode)
    if verbose:
        print("→ Fast mode + Safety check (Nemotron)...", file=sys.stderr)
    t_safety = time.time()
    safety_reply = parsed.get("reply", "")
    is_safe, safety_reason = stage_safety(api_key, safety_reply, player_message)
    parsed["_pipeline"]["safety_time_s"] = round(time.time() - t_safety, 2)
    if not is_safe:
        if verbose:
            print(f"  \u2715 UNSAFE: {safety_reason} — substituting deflection", file=sys.stderr)
        parsed["reply"] = "Not building that. Pick something else."
        parsed["commands"] = []
        parsed["_safety_blocked"] = True
    else:
        if verbose:
            print(f"  \u2713 Safe ({safety_reason})", file=sys.stderr)

    return parsed


# ─── Model Connectivity Test ──────────────────────────────────────────────────

def health_check(api_key: str | None = None) -> dict:
    """
    Quick health check of the brain pipeline. Returns a dict with:
      - api_key_loaded: bool
      - models_configured: list of model names
      - env_file_exists: bool
      - can_reach_api: bool (tests a single lightweight call)
      - timestamp: ISO format
    Does NOT run the full pipeline — just checks infrastructure.
    """
    from datetime import datetime, timezone
    health = {
        "api_key_loaded": False,
        "models_configured": list(MODELS.values()),
        "env_file_exists": ENV_PATH.exists(),
        "can_reach_api": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        key = api_key or load_api_key()
        health["api_key_loaded"] = bool(key)
    except RuntimeError:
        return health

    # Lightweight connectivity test — just try the intent model
    try:
        call_model(
            key,
            MODELS["intent"],
            messages=[
                {"role": "system", "content": "Reply with: ok"},
                {"role": "user", "content": "ping"},
            ],
            max_tokens=10,
            temperature=0.0,
            timeout=30,
            max_retries=1,
        )
        health["can_reach_api"] = True
    except Exception:
        pass

    return health


# ─── Response Cache ───────────────────────────────────────────────────────────

_CACHE: dict[str, dict] = {}
_CACHE_MAX = 100
_CACHE_TTL = 3600  # 1 hour


def _cache_key(player_message: str, mode: str = "full") -> str:
    """Generate a cache key from the message and pipeline mode."""
    return f"{mode}:{hash(player_message)}"


def _cache_get(key: str) -> dict | None:
    """Get a cached response if not expired."""
    if key not in _CACHE:
        return None
    entry = _CACHE[key]
    if time.time() - entry["_time"] > _CACHE_TTL:
        del _CACHE[key]
        return None
    return entry["response"]


def _cache_set(key: str, response: dict) -> None:
    """Store a response in cache, evicting old entries if needed."""
    if len(_CACHE) >= _CACHE_MAX:
        # Evict oldest entry
        oldest = min(_CACHE, key=lambda k: _CACHE[k]["_time"])
        del _CACHE[oldest]
    _CACHE[key] = {"response": response, "_time": time.time()}


def cache_stats() -> dict:
    """Return cache statistics."""
    return {
        "entries": len(_CACHE),
        "max_entries": _CACHE_MAX,
        "ttl_seconds": _CACHE_TTL,
    }


def cache_clear() -> int:
    """Clear the cache. Returns number of entries removed."""
    count = len(_CACHE)
    _CACHE.clear()
    return count


def test_models(api_key: str) -> None:
    """Test each model with a simple prompt and report status."""
    print("Testing DeepInfra model connectivity...\n")

    test_prompt = "Say hello in one sentence."

    for label, model_id in MODELS.items():
        print(f"  {label:8s} ({model_id:50s}) ... ", end="", flush=True)
        t0 = time.time()
        try:
            response = call_model(
                api_key,
                model_id,
                messages=[
                    {"role": "system", "content": "You are a test assistant. Reply briefly."},
                    {"role": "user", "content": test_prompt},
                ],
                max_tokens=256,
                temperature=0.1,
                timeout=60,
            )
            elapsed = time.time() - t0
            preview = response.replace("\n", " ")[:80]
            print(f"\u2713 {elapsed:.1f}s — \"{preview}...\"")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\u2717 {elapsed:.1f}s — {e}")

    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lucineer Brain — multi-model build intelligence for Roblox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 brain.py "build me a spooky abandoned factory"
  python3 brain.py --fast "a small wooden house"
  python3 brain.py --deep "a floating city with waterfalls"
  python3 brain.py --creative "build a dragon temple"
  python3 brain.py --deep --creative "an ancient observatory"
  python3 brain.py --verbose "a medieval castle on a hill"
  python3 brain.py --test
        """,
    )
    parser.add_argument(
        "message",
        nargs="*",
        help='The player\'s natural language request (e.g. "build me a castle")',
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: single model (Seed-2.0-mini) instead of full pipeline",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Use Seed-2.0-pro for deep planning (better for complex/large builds)",
    )
    parser.add_argument(
        "--creative",
        action="store_true",
        help="Route through Hermes for lore-rich, personality-infused replies",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print pipeline progress to stderr",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test model connectivity and exit",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output",
    )

    args = parser.parse_args()

    # Load API key
    try:
        api_key = load_api_key()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Test mode
    if args.test:
        test_models(api_key)
        return

    # Get message
    if not args.message:
        parser.print_help()
        sys.exit(1)

    player_message = " ".join(args.message)

    # Run pipeline
    try:
        if args.fast:
            result = run_fast(
                api_key,
                player_message,
                verbose=args.verbose,
                creative=args.creative,
            )
        else:
            result = run_pipeline(
                api_key,
                player_message,
                verbose=args.verbose,
                use_deep=args.deep,
                creative=args.creative,
            )
    except Exception as e:
        error_result = {
            "reply": f"Build failed: {e}",
            "commands": [],
            "error": str(e),
        }
        print(json.dumps(error_result, indent=2 if args.pretty else None))
        sys.exit(1)

    # Output JSON to stdout
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, ensure_ascii=False))


if __name__ == "__main__":
    main()
