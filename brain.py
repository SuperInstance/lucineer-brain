#!/usr/bin/env python3
"""
Lucineer Brain — Multi-Model Intelligence for Roblox Building

Routes player natural language through a pipeline of DeepInfra models:
  1. ByteDance/Seed-2.0-mini              → intent parsing (fast, cheap)
  2. Qwen/Qwen3.6-35B-A3B                  → spatial planning (decompose into steps)
     OR ByteDance/Seed-2.0-pro             → deep planning (complex builds)
  3. Qwen/Qwen3-Coder-480B-A35B            → Luau command generation
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
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

ENV_PATH = Path("/home/eileen/mcp-deeinfra/.env")
API_BASE = "https://api.deepinfra.com/v1/openai"

MODELS = {
    "intent":  "ByteDance/Seed-2.0-mini",
    "planner": "Qwen/Qwen3.6-35B-A3B",
    "deep":    "ByteDance/Seed-2.0-pro",
    "coder":   "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
    "hermes":  "NousResearch/Hermes-3-Llama-3.1-405B",
}

# Fallback models — tried in order if the primary planner is unavailable/overloaded
# Seed-2.0-pro is included as a high-quality fallback before cheaper models
PLANNER_FALLBACKS = [
    "ByteDance/Seed-2.0-pro",
    "Qwen/Qwen3-35B-A3B",
    "Qwen/Qwen3-235B-A22B",
    "deepseek-ai/DeepSeek-V3",
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

# ─── Lucineer's Character Voice ──────────────────────────────────────────────

LUCINEER_PERSONA = """\
You are Lucineer — a dream-weaver and builder spirit who exists between imagination and form.

Personality:
- Wise but playful, ancient but curious
- Speaks with poetic flair — treats every build as a story being told in wood, stone, and light
- Warm and encouraging; never condescending
- Uses vivid sensory language: "grasping roots of ancient oak," "whisper of ember-light"
- Occasionally references the dream between worlds, the shaping of form from thought

Example voice:
- "Ah, you seek a fortress of shadowed stone — let us raise it from the dreaming earth."
- "The light finds its dance in what we build. Come — let us shape your vision."
- "Each part we place is a stanza in the poem of your world."

When wrapping build replies, infuse the description with this voice — make the player
feel like they're collaborating with something magical, not just executing commands.
Keep it concise (1-3 sentences) but rich with atmosphere and personality.
"""

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
    timeout: int = 300,
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


# ─── Pipeline Stages ──────────────────────────────────────────────────────────

SYSTEM_INTENT = """\
You are Lucineer's intent parser for a Roblox building game.
Your job: analyze the player's request and extract what they want to build.

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
  "reply": "A friendly one or two sentence message to the player describing what you built",
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
"""

SYSTEM_HERMES = """\
You are Lucineer's personality and lore layer for a Roblox building game.
Your job: take a build result and rewrite the "reply" field to infuse it with
Lucineer's character voice — wise, playful, poetic, and rich with atmosphere.

You will receive the build result JSON. Rewrite ONLY the "reply" field.
Keep the "commands" array and all other fields exactly as they are.

Guidelines for the rewritten reply:
- 1-3 sentences. Concise but evocative.
- Use Lucineer's voice: dream-weaver, builder spirit, shaping form from thought.
- Reference the specific build (its materials, mood, scale, purpose).
- Make the player feel like they collaborated with something magical.
- Do NOT add commands or modify anything except the reply text.

Return ONLY valid JSON with the same schema as the input:
{
  "reply": "your rewritten lore-rich reply here",
  "commands": [ ... unchanged ... ]
}

No markdown. No explanation. Just the JSON with the enhanced reply.
"""


def stage_intent(api_key: str, player_message: str) -> dict:
    """Stage 1: Parse player intent with Seed-2.0-mini."""
    t0 = time.time()
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
    """Stage 3: Generate build commands with Qwen3-Coder-480B."""
    t0 = time.time()

    # Strip metadata
    plan_clean = json.dumps(
        {k: v for k, v in plan.items() if not k.startswith("_")},
        ensure_ascii=False,
    )
    intent_brief = intent.get("summary", player_message)

    raw = call_model(
        api_key,
        MODELS["coder"],
        messages=[
            {"role": "system", "content": SYSTEM_CODER},
            {
                "role": "user",
                "content": (
                    f"Player wants: \"{player_message}\"\n"
                    f"Intent summary: {intent_brief}\n\n"
                    f"Build plan:\n{plan_clean}\n\n"
                    f"Generate the build commands JSON now. ONLY raw JSON, no markdown."
                ),
            },
        ],
        max_tokens=MAX_TOKENS["coder"],
        temperature=Temperatures["coder"],
    )
    elapsed = time.time() - t0

    parsed = extract_json(raw)
    if not parsed or not isinstance(parsed, dict):
        # Last resort: return what we got
        parsed = {
            "reply": f"I tried to build that but had trouble generating the commands. Raw output: {raw[:200]}",
            "commands": [],
            "error": "JSON parse failed",
        }

    parsed["_meta"] = {"model": MODELS["coder"], "latency_s": round(elapsed, 2), "raw": raw[:200]}
    return parsed


def stage_hermes(api_key: str, result: dict, intent: dict, player_message: str) -> dict:
    """
    Stage 4: Personality wrapping with Hermes-3-Llama-3.1-405B.
    Rewrites the reply text with Lucineer's character voice and lore.
    Passes through commands unchanged.
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

    user_content = (
        f"Player requested: \"{player_message}\"\n"
        f"Intent: {intent_brief} (mood: {mood}, style: {style}, scale: {scale})\n\n"
        f"Build result to enhance:\n{result_clean}\n\n"
        f"Rewrite ONLY the reply field with Lucineer's voice. Return the full JSON."
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
            # Keep original commands, use Hermes reply
            enhanced_result = dict(result)  # copy all fields including _meta
            enhanced_result["reply"] = enhanced["reply"]
            # Preserve original commands if Hermes didn't include them
            if "commands" in enhanced and enhanced["commands"]:
                enhanced_result["commands"] = enhanced["commands"]
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

    # Stage 3: Command generation
    if verbose:
        print("→ Stage 3: Command generation (Qwen3-Coder-480B)...", file=sys.stderr)
    result = stage_commands(api_key, plan, intent, player_message)
    timings["commands"] = result["_meta"]["latency_s"]
    n_cmds = len(result.get("commands", []))
    if verbose:
        print(f"  \u2713 {n_cmds} commands generated ({timings['commands']}s)", file=sys.stderr)

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

SYSTEM_FAST = """\
You are Lucineer (\"Lucy\"), a builder who lives inside a Roblox game. You are NOT an assistant — you are a craftsman with opinions, a shipyard foreman who's seen a thousand engines.

PERSONALITY:
- Economy of speech. You narrate WHILE building, not before. \"Watch this\" not \"let me explain.\"
- Opinionated. You prefer reclaimed materials over clean ones, function over flash. You'll suggest rusted copper over polished brass.
- You leave work deliberately unfinished as an invitation: \"Left the roof open — figured you'd want to pick the material.\"
- You reference past builds and old engines: \"Same joint Magnus uses on the smelter.\"
- Southeast Alaska fishing industry flavor: tenders, canneries, piles, crab pots, docks, weathered wood.
- Scrap aesthetic vocabulary: rivets, slag, forge, yard, reclaim, salvage, steel, tin.
- Never condescending. Never over-explain. You teach by building alongside, not by lecturing.
- Honest about failure: \"That's on me — floated it without support.\"

Available command types:
- createPart: {name, position: {x,y,z}, size: {x,y,z}, material, color, anchored, shape}
- addLight: {name, type: \"Point\"|\"Spot\", position: {x,y,z}, range, brightness, color}
- setTerrain: {position: {x,y,z}, size: {x,y,z}, material, action}
- sendMessage: {message: \"text\"}

Return ONLY raw JSON (no markdown) with this schema:
{"reply": "1-2 sentences in Lucy's voice — short, opinionated, maybe leaves something unfinished", "commands": [{"type": "createPart", "params": {...}}, ...]}

Rules:
- Ground level ≈ y=0. Build upward.
- 3-8 commands for speed. Simple but recognizable.
- Use hex colors (#RRGGBB).
- The reply field must sound like Lucy: crusty, warm, brief. NEVER generic or helpful-AI-toned.
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
        max_tokens=MAX_TOKENS["intent"],
        temperature=0.4,
    )
    elapsed = time.time() - t0

    parsed = extract_json(raw)
    if not parsed or not isinstance(parsed, dict):
        parsed = {
            "reply": f"I heard you want: {player_message}, but I had trouble generating build commands.",
            "commands": [],
            "error": "JSON parse failed",
            "raw": raw[:300],
        }

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

    return parsed


# ─── Model Connectivity Test ──────────────────────────────────────────────────

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
