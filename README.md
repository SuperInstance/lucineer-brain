# lucineer-brain

**4-stage multi-model intelligence pipeline that converts natural language into structured Roblox build commands.**

Routes player requests through a chain of DeepInfra models — each specialized for one stage — to produce JSON matching Lucineer's CommandExecutor schema: `{"reply": "...", "commands": [...]}`.

---

## Architecture

```
Player Message
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: INTENT PARSE                                           │
│ Model: ByteDance/Seed-2.0-mini    Channel: 10    Allegro 120+  │
│ Temp: 0.3    Max tokens: 1024                                   │
│ Output: { intent, subject, style, scale, mood, keywords,        │
│           summary }                                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: SPATIAL PLANNING                                       │
│ Model: Qwen/Qwen3.6-35B-A3B       Channel: 11    Moderato 90-110│
│   OR   ByteDance/Seed-2.0-pro (deep mode)                       │
│ Temp: 0.5–0.6    Max tokens: 4096                               │
│ Output: { steps: [{ step, action, parts: [{ name, purpose,      │
│           shape_hint, position_hint, size_hint, color_hint,     │
│           material_hint }], lighting, terrain }] }              │
│                                                                 │
│ Fallback chain: Seed-pro → Qwen3.6 → Qwen3-235B → DeepSeek-V3  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: CODE GENERATION                                        │
│ Model: Qwen/Qwen3-Coder-480B-A35B  Channel: 12   Andante 80-100 │
│ Temp: 0.2    Max tokens: 4096                                   │
│ Output: { reply: "...", commands: [{ type, params }] }          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼ (creative mode only)
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: PERSONALITY WRAP                                       │
│ Model: NousResearch/Hermes-3-Llama-3.1-405B  Ch: 13  Adagio 50-70│
│ Temp: 0.8    Max tokens: 2048                                   │
│ Rewrites "reply" field in Lucineer's voice. Commands unchanged. │
│ Fail-safe: if Hermes unavailable, keeps original reply.         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                    JSON to stdout
                  { reply, commands }
```

### Mode Selection

| Mode | Flag | Pipeline | Use Case |
|------|------|----------|----------|
| **Standard** | (default) | 3-stage: intent → plan → code | Normal builds |
| **Deep** | `--deep` | 3-stage with Seed-2.0-pro planner | Complex/large builds |
| **Creative** | `--creative` | 4-stage + Hermes personality wrap | Lore-rich replies |
| **Fast** | `--fast` | Single-model: Seed-2.0-mini only | Quick fallback, ~2-5s |
| **Fast+Creative** | `--fast --creative` | Fast + Hermes | Quick with personality |

---

## Model Configuration

```python
MODELS = {
    "intent":  "ByteDance/Seed-2.0-mini",
    "planner": "Qwen/Qwen3.6-35B-A3B",
    "deep":    "ByteDance/Seed-2.0-pro",
    "coder":   "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo",
    "hermes":  "NousResearch/Hermes-3-Llama-3.1-405B",
}

Temperatures = { "intent": 0.3, "planner": 0.5, "deep": 0.6, "coder": 0.2, "hermes": 0.8 }
MAX_TOKENS   = { "intent": 1024, "planner": 4096, "deep": 4096, "coder": 4096, "hermes": 2048 }
```

### Planner Fallback Chain

```
Primary: Qwen/Qwen3.6-35B-A3B
    ↓ (429/timeout)
ByteDance/Seed-2.0-pro
    ↓
Qwen/Qwen3-235B-A22B
    ↓
deepseek-ai/DeepSeek-V3
```

Each fallback is tried with 2 retries. 429 responses trigger exponential backoff (5s, 10s, 15s).

---

## Character Voice

Lucineer's persona is defined in a ~2000-word system prompt embedded in the code. Key traits:

- **Economical**: short sentences, verbs up front, talks like someone paying by the word
- **Opinionated**: prefers reclaimed materials, argues with players about design
- **SE Alaska aesthetic**: rivets, slag, forge, yard, tide, the Channel, crab pots, canneries
- **The Unfinished Rule**: every solo build is missing something finishable by a novice — "A finished thing belongs to its maker. An unfinished thing belongs to whoever finishes it."
- **Refusal Protocol**: four grounds (breaks world, cheats player, cruel, boring) — never cites rules or limitations
- **Never says**: "Great question!", "I'd be happy to!", "Certainly!", or anything that sounds like a helpful AI

The persona is injected into Stage 1 system prompts and the fast-mode system prompt. Stage 4 (Hermes) receives the full persona plus a rewriting directive.

---

## Output Schema

```json
{
  "reply": "Castle's up — four tower walls in mixed stone, banners flying, torches lit along the parapet. Left the murder holes for you.",
  "commands": [
    {
      "type": "createPart",
      "params": {
        "name": "CastleFloor",
        "position": { "x": 0, "y": 0, "z": 0 },
        "size": { "x": 40, "y": 1, "z": 40 },
        "material": "Slate",
        "color": { "r": 160, "g": 155, "b": 150 },
        "anchored": true
      }
    },
    {
      "type": "addLight",
      "params": {
        "name": "Beacon",
        "parent": "CastleKeep",
        "lightType": "PointLight",
        "brightness": 8,
        "range": 60,
        "color": { "r": 255, "g": 200, "b": 100 }
      }
    }
  ],
  "_pipeline": {
    "mode": "standard",
    "creative": true,
    "total_time_s": 12.4,
    "stage_times_s": { "intent": 1.2, "planning": 4.8, "commands": 3.1, "hermes": 3.3 },
    "intent_summary": "Build a large stone castle with towers"
  }
}
```

---

## CLI

```bash
# Standard 3-stage pipeline
python3 brain.py "build me a castle on the hill"

# Deep planning for complex builds
python3 brain.py --deep "build a floating city with waterfalls"

# Creative mode (adds Hermes personality wrapping)
python3 brain.py --creative "build a dragon temple"

# Deep + creative
python3 brain.py --deep --creative "an ancient observatory"

# Fast single-model fallback
python3 brain.py --fast "a small wooden house"

# Verbose pipeline progress to stderr
python3 brain.py --verbose "a medieval castle"

# Test model connectivity
python3 brain.py --test

# Pretty-printed JSON
python3 brain.py --pretty "a tower"
```

### API Key

The DeepInfra API key is loaded from `/home/eileen/mcp-deeinfra/.env`:

```bash
DEEPINFRA_API_KEY=sk-...
```

Falls back to `DEEPINFRA_API_KEY` environment variable.

---

## JSON Extraction

The `extract_json()` utility handles model outputs that may include:
- Raw JSON (ideal case)
- Markdown code fences (` ```json ... ``` `)
- JSON embedded in prose text
- Incomplete/truncated JSON

It tries direct parse first, then strips fences, then scans for brace/bracket matching at any depth.

---

## File Layout

```
brain.py     # Full pipeline implementation (~800 lines)
README.md    # This file
```

The entire pipeline is a single file — no package structure, no imports beyond stdlib. It communicates via stdin/stdout JSON and stderr progress.

---

## Integration

Called by `process_v2.py` as a subprocess:

```python
result = subprocess.run(
    ['python3', BRAIN_SCRIPT, '--verbose', enhanced_message],
    capture_output=True, text=True, timeout=120,
    cwd=os.path.dirname(BRAIN_SCRIPT)
)
parsed = json.loads(result.stdout)
```

The processor enhances the message with context layers before passing it to the brain:

```
[player_message]

[World Context: Nearby structures: CastleFloor, TowerBase. Players in world: 3.]
[Player Memory: Bond level: 7. Previous builds: castle, bridge, garden.]
[Skill Library: - Lighthouse Builder: Builds a striped lighthouse with beacon...]
```

---

## Related Repositories

| Repository | Role |
|-----------|------|
| [lucineer-worker](../lucineer-worker) | Processor daemon that invokes brain.py |
| [casting-call](../casting-call) | Model routing atlas (informs which models to use) |
| [lucineer-memory](../lucineer-memory) | Provides player context to the brain |
| [lucineer-vector](../lucineer-vector) | Provides skill matches to the brain |
| [lucineer-system](../lucineer-system) | Design docs for the pipeline architecture |

---

## License

MIT
