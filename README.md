# 🧮 Lucineer Brain

Multi-model build intelligence that routes natural language through DeepInfra models to generate Roblox build commands.

## Model Routing

| Step | Model | Purpose |
|------|-------|---------|
| 1 | ByteDance/Seed-2.0-mini | Fast intent parsing |
| 2 | Qwen/Qwen3.6-35B-A3B | Spatial planning |
| 3 | Qwen/Qwen3-Coder-480B | Build command generation |

## Usage

```bash
python3 brain.py "build me a castle on the hill"
```

Outputs JSON: `{"reply": "...", "commands": [...]}`

Part of the [Lucineer system](https://github.com/SuperInstance/lucineer-system).
