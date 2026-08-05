# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-05

### Added
- 4-stage multi-model intelligence pipeline for natural language to Roblox build commands
- Stage 1: Intent parsing (Seed-2.0-mini)
- Stage 2: Spatial reasoning (Qwen3.6 / Seed-2.0-pro)
- Stage 3: Build command generation (Qwen3-Coder-480B)
- Stage 4: Personality wrapping (Hermes-3-Llama-405B)
- JSON output matching Lucineer's CommandExecutor schema
- 89 tests covering all pipeline stages
- MIT LICENSE
- This CHANGELOG.md
