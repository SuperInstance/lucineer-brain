#!/usr/bin/env python3
"""
Playtest Bot for Lucineer Worker Relay
======================================
Sends 20 diverse build requests through the Worker relay endpoint and logs
quality scores, response times, and failure modes.

Usage:
    python3 playtest_bot.py                          # Default: hits live worker
    python3 playtest_bot.py --url http://localhost:8787  # Local wrangler dev
    python3 playtest_bot.py --verbose                # Full output
"""

import argparse
import json
import time
import urllib.request
import urllib.error
import urllib.parse
import sys
from datetime import datetime

# ─── Test Prompts ────────────────────────────────────────────────────────────

TEST_PROMPTS = [
    # ── Basic builds (should hit fast-path templates) ──
    {"message": "build me a tower", "expect": "template", "category": "basic"},
    {"message": "make a small house", "expect": "template", "category": "basic"},
    {"message": "build a castle", "expect": "template", "category": "basic"},
    {"message": "build a bridge", "expect": "template", "category": "basic"},
    {"message": "build a windmill", "expect": "template", "category": "basic"},

    # ── Creative/complex builds (deep path) ──
    {"message": "build a cozy fishing cottage by the water", "expect": "deep", "category": "creative"},
    {"message": "create an abandoned lighthouse on a rocky cliff", "expect": "deep", "category": "creative"},
    {"message": "build a mysterious garden with strange flowers", "expect": "deep", "category": "creative"},
    {"message": "make an ancient stone temple covered in vines", "expect": "deep", "category": "creative"},
    {"message": "build a wooden dock extending into foggy water", "expect": "deep", "category": "creative"},

    # ── Emotional builds ──
    {"message": "I'm scared, build me somewhere safe", "expect": "emotional", "category": "emotional"},
    {"message": "I feel lonely, build something", "expect": "emotional", "category": "emotional"},
    {"message": "I'm sad, just build something small", "expect": "emotional", "category": "emotional"},
    {"message": "I'm so happy! Build me a celebration tower", "expect": "emotional", "category": "emotional"},

    # ── Edge cases ──
    {"message": "build nothing", "expect": "any", "category": "edge"},
    {"message": "build a structure with exactly 0 parts", "expect": "any", "category": "edge"},
    {"message": "build a portal to another dimension", "expect": "any", "category": "edge"},
    {"message": "", "expect": "error", "category": "edge"},

    # ── Iterative builds ──
    {"message": "make the last build taller", "expect": "any", "category": "iterative"},
    {"message": "add a roof to it", "expect": "any", "category": "iterative"},
]

# ─── Quality Scoring ──────────────────────────────────────────────────────────

def score_response(result: dict, elapsed: float, prompt: dict) -> dict:
    """
    Score a response on a 0-10 scale based on:
    - Did it succeed? (0 if error/timeout)
    - Did it return a reply with content?
    - Did it return build commands?
    - How fast was it? (< 2s = fast-path bonus)
    - Was it emotionally aware? (for emotional prompts)
    """
    score = 0
    notes = []

    # Error or queued
    if result.get("status") == "error" or "error" in result:
        return {"score": 0, "notes": f"Error: {result.get('error', 'unknown')}", "elapsed": elapsed}
    if result.get("status") == "queued":
        return {"score": 1, "notes": f"Queued at position {result.get('position', '?')}", "elapsed": elapsed}

    # Got a reply
    reply = result.get("reply", "")
    if reply and len(reply) > 5:
        score += 2
        notes.append("has_reply")
    else:
        notes.append("empty_reply")
        return {"score": max(score, 1), "notes": ", ".join(notes), "elapsed": elapsed}

    # Got commands
    commands = result.get("commands", [])
    if commands and len(commands) > 0:
        score += 2
        notes.append(f"{len(commands)}_commands")
    else:
        notes.append("no_commands")

    # Speed bonus
    if elapsed < 2.0:
        score += 3
        notes.append("instant")
    elif elapsed < 10.0:
        score += 2
        notes.append("fast")
    elif elapsed < 30.0:
        score += 1
        notes.append("acceptable")
    else:
        notes.append("slow")

    # Source quality
    source = result.get("source", "unknown")
    if source == "template":
        score += 1
        notes.append("template")
    elif source == "deep_cache":
        score += 1
        notes.append("cached")
    elif source == "ai" or source == "deep":
        score += 2  # Deep path is more creative
        notes.append("ai_generated")

    # Emotional awareness
    if prompt["category"] == "emotional":
        reply_lower = reply.lower()
        emotion_words = {
            "scared": ["safe", "hear", "steady", "solid", "shelter", "walls"],
            "lonely": ["here", "nearby", "bench", "dock", "presence", "waiting"],
            "sad": ["quiet", "gentle", "small", "careful", "sit"],
            "happy": ["celebration", "bright", "flag", "festival", "flourish"],
        }
        emotion = None
        for e in emotion_words:
            if e in prompt["message"].lower():
                emotion = e
                break
        if emotion and emotion in emotion_words:
            if any(w in reply_lower for w in emotion_words[emotion]):
                score += 1
                notes.append("emotion_aware")
            else:
                notes.append("emotion_ignored")

    # Cached bonus
    if result.get("_cached"):
        notes.append("cache_hit")

    return {"score": min(score, 10), "notes": ", ".join(notes), "elapsed": elapsed}


# ─── Bot Logic ────────────────────────────────────────────────────────────────

def send_request(base_url: str, prompt: dict, player_name: str, session_id: str) -> tuple[dict, float]:
    """Send a build request and return (response_dict, elapsed_seconds).
    Tries POST /api/message first, falls back to GET /api/quick/ for
    template matching if the POST path is blocked."""
    payload = {
        "sessionId": session_id,
        "playerName": player_name,
        "message": prompt["message"],
    }

    data = json.dumps(payload).encode("utf-8")
    url = f"{base_url}/api/message"
    t0 = time.time()

    try:
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        elapsed = time.time() - t0
        return result, elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        if e.code == 403:
            # Cloudflare blocking POST — try GET quick-path as fallback
            try:
                quick_url = f"{base_url}/api/quick/{urllib.parse.quote(prompt['message'])}"
                t_quick = time.time()
                req2 = urllib.request.Request(quick_url)
                with urllib.request.urlopen(req2, timeout=15) as resp2:
                    result2 = json.loads(resp2.read())
                elapsed2 = time.time() - t_quick
                if result2.get("status") == "complete":
                    return result2, elapsed2
                # Not a template match — return the original 403
            except Exception:
                pass
        body = e.read().decode("utf-8", errors="replace")[:200]
        return {"error": f"HTTP {e.code}: {body}", "status": "error"}, elapsed
    except urllib.error.URLError as e:
        elapsed = time.time() - t0
        return {"error": f"URL Error: {e}", "status": "error"}, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return {"error": f"Exception: {e}", "status": "error"}, elapsed


def check_health(base_url: str) -> dict | None:
    """Check the health endpoint."""
    try:
        url = f"{base_url}/api/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  ⚠ Health check failed: {e}", file=sys.stderr)
        return None


def run_playtest(base_url: str, verbose: bool = False) -> dict:
    """Run the full playtest suite. Returns summary report."""
    print(f"\n{'='*70}")
    print(f"  LUCINEER PLAYTEST BOT")
    print(f"  Target: {base_url}")
    print(f"  Time:   {datetime.now().isoformat()}")
    print(f"  Prompts: {len(TEST_PROMPTS)}")
    print(f"{'='*70}\n")

    # Health check first
    health = check_health(base_url)
    if health:
        print(f"  ✓ Health: {health.get('status', '?')}")
        if 'queue' in health:
            print(f"    Queue: {health['queue']}")
        if 'cache' in health:
            print(f"    Cache: {health['cache']}")
    else:
        print(f"  ⚠ Health endpoint unreachable — continuing anyway")
    print()

    results = []
    session_id = f"playtest_bot_{int(time.time())}"

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        player_name = f"PlaytestBot_{i}"
        category = prompt["category"]
        message_preview = prompt["message"][:50] + ("..." if len(prompt["message"]) > 50 else "")
        print(f"  [{i:2d}/{len(TEST_PROMPTS)}] ({category:10s}) \"{message_preview}\"", end="", flush=True)

        result, elapsed = send_request(base_url, prompt, player_name, session_id)
        scored = score_response(result, elapsed, prompt)
        results.append({**prompt, "result": result, "scored": scored})

        score_str = f"{scored['score']}/10"
        time_str = f"{elapsed:.1f}s"
        status = result.get("status", result.get("error", "?")[:30])

        if scored["score"] >= 7:
            print(f" → {score_str} in {time_str} ✓")
        elif scored["score"] >= 4:
            print(f" → {score_str} in {time_str} ~")
        else:
            print(f" → {score_str} in {time_str} ✗")

        if verbose and scored["score"] < 7:
            reply_preview = (result.get("reply", "") or "")[:80]
            print(f"         notes: {scored['notes']}")
            if reply_preview:
                print(f"         reply: \"{reply_preview}\"")

        # Brief delay between requests to be polite
        time.sleep(0.5)

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}\n")

    scores = [r["scored"]["score"] for r in results]
    times = [r["scored"]["elapsed"] for r in results]

    avg_score = sum(scores) / len(scores)
    avg_time = sum(times) / len(times)
    max_score = max(scores)
    min_score = min(scores)
    success_count = sum(1 for s in scores if s >= 4)
    fail_count = sum(1 for s in scores if s <= 1)

    print(f"  Total prompts:     {len(results)}")
    print(f"  Average score:     {avg_score:.1f}/10")
    print(f"  Score range:       {min_score}–{max_score}")
    print(f"  Successes (≥4):    {success_count}/{len(results)} ({100*success_count/len(results):.0f}%)")
    print(f"  Failures (≤1):     {fail_count}/{len(results)} ({100*fail_count/len(results):.0f}%)")
    print(f"  Average response:  {avg_time:.1f}s")
    print(f"  Fastest response:  {min(times):.1f}s")
    print(f"  Slowest response:  {max(times):.1f}s")

    # By category
    print(f"\n  By Category:")
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["scored"]["score"])

    for cat, cat_scores in sorted(categories.items()):
        cat_avg = sum(cat_scores) / len(cat_scores)
        print(f"    {cat:12s}: avg {cat_avg:.1f}/10 ({len(cat_scores)} prompts)")

    # Failure modes
    failures = [r for r in results if r["scored"]["score"] <= 1]
    if failures:
        print(f"\n  Failure Modes:")
        for f in failures:
            print(f"    [{f['category']}] \"{f['message'][:40]}\" → {f['scored']['notes']}")

    # Fastest and slowest
    fastest = min(results, key=lambda r: r["scored"]["elapsed"])
    slowest = max(results, key=lambda r: r["scored"]["elapsed"])
    print(f"\n  Fastest: \"{fastest['message'][:40]}\" in {fastest['scored']['elapsed']:.1f}s ({fastest['scored']['score']}/10)")
    print(f"  Slowest: \"{slowest['message'][:40]}\" in {slowest['scored']['elapsed']:.1f}s ({slowest['scored']['score']}/10)")

    print(f"\n{'='*70}\n")

    # ─── JSON report ──────────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "target": base_url,
        "health": health,
        "summary": {
            "total": len(results),
            "avg_score": round(avg_score, 1),
            "avg_time_s": round(avg_time, 1),
            "success_rate": round(100 * success_count / len(results), 0),
            "failure_rate": round(100 * fail_count / len(results), 0),
        },
        "categories": {cat: {"avg_score": round(sum(s) / len(s), 1), "count": len(s)}
                       for cat, s in categories.items()},
        "failures": [{"message": f["message"], "notes": f["scored"]["notes"]}
                     for f in failures],
        "results": [{k: v for k, v in r.items() if k != "result"} for r in results],
    }

    return report


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lucineer Playtest Bot")
    parser.add_argument(
        "--url",
        default="https://lucineer-relay.casey-digennaro.workers.dev",
        help="Worker relay base URL",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full reply text for low-scoring responses",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save JSON report to this file",
    )
    args = parser.parse_args()

    report = run_playtest(args.url, verbose=args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {args.output}\n")
    else:
        # Save to default location
        output_path = f"/home/eileen/projects/lucineer-brain/playtest_results_{int(time.time())}.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to: {output_path}\n")


if __name__ == "__main__":
    main()
