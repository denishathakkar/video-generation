"""Test harness: run the Claude story script + illustration pipeline without video assembly.

Usage:
    .venv/bin/python test_illustration.py "The story of Krishna and Kaliya snake"
    .venv/bin/python test_illustration.py "Story idea" --scenes 5 --style "Indian storybook"

Outputs:
    test_outputs/<slug>/scene_NN.png         (or .txt for failures)
    test_outputs/<slug>/script.json
    test_outputs/<slug>/report.txt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

from story_generator import generate_story_script
from illustration import generate_illustration, build_full_prompt


def slugify(text: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in text)[:60].strip()
    return safe.replace(" ", "_").lower() or "story"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idea", help="Story idea")
    ap.add_argument("--audience", default="Kids 8-12")
    ap.add_argument("--scenes", type=int, default=5)
    ap.add_argument("--style", default="Indian storybook")
    ap.add_argument("--vocab", default="Light")
    args = ap.parse_args()

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not anthropic_key:
        sys.exit("ANTHROPIC_API_KEY missing in .env")
    if not hf_token:
        sys.exit("HF_TOKEN missing in .env")

    out_root = Path(__file__).parent / "test_outputs"
    out_dir = out_root / slugify(args.idea)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    print("\n[1/3] Generating story script with Claude...")
    client = Anthropic(api_key=anthropic_key)
    t0 = time.time()
    script = generate_story_script(
        idea=args.idea,
        audience=args.audience,
        num_scenes=args.scenes,
        art_style=args.style,
        vocab_level=args.vocab,
        anthropic_client=client,
    )
    print(f"  ✓ Script generated in {time.time() - t0:.1f}s")
    (out_dir / "script.json").write_text(json.dumps(script, indent=2))

    character_cards = script.get("character_cards", {})
    scenes = script.get("scenes", [])
    print(f"  Title: {script.get('title')}")
    print(f"  Characters: {list(character_cards.keys())}")
    print(f"  Scenes: {len(scenes)}")

    print(f"\n[2/3] Illustrating {len(scenes)} scenes via HF FLUX...")
    seed_base = 42
    results = []
    for i, scene in enumerate(scenes, start=1):
        img_path = out_dir / f"scene_{i:02d}.png"
        prompt = scene.get("image_prompt", "")
        full_prompt = build_full_prompt(prompt, args.style, character_cards)
        scene_type = scene.get("scene_type", "?")

        t0 = time.time()
        result = generate_illustration(
            image_prompt=prompt,
            art_style=args.style,
            save_path=img_path,
            character_cards=character_cards,
            seed=seed_base + i,
            hf_token=hf_token,
        )
        elapsed = time.time() - t0

        if result is not None and img_path.exists() and img_path.stat().st_size > 50_000:
            size_kb = img_path.stat().st_size // 1024
            status = "✅"
            note = f"{size_kb} KB"
        else:
            status = "❌"
            note = "FAILED (likely safety filter)"
            (out_dir / f"scene_{i:02d}.FAILED.txt").write_text(
                f"Image prompt:\n{prompt}\n\n"
                f"Full prompt sent to FLUX:\n{full_prompt}\n"
            )

        line = f"[{i}/{len(scenes)}] {status} Scene {i} ({scene_type}) — {note}  ({elapsed:.1f}s)"
        print(line)
        if status == "❌":
            print(f"        Prompt: {prompt[:150]}{'...' if len(prompt) > 150 else ''}")
        results.append({
            "scene": i,
            "type": scene_type,
            "ok": status == "✅",
            "note": note,
            "elapsed_s": round(elapsed, 1),
            "prompt": prompt,
        })

    print("\n[3/3] Report:")
    ok = sum(1 for r in results if r["ok"])
    print(f"  {ok}/{len(results)} succeeded")
    print(f"  Files at: {out_dir}")
    (out_dir / "report.txt").write_text(
        f"Idea: {args.idea}\n"
        f"Title: {script.get('title')}\n"
        f"Succeeded: {ok}/{len(results)}\n\n"
        + "\n".join(
            f"Scene {r['scene']} ({r['type']}): {'OK' if r['ok'] else 'FAIL'} — {r['note']} ({r['elapsed_s']}s)\n"
            f"  Prompt: {r['prompt']}\n"
            for r in results
        )
    )


if __name__ == "__main__":
    main()
