"""Compare illustration providers side-by-side on the same prompts.

Usage:
    .venv/bin/python test_compare.py                           # uses default sample prompts
    .venv/bin/python test_compare.py --idea "Krishna and Kaliya"  # uses Claude to generate scenes
    .venv/bin/python test_compare.py --providers hf-sd,gemini  # subset
    .venv/bin/python test_compare.py --scenes 3

Outputs (per provider):
    test_outputs/compare/<provider>/scene_NN.png
Plus a combined HTML report at:
    test_outputs/compare/report.html
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from illustration import generate_illustration, PROVIDERS

SAMPLE_PROMPTS = [
    {
        "image_prompt": "A young blue-skinned boy with a peacock feather in his hair, sitting peacefully on a tree branch above a calm river at sunrise, painterly storybook style",
        "scene_type": "character",
    },
    {
        "image_prompt": "A wise elephant-headed deity sitting cross-legged in a lotus pose, surrounded by glowing diyas at dusk, ornate temple background",
        "scene_type": "character",
    },
    {
        "image_prompt": "Wide establishing shot of an ancient Indian village by a river, golden hour, cows grazing, women carrying clay pots, mango trees, vibrant warm colors",
        "scene_type": "establishing",
    },
    {
        "image_prompt": "A graceful goddess with multiple arms dancing peacefully on a lotus flower, golden divine light, surrounded by colorful flowers, painterly children's book illustration",
        "scene_type": "action",
    },
    {
        "image_prompt": "A monkey warrior with orange fur and golden ornaments standing on a mountaintop at dawn, holding a glowing mace, calm dignified pose, soft pastel sky",
        "scene_type": "emotional",
    },
]


def maybe_generate_via_claude(idea, num_scenes, style):
    from anthropic import Anthropic
    from story_generator import generate_story_script
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY missing")
    client = Anthropic(api_key=key)
    print(f"Asking Claude for {num_scenes} scenes about: {idea}")
    script = generate_story_script(idea, "Kids 8-12", num_scenes, style, "Light", client)
    scenes = []
    for s in script.get("scenes", [])[:num_scenes]:
        scenes.append({"image_prompt": s.get("image_prompt", ""), "scene_type": s.get("scene_type", "?")})
    return scenes, script.get("character_cards", {})


def html_cell(rel_path, abs_path, label, elapsed, size_kb):
    if abs_path and Path(abs_path).exists() and Path(abs_path).stat().st_size > 50_000:
        return f'<td><div style="font-size:12px;color:#555">{label} — {elapsed:.1f}s, {size_kb}KB</div><img src="{rel_path}" style="width:480px;max-width:100%;border:1px solid #ddd"/></td>'
    return f'<td><div style="font-size:12px;color:#a00">{label} — FAILED</div><div style="width:480px;height:270px;background:#fee;display:flex;align-items:center;justify-content:center;color:#a00">No image</div></td>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", default=None, help="Story idea (uses Claude). If omitted, uses canned sample prompts.")
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--style", default="Indian storybook")
    ap.add_argument("--providers", default="hf-sd,gemini,hf-flux", help="Comma-separated provider list")
    args = ap.parse_args()

    out_root = Path(__file__).parent / "test_outputs" / "compare"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.idea:
        scenes, character_cards = maybe_generate_via_claude(args.idea, args.scenes, args.style)
    else:
        scenes = SAMPLE_PROMPTS[: args.scenes]
        character_cards = {}

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    print(f"Comparing providers: {providers}")
    print(f"Scenes to render: {len(scenes)}")

    results = {p: [] for p in providers}
    for p in providers:
        prov_dir = out_root / p
        prov_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Provider: {p} — {PROVIDERS.get(p, p)}")
        for i, scene in enumerate(scenes, start=1):
            out_path = prov_dir / f"scene_{i:02d}.png"
            t0 = time.time()
            res = generate_illustration(
                image_prompt=scene["image_prompt"],
                art_style=args.style,
                save_path=out_path,
                character_cards=character_cards,
                seed=42 + i,
                provider=p,
            )
            elapsed = time.time() - t0
            ok = res is not None and out_path.exists() and out_path.stat().st_size > 50_000
            size_kb = out_path.stat().st_size // 1024 if out_path.exists() else 0
            status = "✅" if ok else "❌"
            print(f"  [{i}/{len(scenes)}] {status} scene_{i:02d}.png ({elapsed:.1f}s, {size_kb}KB)")
            results[p].append({
                "rel_path": str(out_path.relative_to(out_root)) if ok else None,
                "abs_path": str(out_path) if ok else None,
                "elapsed": elapsed, "size_kb": size_kb, "ok": ok,
            })

    html = ["<html><head><meta charset='utf-8'><title>Provider comparison</title></head><body style='font-family:sans-serif;background:#fafafa'>"]
    html.append("<h1>Illustration Provider Comparison</h1>")
    html.append(f"<p>Style: {args.style} | Scenes: {len(scenes)}</p>")
    html.append("<table style='border-collapse:collapse;background:white'><tr><th style='padding:8px'>Scene</th>")
    for p in providers:
        html.append(f"<th style='padding:8px'>{p}<br><span style='font-size:10px;color:#888'>{PROVIDERS.get(p, '')}</span></th>")
    html.append("</tr>")
    for i, scene in enumerate(scenes, start=1):
        html.append("<tr>")
        html.append(f"<td style='padding:8px;vertical-align:top;max-width:300px;font-size:12px'><b>Scene {i}</b> ({scene.get('scene_type','?')})<br>{scene['image_prompt'][:200]}...</td>")
        for p in providers:
            r = results[p][i - 1]
            html.append(html_cell(r["rel_path"], r["abs_path"], p, r["elapsed"], r["size_kb"]))
        html.append("</tr>")
    html.append("</table></body></html>")
    report_path = out_root / "report.html"
    report_path.write_text("\n".join(html))
    print(f"\nReport written to: {report_path}")
    print(f"Open it in your browser: file://{report_path}")


if __name__ == "__main__":
    main()
