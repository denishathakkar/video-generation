"""Phase 1 story generator skeleton.

This module mirrors the structure of video_generator.py but for stories.
Phase 1 = scaffolding only:
- Accepts story idea + audience + length + art style + voice + vocab level
- Uses Claude to write a structured story script (real)
- Renders PLACEHOLDER illustration slides with PIL (no AI image gen yet)
- Synthesizes Edge TTS voiceover (real)
- Assembles MP4 (real)

Phases 2-6 will replace the placeholder illustrations with HF FLUX images,
add Ken Burns + transitions, tough-word overlays, recap slide, YouTube bundle.
"""

import os
import json
import random
import tempfile
import textwrap
from pathlib import Path
from typing import Callable

from anthropic import Anthropic
from PIL import Image, ImageDraw, ImageFont

from video_generator import (
    VIDEO_W,
    VIDEO_H,
    OUTPUTS_DIR,
    VOICES,
    DEFAULT_VOICE,
    synthesize_voice,
    build_clip,
    build_ken_burns_clip,
    _font,
    _draw_text_centered,
    _rounded_rect,
    _draw_sparkle,
    _draw_star,
)
from illustration import generate_illustration
from moviepy.editor import concatenate_videoclips, CompositeVideoClip

AUDIENCES = {
    "Kids 4-7": {"vocab": "very simple", "tone": "playful and warm"},
    "Kids 8-12": {"vocab": "clear, age-appropriate", "tone": "engaging and adventurous"},
    "Teens": {"vocab": "richer vocabulary", "tone": "vivid and dramatic"},
    "Adults": {"vocab": "literary", "tone": "evocative and thoughtful"},
}

LENGTHS = {
    "Short (5 scenes)": 5,
    "Standard (8 scenes)": 8,
    "Long (12 scenes)": 12,
}

ART_STYLES = {
    "Indian storybook": "traditional Indian storybook illustration, warm earth tones, intricate borders, folk art influence",
    "Watercolor": "soft watercolor painting, gentle washes, dreamlike, pastel palette",
    "3D cartoon": "3D animated cartoon style, Pixar-inspired, expressive characters, vibrant lighting",
    "Comic": "comic book illustration, bold outlines, dynamic poses, halftone shading",
}

VOCAB_LEVELS = {
    "Off": 0,
    "Light": 2,
    "Heavy": 5,
}

STORY_PALETTES = [
    {"bg": (255, 236, 200), "fg": (60, 30, 20), "accent": (192, 86, 33)},
    {"bg": (220, 235, 255), "fg": (20, 40, 80), "accent": (60, 90, 180)},
    {"bg": (235, 220, 255), "fg": (50, 20, 80), "accent": (130, 60, 180)},
    {"bg": (220, 250, 230), "fg": (20, 60, 30), "accent": (40, 130, 70)},
    {"bg": (255, 220, 220), "fg": (80, 20, 20), "accent": (200, 60, 60)},
]


def generate_story_script(
    idea: str,
    audience: str,
    num_scenes: int,
    art_style: str,
    vocab_level: str,
    anthropic_client: Anthropic,
) -> dict:
    aud = AUDIENCES.get(audience, AUDIENCES["Kids 8-12"])
    tough_word_count = VOCAB_LEVELS.get(vocab_level, 0)
    style_desc = ART_STYLES.get(art_style, ART_STYLES["Indian storybook"])

    system = (
        "You are a children's storyteller and educator. "
        "You return ONLY valid JSON, no markdown, no commentary."
    )
    user = f"""Write an illustrated story video script about: "{idea}"

Target audience: {audience} ({aud['tone']}, {aud['vocab']} vocabulary)
Number of scenes: {num_scenes}
Visual art style: {art_style} — {style_desc}
Tough English words to highlight per scene: {tough_word_count} (set 0 if vocab level is off)

Requirements:
- Story should have a clear narrative arc: setup → conflict → climax → resolution
- Each scene needs a vivid narration (2-4 sentences) AND a detailed image_prompt
- Image prompts must be richly visual: describe characters by NAME (so they can be looked up in character_cards), setting, mood, lighting, time of day, camera angle, gentle action
- IMPORTANT — image_prompt guidelines to avoid AI safety filter rejections:
  * NEVER use words like: weapon, strike, fight, battle, blood, kill, attack, violent, slay, decapitate, sword raised, blade, gore
  * Replace conflict/action with peaceful tableau wording: "standing protectively", "raising hand in blessing", "facing each other in tense moment", "guarding the entrance with calm strength"
  * For "battle" scenes, describe the moment BEFORE or AFTER the action, not during — e.g. "warriors standing tall, eyes meeting across a peaceful courtyard", "hero standing victorious as soft light breaks through clouds"
  * Keep tone wholesome, painterly, storybook — like an illustrated children's book, never realistic violence
  * Replace "trident raised to strike" with "trident held in noble pose", "demon defeated" with "evil banished, peace restored"
- For EVERY recurring character (people, gods, animals), add an entry to "character_cards" with detailed appearance (skin tone, hair, clothing, accessories, age, distinguishing features). Keep descriptions family-friendly — no scary weapons, no menacing features
- Use character names exactly as listed in character_cards when writing image_prompts (this enables consistent illustration across scenes)
- scene_type is one of: "establishing" (wide setting shot), "character" (close on a character), "action" (dynamic event), "emotional" (reaction/feeling), "resolution" (closing scene)
- For each scene, if vocab count > 0, identify {tough_word_count} tough English word(s) used in narration with a kid-friendly definition
- End with a clear moral or lesson
- Generate a thumbnail_prompt: the single most dramatic visual moment of the story (using the same gentle wording rules), optimized for a YouTube thumbnail (vibrant, eye-catching, simple composition)

Return ONLY this JSON shape:
{{
  "title": "Story title under 60 chars",
  "subtitle": "Optional one-line hook",
  "audience": "{audience}",
  "art_style": "{art_style}",
  "character_cards": {{
    "CharacterName": "Detailed visual description: age, skin/fur tone, hair, clothing, accessories, expression"
  }},
  "scenes": [
    {{
      "scene_number": 1,
      "scene_type": "establishing",
      "narration": "2-4 sentences spoken aloud by narrator",
      "image_prompt": "Detailed visual prompt referencing characters by name",
      "tough_words": [{{"word": "valiant", "meaning": "brave and courageous"}}]
    }}
  ],
  "moral": "One-line takeaway",
  "thumbnail_prompt": "Most dramatic moment for YouTube thumbnail, vibrant and eye-catching",
  "youtube": {{
    "title": "Catchy YouTube title under 60 chars (can differ from story title)",
    "description": "2-3 line description with what the story is about and who it's for",
    "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
  }}
}}

Provide exactly {num_scenes} scenes."""

    msg = anthropic_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4500,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def render_placeholder_illustration(scene: dict, palette: dict, save_path: Path) -> Path:
    """Phase 1: PIL placeholder. Phase 3 will replace with AI-generated illustration."""
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), color=palette["bg"])
    draw = ImageDraw.Draw(img)

    for _ in range(8):
        x = random.randint(0, VIDEO_W)
        y = random.randint(0, VIDEO_H)
        r = random.randint(60, 180)
        color = tuple(min(255, c + random.randint(-20, 20)) for c in palette["accent"])
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color + (40,) if len(color) == 3 else color)

    scene_num = scene.get("scene_number", "?")
    _rounded_rect(draw, (60, 50, 240, 130), 30, palette["accent"])
    label = f"Scene {scene_num}"
    lw = draw.textlength(label, font=_font(36))
    draw.text((150 - lw / 2, 75), label, font=_font(36), fill=(255, 255, 255))

    _rounded_rect(draw, (60, 180, VIDEO_W - 60, VIDEO_H - 180), 30, (255, 255, 255))
    _rounded_rect(draw, (90, 210, VIDEO_W - 90, VIDEO_H - 210), 20, palette["bg"])

    prompt_preview = scene.get("image_prompt", "")[:200]
    title_text = "Illustration"
    title_font = _font(40)
    tw = draw.textlength(title_text, font=title_font)
    star_r = 24
    gap = 14
    total_w = star_r * 2 + gap + tw
    star_cx = int((VIDEO_W - total_w) / 2 + star_r)
    star_cy = 262
    _draw_star(draw, (star_cx, star_cy), star_r, palette["accent"])
    draw.text((star_cx + star_r + gap, 240), title_text, font=title_font, fill=palette["accent"])

    body_font = _font(24)
    avg = body_font.getlength("M") or 14
    wrap_at = max(20, int((VIDEO_W - 220) / avg * 1.7))
    lines = textwrap.wrap(prompt_preview, width=wrap_at)[:8]
    line_h = 34
    y_start = 320
    for i, line in enumerate(lines):
        w = draw.textlength(line, font=body_font)
        draw.text(((VIDEO_W - w) / 2, y_start + i * line_h), line, font=body_font, fill=palette["fg"])

    footer = "(Placeholder — AI illustration coming in Phase 3)"
    fw = draw.textlength(footer, font=_font(18))
    draw.text(((VIDEO_W - fw) / 2, VIDEO_H - 240), footer, font=_font(18), fill=palette["accent"])

    img.save(save_path)
    return save_path


def render_title_slide(script: dict, palette: dict, save_path: Path) -> Path:
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), color=palette["bg"])
    draw = ImageDraw.Draw(img)
    _rounded_rect(draw, (60, 180, VIDEO_W - 60, 540), 40, palette["accent"])
    _rounded_rect(draw, (80, 200, VIDEO_W - 80, 520), 32, (255, 255, 255))
    _draw_text_centered(draw, script.get("title", "A Story"), 260, _font(64), palette["fg"])
    if script.get("subtitle"):
        _draw_text_centered(draw, script["subtitle"], 400, _font(32), palette["accent"])
    img.save(save_path)
    return save_path


def render_moral_slide(script: dict, palette: dict, save_path: Path) -> Path:
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), color=palette["bg"])
    draw = ImageDraw.Draw(img)
    _rounded_rect(draw, (60, 80, VIDEO_W - 60, 180), 30, palette["accent"])
    heading = "The Lesson"
    heading_font = _font(48)
    text_w = draw.textlength(heading, font=heading_font)
    sparkle_r = 26
    gap = 18
    total_w = sparkle_r * 2 + gap * 2 + text_w + sparkle_r * 2
    left_cx = int((VIDEO_W - total_w) / 2 + sparkle_r)
    text_x = left_cx + sparkle_r + gap
    right_cx = int(text_x + text_w + gap + sparkle_r)
    icon_cy = 130
    _draw_sparkle(draw, (left_cx, icon_cy), sparkle_r, (255, 255, 255))
    _draw_sparkle(draw, (right_cx, icon_cy), sparkle_r, (255, 255, 255))
    draw.text((text_x, 100), heading, font=heading_font, fill=(255, 255, 255))

    _rounded_rect(draw, (100, 240, VIDEO_W - 100, VIDEO_H - 120), 30, (255, 255, 255))
    moral = script.get("moral", "Every story teaches us something.")
    _draw_text_centered(draw, moral, 320, _font(38), palette["fg"], max_width=VIDEO_W - 240)
    img.save(save_path)
    return save_path


def generate_story_video(
    idea: str,
    audience: str = "Kids 8-12",
    length_label: str = "Standard (8 scenes)",
    art_style: str = "Indian storybook",
    vocab_level: str = "Light",
    voice: str = DEFAULT_VOICE,
    anthropic_key: str | None = None,
    hf_token: str | None = None,
    gemini_key: str | None = None,
    image_provider: str = "gemini",
    progress: Callable[[str, float], None] | None = None,
) -> Path:
    anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY is required")
    hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    gemini_key = gemini_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    client = Anthropic(api_key=anthropic_key)
    num_scenes = LENGTHS.get(length_label, 8)

    def step(msg: str, pct: float):
        if progress:
            progress(msg, pct)

    step("Writing story script with Claude...", 0.1)
    script = generate_story_script(idea, audience, num_scenes, art_style, vocab_level, client)

    workdir = Path(tempfile.mkdtemp(prefix="storygen_"))
    palette = random.choice(STORY_PALETTES)
    story_seed = random.randint(1, 2**31 - 1)
    character_cards = script.get("character_cards", {}) or {}
    clips = []

    step("Rendering title slide...", 0.15)
    title_img = workdir / "title.png"
    render_title_slide(script, palette, title_img)
    title_narration = f"{script.get('title', 'A story.')}. {script.get('subtitle', '')}".strip(". ")
    title_audio = workdir / "title.mp3"
    synthesize_voice(title_narration, title_audio, voice=voice)
    clips.append(build_clip(title_img, title_audio))

    scenes = script.get("scenes", [])
    total = max(1, len(scenes))
    for i, scene in enumerate(scenes):
        step(f"Illustrating scene {i + 1}/{total}...", 0.2 + 0.65 * (i / total))
        img_path = workdir / f"scene_{i:02d}.png"

        have_key_for_provider = (
            (image_provider.startswith("hf") and hf_token)
            or (image_provider.startswith("gemini") and gemini_key)
        )
        illustration_ok = False
        if have_key_for_provider:
            result = generate_illustration(
                image_prompt=scene.get("image_prompt", ""),
                art_style=art_style,
                save_path=img_path,
                character_cards=character_cards,
                seed=story_seed + i,
                hf_token=hf_token,
                gemini_key=gemini_key,
                provider=image_provider,
            )
            illustration_ok = result is not None

        if not illustration_ok:
            render_placeholder_illustration(scene, palette, img_path)

        aud_path = workdir / f"aud_{i:02d}.mp3"
        synthesize_voice(scene.get("narration", ""), aud_path, voice=voice)

        zoom_start, zoom_end = (1.0, 1.15) if i % 2 == 0 else (1.15, 1.0)
        pan_x = [0.0, 0.5, -0.5, 0.3, -0.3][i % 5]
        pan_y = [0.3, -0.3, 0.0, -0.4, 0.4][i % 5]
        clips.append(
            build_ken_burns_clip(
                img_path,
                aud_path,
                zoom_start=zoom_start,
                zoom_end=zoom_end,
                pan_x_pct=pan_x,
                pan_y_pct=pan_y,
            )
        )

    step("Rendering moral slide...", 0.9)
    moral_img = workdir / "moral.png"
    render_moral_slide(script, palette, moral_img)
    moral_audio = workdir / "moral.mp3"
    synthesize_voice(f"The lesson of this story: {script.get('moral', '')}", moral_audio, voice=voice)
    clips.append(build_clip(moral_img, moral_audio))

    step("Assembling final video...", 0.95)
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in script.get("title", "story"))[:60].strip()
    output_path = OUTPUTS_DIR / f"{safe_title or 'story'}.mp4"

    fade = 0.6
    faded = []
    for idx, c in enumerate(clips):
        cc = c.crossfadein(fade) if idx > 0 else c
        faded.append(cc)
    final = concatenate_videoclips(faded, method="compose", padding=-fade)
    final.write_videofile(
        str(output_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        verbose=False,
        logger=None,
    )
    step("Done!", 1.0)
    return output_path
