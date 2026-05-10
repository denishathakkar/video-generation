import os
import json
import tempfile
import textwrap
import random
from pathlib import Path
from typing import Callable

import asyncio

import edge_tts
import numpy as np
from anthropic import Anthropic
from gtts import gTTS

VOICES = {
    "US Female (Aria - warm)": "en-US-AriaNeural",
    "US Female (Jenny - friendly teacher)": "en-US-JennyNeural",
    "US Male (Guy - clear)": "en-US-GuyNeural",
    "UK Female (Sonia)": "en-GB-SoniaNeural",
    "UK Male (Ryan)": "en-GB-RyanNeural",
    "Indian Female (Neerja)": "en-IN-NeerjaNeural",
    "Indian Male (Prabhat)": "en-IN-PrabhatNeural",
    "Australian Female (Natasha)": "en-AU-NatashaNeural",
}
DEFAULT_VOICE = "en-US-AriaNeural"

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    concatenate_videoclips,
)
from PIL import Image, ImageDraw, ImageFont

VIDEO_W, VIDEO_H = 1280, 720
OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

PALETTES = [
    {"bg": (255, 230, 109), "fg": (40, 40, 80), "accent": (255, 107, 107), "good": (76, 175, 80)},
    {"bg": (255, 154, 158), "fg": (40, 20, 60), "accent": (102, 51, 153), "good": (46, 204, 113)},
    {"bg": (161, 196, 253), "fg": (30, 30, 80), "accent": (255, 87, 87), "good": (39, 174, 96)},
    {"bg": (255, 195, 113), "fg": (60, 30, 20), "accent": (192, 57, 43), "good": (39, 174, 96)},
    {"bg": (185, 251, 192), "fg": (20, 60, 40), "accent": (231, 76, 60), "good": (39, 174, 96)},
    {"bg": (215, 174, 251), "fg": (40, 20, 70), "accent": (255, 121, 63), "good": (39, 174, 96)},
]

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
    "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_text_centered(draw, text, y, font, fill, max_width=VIDEO_W - 120):
    avg_char = font.getlength("M") or 30
    wrap_at = max(10, int(max_width / avg_char * 1.6))
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=wrap_at) or [""])
    line_h = font.size + 12
    for i, line in enumerate(lines):
        w = draw.textlength(line, font=font)
        draw.text(((VIDEO_W - w) / 2, y + i * line_h), line, font=font, fill=fill)
    return y + len(lines) * line_h


def _rounded_rect(draw, xy, radius, fill, outline=None, width=0):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _draw_star(draw, center, radius, fill, points=5):
    import math
    cx, cy = center
    pts = []
    for i in range(points * 2):
        r = radius if i % 2 == 0 else radius * 0.45
        angle = math.pi / 2 + i * math.pi / points
        pts.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
    draw.polygon(pts, fill=fill)


def _draw_sparkle(draw, center, radius, fill):
    cx, cy = center
    pts = [
        (cx, cy - radius),
        (cx + radius * 0.25, cy - radius * 0.25),
        (cx + radius, cy),
        (cx + radius * 0.25, cy + radius * 0.25),
        (cx, cy + radius),
        (cx - radius * 0.25, cy + radius * 0.25),
        (cx - radius, cy),
        (cx - radius * 0.25, cy - radius * 0.25),
    ]
    draw.polygon(pts, fill=fill)


def _draw_checkmark(draw, center, size, color, thickness=8):
    cx, cy = center
    s = size
    draw.line([(cx - s * 0.5, cy + s * 0.05), (cx - s * 0.1, cy + s * 0.45)], fill=color, width=thickness)
    draw.line([(cx - s * 0.1, cy + s * 0.45), (cx + s * 0.55, cy - s * 0.4)], fill=color, width=thickness)


def _draw_question_mark(draw, center, size, color, font_size=None):
    cx, cy = center
    fs = font_size or int(size * 1.4)
    f = _font(fs)
    w = draw.textlength("?", font=f)
    draw.text((cx - w / 2, cy - fs * 0.6), "?", font=f, fill=color)


def _draw_thinking_bubble(draw, center, radius, fill):
    cx, cy = center
    draw.ellipse((cx - radius, cy - radius * 0.6, cx + radius, cy + radius * 0.6), fill=fill)
    small_r = radius * 0.25
    draw.ellipse(
        (cx - radius - small_r * 1.8, cy + radius * 0.5, cx - radius - small_r * 0.4, cy + radius * 0.5 + small_r * 1.4),
        fill=fill,
    )
    draw.ellipse(
        (cx - radius - small_r * 2.8, cy + radius * 1.1, cx - radius - small_r * 2.0, cy + radius * 1.1 + small_r * 0.8),
        fill=fill,
    )


def _draw_party_burst(draw, center, radius, colors):
    import math
    cx, cy = center
    for i in range(10):
        angle = i * math.pi / 5
        x1 = cx + math.cos(angle) * radius * 0.45
        y1 = cy + math.sin(angle) * radius * 0.45
        x2 = cx + math.cos(angle) * radius
        y2 = cy + math.sin(angle) * radius
        col = colors[i % len(colors)]
        draw.line([(x1, y1), (x2, y2)], fill=col, width=6)
    draw.ellipse((cx - radius * 0.35, cy - radius * 0.35, cx + radius * 0.35, cy + radius * 0.35), fill=colors[0])


def _draw_confetti(draw, palette, n=40, seed=0):
    rng = random.Random(seed)
    colors = [palette["accent"], palette["good"], (255, 255, 255), palette["fg"]]
    for _ in range(n):
        x = rng.randint(0, VIDEO_W)
        y = rng.randint(0, VIDEO_H)
        r = rng.randint(6, 14)
        c = rng.choice(colors)
        if rng.random() < 0.5:
            draw.ellipse((x, y, x + r, y + r), fill=c)
        else:
            draw.rectangle((x, y, x + r, y + r), fill=c)


def render_title(scene, palette, save_path) -> Path:
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), color=palette["bg"])
    draw = ImageDraw.Draw(img)
    _draw_confetti(draw, palette, n=60, seed=1)
    _rounded_rect(draw, (80, 200, VIDEO_W - 80, 540), 40, palette["accent"])
    _rounded_rect(draw, (96, 216, VIDEO_W - 96, 524), 32, (255, 255, 255))
    _draw_text_centered(draw, scene["text"], 290, _font(72), palette["fg"])
    if scene.get("subtitle"):
        _draw_text_centered(draw, scene["subtitle"], 430, _font(40), palette["accent"])
    img.save(save_path)
    return save_path


def render_lesson(scene, palette, save_path) -> Path:
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), color=palette["bg"])
    draw = ImageDraw.Draw(img)
    _rounded_rect(draw, (60, 40, VIDEO_W - 60, 140), 30, palette["accent"])
    heading = scene.get("heading", "Lesson")
    w = draw.textlength(heading, font=_font(48))
    draw.text(((VIDEO_W - w) / 2, 60), heading, font=_font(48), fill=(255, 255, 255))

    bullets = scene.get("bullets", [])[:5]
    n = max(1, len(bullets))
    available_h = VIDEO_H - 180
    row_gap = 14
    row_h = max(80, min(140, (available_h - row_gap * (n - 1)) // n))
    bullet_font_size = 28 if row_h < 110 else 30
    bf = _font(bullet_font_size)
    line_h = bullet_font_size + 8
    max_lines = max(1, (row_h - 24) // line_h)
    text_x = 170
    text_box_w = VIDEO_W - text_x - 70

    y = 165
    for i, b in enumerate(bullets):
        _rounded_rect(draw, (90, y, VIDEO_W - 90, y + row_h), 20, (255, 255, 255))
        bullet_color = [palette["accent"], palette["good"], (255, 152, 0), (33, 150, 243), (156, 39, 176)][i % 5]
        cx, cy = 125, y + row_h // 2
        draw.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=bullet_color)

        avg = bf.getlength("M") or 16
        wrap_at = max(10, int(text_box_w / avg * 1.7))
        lines = textwrap.wrap(b, width=wrap_at)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = textwrap.shorten(lines[-1] + " " + " ".join(textwrap.wrap(b, width=wrap_at)[max_lines:]), width=wrap_at, placeholder="...")
        text_block_h = len(lines) * line_h
        ty = y + (row_h - text_block_h) // 2
        for j, line in enumerate(lines):
            draw.text((text_x, ty + j * line_h), line, font=bf, fill=palette["fg"])
        y += row_h + row_gap
    img.save(save_path)
    return save_path


def render_question(scene, palette, save_path, reveal=False) -> Path:
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), color=palette["bg"])
    draw = ImageDraw.Draw(img)

    _rounded_rect(draw, (60, 40, VIDEO_W - 60, 130), 25, palette["accent"])
    qnum = scene.get("question_number", "Question")
    label = qnum if not reveal else "Answer"
    label_font = _font(46)
    text_w = draw.textlength(label, font=label_font)
    icon_size = 50
    gap = 18
    total_w = icon_size + gap + text_w
    icon_cx = int((VIDEO_W - total_w) / 2 + icon_size / 2)
    icon_cy = 60 + 23
    text_x = icon_cx + icon_size / 2 + gap
    if reveal:
        draw.ellipse(
            (icon_cx - icon_size / 2, icon_cy - icon_size / 2, icon_cx + icon_size / 2, icon_cy + icon_size / 2),
            fill=(255, 255, 255),
        )
        _draw_checkmark(draw, (icon_cx, icon_cy), icon_size * 0.7, palette["good"], thickness=8)
    else:
        draw.ellipse(
            (icon_cx - icon_size / 2, icon_cy - icon_size / 2, icon_cx + icon_size / 2, icon_cy + icon_size / 2),
            fill=(255, 255, 255),
        )
        _draw_question_mark(draw, (icon_cx, icon_cy), icon_size * 0.7, palette["accent"], font_size=44)
    draw.text((text_x, 60), label, font=label_font, fill=(255, 255, 255))

    sentence = scene["sentence"]
    correct = scene.get("correct_option", "")
    correct_text = ""
    options = scene.get("options", [])
    if reveal and correct and options:
        idx = "ABCD".find(correct.upper())
        if 0 <= idx < len(options):
            correct_text = options[idx]
        display = sentence.replace("___", f"[{correct_text}]")
    else:
        display = sentence

    _rounded_rect(draw, (80, 160, VIDEO_W - 80, 320), 25, (255, 255, 255))
    _draw_text_centered(draw, display, 195, _font(42), palette["fg"], max_width=VIDEO_W - 200)

    letters = ["A", "B", "C", "D"]
    box_w = (VIDEO_W - 240) // 2
    box_h = 110
    start_x = 120
    start_y = 360
    gap = 30
    for i, opt in enumerate(options[:4]):
        col = i % 2
        row = i // 2
        x = start_x + col * (box_w + gap)
        y = start_y + row * (box_h + gap)
        is_correct = reveal and letters[i] == correct.upper()
        fill_color = palette["good"] if is_correct else (255, 255, 255)
        text_color = (255, 255, 255) if is_correct else palette["fg"]
        _rounded_rect(draw, (x, y, x + box_w, y + box_h), 20, fill_color, outline=palette["accent"], width=4)
        letter_color = (255, 255, 255) if is_correct else palette["accent"]
        draw.ellipse((x + 18, y + 30, x + 68, y + 80), fill=letter_color)
        lw = draw.textlength(letters[i], font=_font(34))
        lc = (palette["accent"], (255, 255, 255))[is_correct]
        draw.text((x + 18 + (50 - lw) / 2, y + 34), letters[i], font=_font(34), fill=lc)
        opt_font = _font(30)
        avg = opt_font.getlength("M") or 18
        wrap_at = max(8, int((box_w - 100) / avg * 1.6))
        lines = textwrap.wrap(opt, width=wrap_at)[:2]
        for j, line in enumerate(lines):
            draw.text((x + 90, y + 30 + j * 36), line, font=opt_font, fill=text_color)

    if reveal and scene.get("explanation"):
        _rounded_rect(draw, (60, 600, VIDEO_W - 60, 690), 20, palette["good"])
        exp_font = _font(26)
        avg = exp_font.getlength("M") or 16
        wrap_at = int((VIDEO_W - 160) / avg * 1.6)
        line = textwrap.shorten(scene["explanation"], width=wrap_at * 2, placeholder="...")
        w = draw.textlength(line, font=exp_font)
        draw.text(((VIDEO_W - w) / 2, 626), line, font=exp_font, fill=(255, 255, 255))

    img.save(save_path)
    return save_path


def render_think(scene, palette, save_path, number: int) -> Path:
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), color=palette["bg"])
    draw = ImageDraw.Draw(img)
    _draw_confetti(draw, palette, n=30, seed=10 + number)
    label = "Think!"
    label_font = _font(80)
    text_w = draw.textlength(label, font=label_font)
    bubble_r = 50
    gap = 25
    total_w = bubble_r * 2 + gap + text_w
    bubble_cx = int((VIDEO_W - total_w) / 2 + bubble_r)
    bubble_cy = 165
    _draw_thinking_bubble(draw, (bubble_cx, bubble_cy), bubble_r, palette["accent"])
    draw.text((bubble_cx + bubble_r + gap, 120), label, font=label_font, fill=palette["fg"])
    big = str(number)
    bf = _font(280)
    bw = draw.textlength(big, font=bf)
    cx = (VIDEO_W - bw) / 2 + bw / 2
    cy = 460
    draw.ellipse((cx - 150, cy - 150, cx + 150, cy + 150), fill=palette["accent"])
    draw.text((cx - bw / 2, cy - 170), big, font=bf, fill=(255, 255, 255))
    img.save(save_path)
    return save_path


def render_recap(scene, palette, save_path) -> Path:
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), color=palette["bg"])
    draw = ImageDraw.Draw(img)
    _draw_confetti(draw, palette, n=50, seed=2)
    _rounded_rect(draw, (80, 80, VIDEO_W - 80, 180), 30, palette["good"])
    title = scene.get("heading", "Great job!")
    title_font = _font(54)
    text_w = draw.textlength(title, font=title_font)
    burst_r = 38
    gap = 22
    total_w = burst_r * 2 + gap + text_w
    burst_cx = int((VIDEO_W - total_w) / 2 + burst_r)
    burst_cy = 130
    _draw_party_burst(
        draw,
        (burst_cx, burst_cy),
        burst_r,
        [(255, 235, 100), (255, 180, 80), (255, 120, 120), (140, 200, 255), (200, 140, 255)],
    )
    draw.text((burst_cx + burst_r + gap, 100), title, font=title_font, fill=(255, 255, 255))
    bullets = scene.get("bullets", [])[:4]
    n = max(1, len(bullets))
    available_h = VIDEO_H - 250
    row_gap = 14
    row_h = max(80, min(120, (available_h - row_gap * (n - 1)) // n))
    bf = _font(28)
    line_h = 36
    max_lines = max(1, (row_h - 24) // line_h)
    text_x = 195
    text_box_w = VIDEO_W - text_x - 90

    y = 230
    for i, b in enumerate(bullets):
        _rounded_rect(draw, (110, y, VIDEO_W - 110, y + row_h), 20, (255, 255, 255))
        col = [palette["accent"], palette["good"], (255, 152, 0), (33, 150, 243)][i % 4]
        cx, cy = 150, y + row_h // 2
        draw.ellipse((cx - 14, cy - 14, cx + 14, cy + 14), fill=col)
        avg = bf.getlength("M") or 16
        wrap_at = max(10, int(text_box_w / avg * 1.7))
        lines = textwrap.wrap(b, width=wrap_at)[:max_lines]
        text_block_h = len(lines) * line_h
        ty = y + (row_h - text_block_h) // 2
        for j, line in enumerate(lines):
            draw.text((text_x, ty + j * line_h), line, font=bf, fill=palette["fg"])
        y += row_h + row_gap
    img.save(save_path)
    return save_path


def render_scene(scene, palette, save_path, reveal=False) -> Path:
    t = scene.get("type")
    if t == "title":
        return render_title(scene, palette, save_path)
    if t == "lesson":
        return render_lesson(scene, palette, save_path)
    if t == "question":
        return render_question(scene, palette, save_path, reveal=reveal)
    if t == "recap":
        return render_recap(scene, palette, save_path)
    return render_title({"text": scene.get("text", ""), "subtitle": ""}, palette, save_path)


def generate_script(idea: str, num_questions: int, anthropic_client: Anthropic) -> dict:
    system = (
        "You design playful, kid-friendly educational quiz videos. "
        "Return ONLY valid JSON, no markdown, no commentary."
    )
    user = f"""Create a fun educational quiz video about: "{idea}"

Structure: 1 title slide, 1 short lesson slide, {num_questions} multiple-choice questions (each with a question slide AND an answer slide), and 1 recap slide.

Requirements:
- Tone: warm, encouraging, suitable for children
- Each question is fill-in-the-blank with EXACTLY 4 options labeled A/B/C/D
- The correct option must be one of "A", "B", "C", or "D"
- Narration is what a teacher would say out loud — friendly and clear

Return ONLY this JSON shape, nothing else:
{{
  "title": "Short video title",
  "scenes": [
    {{"type": "title", "text": "Big title shown on screen", "subtitle": "Optional subtitle", "narration": "Welcome line spoken aloud"}},
    {{"type": "lesson", "heading": "Short heading", "bullets": ["Rule 1", "Rule 2", "Rule 3"], "narration": "Teacher explanation 2-3 sentences"}},
    {{"type": "question", "question_number": "Question 1", "sentence": "I saw ___ elephant at the zoo.", "options": ["a", "an", "the", "no article"], "correct_option": "B", "explanation": "We use 'an' before vowel sounds.", "narration_question": "Read the question aloud and ask the student to pick A B C or D", "narration_answer": "Reveal the correct answer and explain why"}}
  ],
  "recap": {{"heading": "Great job!", "bullets": ["Key takeaway 1", "Key takeaway 2", "Key takeaway 3"], "narration": "Closing encouragement"}}
}}

Provide exactly {num_questions} question objects in the scenes array. Sentences must contain "___" where the blank goes."""

    msg = anthropic_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4000,
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


async def _edge_tts_save(text: str, voice: str, save_path: Path):
    communicate = edge_tts.Communicate(text=text, voice=voice, rate="+0%")
    await communicate.save(str(save_path))


def synthesize_voice(text: str, save_path: Path, voice: str = DEFAULT_VOICE) -> Path:
    asyncio.run(_edge_tts_save(text, voice, save_path))
    if not save_path.exists() or save_path.stat().st_size == 0:
        raise RuntimeError("Edge TTS produced no audio")
    return save_path


def build_clip(image_path: Path, audio_path: Path | None, min_duration: float = 2.0) -> ImageClip:
    if audio_path is not None:
        audio = AudioFileClip(str(audio_path))
        duration = max(audio.duration + 0.4, min_duration)
        clip = ImageClip(str(image_path)).set_duration(duration).set_audio(audio)
    else:
        clip = ImageClip(str(image_path)).set_duration(min_duration)
    return clip.resize((VIDEO_W, VIDEO_H))


def build_ken_burns_clip(
    image_path: Path,
    audio_path: Path | None,
    min_duration: float = 3.0,
    zoom_start: float = 1.0,
    zoom_end: float = 1.12,
    pan_x_pct: float = 0.0,
    pan_y_pct: float = 0.0,
) -> ImageClip:
    """Image clip with slow zoom + optional pan. Renders the image larger and crops
    a moving viewport over time, which avoids MoviePy's per-frame resize cost.
    """
    if audio_path is not None:
        audio = AudioFileClip(str(audio_path))
        duration = max(audio.duration + 0.4, min_duration)
    else:
        audio = None
        duration = min_duration

    pil_img = Image.open(str(image_path)).convert("RGB")
    src_w, src_h = pil_img.size
    target_aspect = VIDEO_W / VIDEO_H
    src_aspect = src_w / src_h
    if src_aspect > target_aspect:
        new_h = src_h
        new_w = int(new_h * target_aspect)
    else:
        new_w = src_w
        new_h = int(new_w / target_aspect)
    left = (src_w - new_w) // 2
    top = (src_h - new_h) // 2
    pil_img = pil_img.crop((left, top, left + new_w, top + new_h))

    scale_factor = max(zoom_start, zoom_end) * 1.05
    big_w = int(VIDEO_W * scale_factor)
    big_h = int(VIDEO_H * scale_factor)
    big_img = pil_img.resize((big_w, big_h), Image.LANCZOS)
    big_arr = np.array(big_img)

    def make_frame(t):
        progress = 0.0 if duration <= 0 else min(max(t / duration, 0.0), 1.0)
        zoom = zoom_start + (zoom_end - zoom_start) * progress
        crop_w = int(VIDEO_W * (scale_factor / zoom))
        crop_h = int(VIDEO_H * (scale_factor / zoom))
        max_x = big_w - crop_w
        max_y = big_h - crop_h
        center_x = max_x // 2 + int(max_x * pan_x_pct * (progress - 0.5))
        center_y = max_y // 2 + int(max_y * pan_y_pct * (progress - 0.5))
        x0 = max(0, min(center_x, max_x))
        y0 = max(0, min(center_y, max_y))
        frame = big_arr[y0:y0 + crop_h, x0:x0 + crop_w]
        if frame.shape[0] != VIDEO_H or frame.shape[1] != VIDEO_W:
            frame = np.array(Image.fromarray(frame).resize((VIDEO_W, VIDEO_H), Image.LANCZOS))
        return frame

    from moviepy.editor import VideoClip
    clip = VideoClip(make_frame, duration=duration)
    if audio is not None:
        clip = clip.set_audio(audio)
    return clip


def generate_video(
    idea: str,
    num_questions: int = 5,
    anthropic_key: str | None = None,
    pexels_key: str | None = None,
    voice: str = DEFAULT_VOICE,
    progress: Callable[[str, float], None] | None = None,
) -> Path:
    anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY is required")

    client = Anthropic(api_key=anthropic_key)

    def step(msg: str, pct: float):
        if progress:
            progress(msg, pct)

    step("Writing quiz script with Claude...", 0.1)
    script = generate_script(idea, num_questions, client)

    workdir = Path(tempfile.mkdtemp(prefix="vidgen_"))
    clips = []

    flat_steps: list[tuple[str, dict, bool, str, float]] = []
    palette = random.choice(PALETTES)

    for sc in script["scenes"]:
        if sc["type"] == "title":
            flat_steps.append(("title", sc, False, sc.get("narration", ""), 0.0))
        elif sc["type"] == "lesson":
            flat_steps.append(("lesson", sc, False, sc.get("narration", ""), 0.0))
        elif sc["type"] == "question":
            flat_steps.append(("question", sc, False, sc.get("narration_question", ""), 0.0))
            for n in (3, 2, 1):
                flat_steps.append(("think", {"number": n}, False, "", 1.0))
            flat_steps.append(("question", sc, True, sc.get("narration_answer", ""), 0.0))

    if "recap" in script:
        recap_scene = {**script["recap"], "type": "recap"}
        flat_steps.append(("recap", recap_scene, False, recap_scene.get("narration", ""), 0.0))

    total = len(flat_steps)
    for i, (kind, sc, reveal, narration, fixed_dur) in enumerate(flat_steps):
        step(f"Rendering slide {i + 1}/{total}...", 0.2 + 0.7 * (i / total))
        img_path = workdir / f"slide_{i:02d}_{kind}_{int(reveal)}.png"

        if kind == "think":
            render_think(sc, palette, img_path, number=sc["number"])
        else:
            render_scene(sc, palette, img_path, reveal=reveal)

        aud_path = None
        if narration.strip():
            aud_path = workdir / f"aud_{i:02d}.mp3"
            synthesize_voice(narration, aud_path, voice=voice)

        if fixed_dur > 0:
            clips.append(ImageClip(str(img_path)).set_duration(fixed_dur).resize((VIDEO_W, VIDEO_H)))
        else:
            clips.append(build_clip(img_path, aud_path))

    step("Assembling final video...", 0.95)
    safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in script["title"])[:60].strip()
    output_path = OUTPUTS_DIR / f"{safe_title or 'video'}.mp4"

    final = concatenate_videoclips(clips, method="compose")
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
