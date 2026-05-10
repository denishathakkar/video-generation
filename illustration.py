"""AI illustration generation with multiple provider backends.

Providers:
- "hf-flux"     : FLUX.1-schnell via HF routed providers (PAID after free quota — needs HF PRO or pay-as-you-go)
- "hf-sd"       : Stable Diffusion XL via Hugging Face Inference API (free tier, no paid router)
- "gemini"      : Google Gemini 2.5 Flash Image (free tier on Google AI Studio)

Key tricks for consistent storybook visuals:
- Style template injected into every prompt
- Character cards injected when characters appear in the prompt
- Per-story random seed kept stable across scenes
- Safety suffix that pushes results toward family-friendly storybook art
"""

import base64
import hashlib
import io
import os
from pathlib import Path

from huggingface_hub import InferenceClient
from PIL import Image

CACHE_DIR = Path(__file__).parent / "outputs" / "_image_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROVIDERS = {
    "hf-flux": "Hugging Face — FLUX.1-schnell (paid after free quota)",
    "hf-sd": "Hugging Face — Stable Diffusion XL (paid after free quota)",
    "gemini": "Google Gemini 2.5 Flash Image / Nano Banana (free tier)",
    "gemini-pro": "Google Gemini 3 Pro Image / Nano Banana Pro (paid)",
}
DEFAULT_PROVIDER = "gemini"

HF_FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"
HF_SD_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

STYLE_TEMPLATES = {
    "Indian storybook": (
        "traditional Indian storybook illustration, Amar Chitra Katha style, "
        "warm earth tones, intricate decorative borders, folk art, "
        "rich saturated colors, dramatic lighting, flat painterly textures"
    ),
    "Watercolor": (
        "soft watercolor painting illustration, gentle washes, dreamlike pastels, "
        "subtle paper texture, delicate brush strokes, light airy mood"
    ),
    "3D cartoon": (
        "3D animated cartoon style, Pixar-inspired, expressive characters, "
        "vibrant lighting, glossy materials, family-friendly, cinematic composition"
    ),
    "Comic": (
        "comic book illustration, bold black outlines, dynamic poses, "
        "halftone shading, vivid flat colors, dramatic angles"
    ),
}

NEGATIVE_PROMPT = (
    "text, watermark, signature, low quality, blurry, deformed, mutated, "
    "extra limbs, bad anatomy, ugly, scary, gore, violent, weapons drawn, "
    "blood, dark threatening, horror, photorealistic violence"
)


def _cache_key(prompt: str, provider: str, seed: int) -> str:
    h = hashlib.sha256(f"{provider}|{seed}|{prompt}".encode("utf-8")).hexdigest()[:16]
    return h


def _inject_character_cards(image_prompt: str, character_cards: dict[str, str]) -> str:
    if not character_cards:
        return image_prompt
    referenced = []
    lower = image_prompt.lower()
    for name, desc in character_cards.items():
        if name.lower() in lower:
            referenced.append(f"{name}: {desc}")
    if not referenced:
        return image_prompt
    return image_prompt + " | Character references: " + " ; ".join(referenced)


def build_full_prompt(image_prompt, art_style, character_cards=None):
    style = STYLE_TEMPLATES.get(art_style, STYLE_TEMPLATES["Indian storybook"])
    enriched = _inject_character_cards(image_prompt, character_cards or {})
    safety = (
        "wholesome family-friendly children's illustration, peaceful tone, "
        "no violence, no weapons drawn, painterly storybook art"
    )
    return f"{enriched}. Style: {style}. {safety}. Aspect 16:9, high detail, no text, no watermark."


def _gen_hf(model, full_prompt, seed, width, height, hf_token, steps):
    client = InferenceClient(model=model, token=hf_token)
    img = client.text_to_image(
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_inference_steps=steps,
        seed=seed,
    )
    if isinstance(img, bytes):
        img = Image.open(io.BytesIO(img))
    return img.convert("RGB")


def _gen_gemini(full_prompt, width, height, api_key, model_id="gemini-2.5-flash-image"):
    """Use Google Gemini image models (Nano Banana family). Retries on 429."""
    import time
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    last_err = None
    for attempt in range(4):
        try:
            r = requests.post(url, json=payload, timeout=90)
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                print(f"[gemini:{model_id}] 429 rate-limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            for cand in data.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    inline = part.get("inlineData") or part.get("inline_data")
                    if inline and "data" in inline:
                        raw = base64.b64decode(inline["data"])
                        return Image.open(io.BytesIO(raw)).convert("RGB")
            raise RuntimeError(f"No image in response: {str(data)[:300]}")
        except Exception as e:
            last_err = e
            if attempt < 3:
                time.sleep(3)
                continue
            raise
    raise RuntimeError(f"Gemini failed after retries: {last_err}")


def generate_illustration(
    image_prompt: str,
    art_style: str,
    save_path: Path,
    character_cards: dict[str, str] | None = None,
    seed: int = 42,
    hf_token: str | None = None,
    gemini_key: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    width: int = 1280,
    height: int = 720,
) -> Path | None:
    hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    gemini_key = gemini_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    full_prompt = build_full_prompt(image_prompt, art_style, character_cards)

    cache_path = CACHE_DIR / f"{_cache_key(full_prompt, provider, seed)}.png"
    if cache_path.exists() and cache_path.stat().st_size > 50_000:
        Image.open(cache_path).convert("RGB").resize((width, height), Image.LANCZOS).save(save_path)
        return save_path

    try:
        if provider == "hf-flux":
            if not hf_token:
                raise RuntimeError("HF_TOKEN missing")
            img = _gen_hf(HF_FLUX_MODEL, full_prompt, seed, width, height, hf_token, steps=4)
        elif provider == "hf-sd":
            if not hf_token:
                raise RuntimeError("HF_TOKEN missing")
            img = _gen_hf(HF_SD_MODEL, full_prompt, seed, width, height, hf_token, steps=25)
        elif provider == "gemini":
            if not gemini_key:
                raise RuntimeError("GEMINI_API_KEY missing")
            img = _gen_gemini(full_prompt, width, height, gemini_key, model_id="gemini-2.5-flash-image")
        elif provider == "gemini-pro":
            if not gemini_key:
                raise RuntimeError("GEMINI_API_KEY missing")
            img = _gen_gemini(full_prompt, width, height, gemini_key, model_id="gemini-3-pro-image-preview")
        else:
            raise ValueError(f"Unknown provider: {provider}")

        img.save(cache_path)
        if cache_path.stat().st_size < 50_000:
            cache_path.unlink(missing_ok=True)
            print(f"[illustration:{provider}] Got tiny image, likely filtered. Failing.")
            return None

        img.resize((width, height), Image.LANCZOS).save(save_path)
        return save_path
    except Exception as e:
        print(f"[illustration:{provider}] failed: {e}")
        return None
