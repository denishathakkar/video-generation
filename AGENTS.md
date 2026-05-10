# Personal Finance / Video Generator

## What this project is
A Streamlit app that generates two kinds of educational YouTube videos:
1. **Quiz** — multiple-choice fill-in-the-blank for English/math/general knowledge
2. **Story** — illustrated narrative videos for kids (Indian mythology, fables)

Target end user: a non-technical teacher. The author deploys it, the
teacher uses it via a deployed URL with a password.

## Tech stack
- Python 3.11 (NOT 3.14 — Pillow/MoviePy don't build on 3.14)
- Streamlit UI
- Claude API for script writing
- Edge TTS for narration (NOT gTTS — gTTS sounds robotic)
- Google Gemini "Nano Banana" (`gemini-2.5-flash-image`) for illustrations.
  Do NOT use HF FLUX — free-tier credits are depleted and it 402s.
- MoviePy 1.0.3 (pinned) for video assembly
- Pillow >= 11 (older versions don't build on Streamlit Cloud)

## Key files
- `app.py` — Streamlit UI with Quiz/Story mode switcher
- `video_generator.py` — quiz pipeline + shared helpers (TTS, Ken Burns, palettes)
- `story_generator.py` — story pipeline (Claude script + illustrations + assembly)
- `illustration.py` — multi-provider image generation (hf-flux / hf-sd / gemini / gemini-pro)
- `test_illustration.py` — runs the illustration pipeline for one story, no video assembly
- `test_compare.py` — generates the same prompt across multiple providers, outputs HTML report

## Environment
Secrets live in `.env` (gitignored). Required:
- `ANTHROPIC_API_KEY` — Claude
- `GEMINI_API_KEY` — image generation
- `APP_PASSWORD` — gates the Streamlit app
Optional:
- `HF_TOKEN` — only needed if reverting to HF providers
- `PEXELS_API_KEY` — for the older quiz "stock photo" path (unused now)

## How to run locally
```bash
source .venv/bin/activate
streamlit run app.py
```
The venv uses Python 3.11. ImageMagick is installed via brew but NOT
required — captions render via PIL directly.

## Deployment
Streamlit Community Cloud, repo: `denishathakkar/video-generation` (private).
Build config:
- `.python-version` (NOT `runtime.txt`) pins Python to 3.11
- `packages.txt` installs ffmpeg + fonts on the Linux build
Secrets are pasted in the Streamlit Cloud "Secrets" UI (TOML format).

## Known gotchas / lessons from past sessions
- `gemini-2.5-flash-image-preview` does NOT exist. Use `gemini-2.5-flash-image`.
- The first call to a fresh Gemini key sometimes returns 429. The illustration
  module retries with backoff — don't remove that.
- Streamlit Cloud's free tier is 1 GB RAM. Stories longer than ~10 scenes
  may OOM during MoviePy assembly.
- Story image prompts must avoid violence wording ("strike", "weapon",
  "blood") — content filters silently return tiny garbage images. The Claude
  prompt already instructs Claude to soften violence into peaceful tableaus.
- HF "Read" tokens don't grant inference access; need a fine-grained token
  with "Make calls to Inference Providers" checked under User permissions.

## How to test illustration changes WITHOUT building a full video
Run `.venv/bin/python test_compare.py --providers gemini --scenes 5` to
generate just images. Output goes to `test_outputs/compare/`.
This is the "test harness" — use it whenever modifying illustration logic.

For full-pipeline scene-level diagnostics (Claude script + per-scene
illustration, no video assembly), use `test_illustration.py`.

## Style / conventions
- No emoji unless explicitly asked.
- Don't add comments that just describe what the code does. Comments explain *why*.
- Don't add backwards-compatibility shims when changing code.
- Don't auto-add features. Ask before scope creep.
- Default to free-tier services. Costs get user-confirmed.

## What NOT to do
- Don't suggest DALL-E 3, Replicate, or other paid services without asking.
- Don't add silent fallbacks that mask real errors (we learned this twice —
  once when Edge TTS silently fell back to gTTS, once when HF 402 errors
  silently fell back to placeholders).
- Don't push to GitHub from the agent. The user pushes manually after review.
