# Simple Video Generator

Type a topic → get a finished YouTube-ready MP4. Powered by Claude (script), gTTS (voiceover), Pexels (images), MoviePy (assembly).

## How it works

```
idea  →  Claude writes structured script  →  per scene: stock image + voiceover
                                                                ↓
                                              MoviePy stitches → output.mp4
```

## Cost
- Claude API: ~$0.05 per 2-minute video
- Everything else: free
- Hosting (Streamlit Cloud free tier): $0

---

## Run Locally (your Mac)

### 1. Install system dependency
MoviePy needs FFmpeg and ImageMagick (for captions):

```bash
brew install ffmpeg imagemagick
```

### 2. Set up Python environment
```bash
cd /Users/denisha.thakkar/Documents/personal-finance
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Get API keys
- **Claude** (required): https://console.anthropic.com/ → API Keys
- **Pexels** (optional, for stock images): https://www.pexels.com/api/ → free, instant

### 4. Configure secrets
```bash
cp .env.example .env
```
Edit `.env` and paste your keys.

### 5. Run
```bash
streamlit run app.py
```
Browser opens at http://localhost:8501. Enter a topic, click Generate.

### Test prompt
> Teach English - Fill in the blanks with articles (a / an / the). 5 example sentences for Grade 4 students.

---

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to a **private GitHub repo**.
2. Go to https://share.streamlit.io → "New app" → connect your repo.
3. Main file: `app.py`.
4. Under **Settings → Secrets**, paste:
   ```
   ANTHROPIC_API_KEY = "sk-ant-..."
   PEXELS_API_KEY    = "..."
   APP_PASSWORD      = "familycode2026"
   ```
5. Deploy. You'll get a URL like `https://your-app.streamlit.app`.
6. Send the URL + password to your mother-in-law. She opens it in any browser.

### Notes for Streamlit Cloud
- Free tier: 1 GB RAM. Keep videos to 1–3 minutes.
- App sleeps after inactivity → first load takes ~30 sec.
- `APP_PASSWORD` blocks strangers. Without it, anyone with the link can use your API budget.

---

## File map
- `app.py` — Streamlit UI
- `video_generator.py` — pipeline (script → voice → images → mp4)
- `requirements.txt` — Python deps
- `.env.example` — secrets template
- `outputs/` — generated videos land here
