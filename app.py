import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from video_generator import generate_video, VOICES

load_dotenv()

st.set_page_config(page_title="Video Generator", page_icon="🎬", layout="centered")

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if APP_PASSWORD:
    pwd = st.text_input("Enter access password", type="password")
    if pwd != APP_PASSWORD:
        st.stop()

st.title("🎬 Simple Video Generator")
st.write("Type your topic. Get a YouTube-ready video.")

with st.form("video_form"):
    idea = st.text_area(
        "What is your video about?",
        placeholder="e.g. Teach English - Fill in the blanks with articles (a / an / the)",
        height=100,
    )
    num_questions = st.select_slider(
        "How many quiz questions?",
        options=[3, 5, 8, 10],
        value=5,
    )
    voice_label = st.selectbox(
        "Narrator voice",
        options=list(VOICES.keys()),
        index=1,
    )
    submitted = st.form_submit_button("Generate Video", type="primary")

if submitted:
    if not idea.strip():
        st.error("Please enter a topic.")
        st.stop()

    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    pexels_key = os.getenv("PEXELS_API_KEY") or st.secrets.get("PEXELS_API_KEY", "")
    if not anthropic_key:
        st.error("Missing ANTHROPIC_API_KEY. Add it to .env (local) or Streamlit secrets (cloud).")
        st.stop()

    progress_bar = st.progress(0.0)
    status = st.empty()

    def on_progress(msg: str, pct: float):
        status.info(msg)
        progress_bar.progress(min(max(pct, 0.0), 1.0))

    try:
        with st.spinner("Working..."):
            output_path: Path = generate_video(
                idea=idea,
                num_questions=num_questions,
                anthropic_key=anthropic_key,
                pexels_key=pexels_key,
                voice=VOICES[voice_label],
                progress=on_progress,
            )
        status.success("Video ready!")
        st.video(str(output_path))
        with open(output_path, "rb") as f:
            st.download_button(
                "⬇️ Download MP4",
                data=f,
                file_name=output_path.name,
                mime="video/mp4",
            )
    except Exception as e:
        st.error(f"Generation failed: {e}")
