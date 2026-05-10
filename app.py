import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from video_generator import generate_video, VOICES
from story_generator import (
    generate_story_video,
    AUDIENCES,
    LENGTHS,
    ART_STYLES,
    VOCAB_LEVELS,
)

load_dotenv()

st.set_page_config(page_title="Video Generator", page_icon="🎬", layout="centered")

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if APP_PASSWORD:
    pwd = st.text_input("Enter access password", type="password")
    if pwd != APP_PASSWORD:
        st.stop()

st.title("🎬 Simple Video Generator")

mode = st.radio(
    "What kind of video?",
    options=["Quiz", "Story"],
    horizontal=True,
    help="Quiz = multiple-choice educational video. Story = illustrated narrative for kids.",
)

if mode == "Quiz":
    st.write("Type your topic. Get a quiz-style educational video.")
else:
    st.write("Type a story idea. Get an illustrated story video.")
    st.caption("Phase 1: illustrations are placeholders. Real AI art comes in Phase 3.")


with st.form("video_form"):
    if mode == "Quiz":
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
    else:
        idea = st.text_area(
            "What is your story about?",
            placeholder="e.g. The story of Prahlad and Holika (kid-friendly retelling)",
            height=100,
        )
        col1, col2 = st.columns(2)
        with col1:
            audience = st.selectbox("Audience", options=list(AUDIENCES.keys()), index=1)
            art_style = st.selectbox("Art style", options=list(ART_STYLES.keys()), index=0)
        with col2:
            length_label = st.selectbox("Length", options=list(LENGTHS.keys()), index=1)
            vocab_level = st.selectbox(
                "Vocabulary teaching",
                options=list(VOCAB_LEVELS.keys()),
                index=1,
                help="Off = no tough word highlighting. Light = 2/scene. Heavy = 5/scene.",
            )
        voice_label = st.selectbox(
            "Narrator voice",
            options=list(VOICES.keys()),
            index=1,
        )

    submitted = st.form_submit_button(f"Generate {mode} Video", type="primary")

if submitted:
    if not idea.strip():
        st.error("Please enter a topic.")
        st.stop()

    anthropic_key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    pexels_key = os.getenv("PEXELS_API_KEY") or st.secrets.get("PEXELS_API_KEY", "")
    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or st.secrets.get("HF_TOKEN", "")
    )
    gemini_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or st.secrets.get("GEMINI_API_KEY", "")
    )
    if not anthropic_key:
        st.error("Missing ANTHROPIC_API_KEY. Add it to .env (local) or Streamlit secrets (cloud).")
        st.stop()
    if mode == "Story" and not gemini_key:
        st.warning(
            "GEMINI_API_KEY not set — story scenes will render as placeholders. "
            "Get a free key at https://aistudio.google.com/apikey and add GEMINI_API_KEY to your .env."
        )

    progress_bar = st.progress(0.0)
    status = st.empty()

    def on_progress(msg: str, pct: float):
        status.info(msg)
        progress_bar.progress(min(max(pct, 0.0), 1.0))

    try:
        with st.spinner("Working..."):
            if mode == "Quiz":
                output_path: Path = generate_video(
                    idea=idea,
                    num_questions=num_questions,
                    anthropic_key=anthropic_key,
                    pexels_key=pexels_key,
                    voice=VOICES[voice_label],
                    progress=on_progress,
                )
            else:
                output_path = generate_story_video(
                    idea=idea,
                    audience=audience,
                    length_label=length_label,
                    art_style=art_style,
                    vocab_level=vocab_level,
                    voice=VOICES[voice_label],
                    anthropic_key=anthropic_key,
                    hf_token=hf_token,
                    gemini_key=gemini_key,
                    image_provider="gemini",
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
