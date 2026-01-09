import streamlit as st
from PIL import Image
import os
import tempfile

from utils.text_inference import predict_text
from utils.image_inference import predict_image
from utils.audio_inference import predict_audio


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Multimodal Fake News Detection",
    layout="centered"
)

# =========================
# CUSTOM CSS (REACT-INSPIRED)
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #020617, #0f172a);
}
.main {
    background: transparent;
}
.card {
    background: rgba(15, 23, 42, 0.7);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    border: 1px solid rgba(148, 163, 184, 0.2);
}
.title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #22d3ee, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    color: #cbd5f5;
    margin-bottom: 30px;
}
.metric-box {
    padding: 18px;
    border-radius: 14px;
    margin-top: 15px;
}
.fake {
    background: rgba(220, 38, 38, 0.2);
    border: 1px solid rgba(220, 38, 38, 0.4);
}
.real {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.4);
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="title">Multimodal Fake News Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-based credibility analysis using <b>Text</b>, <b>Images</b>, and <b>Audio</b></div>',
    unsafe_allow_html=True
)

# =========================
# INPUT CARD
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)

tab_text, tab_image, tab_audio = st.tabs(["📝 Text", "🖼 Image", "🎧 Audio"])

with tab_text:
    text_input = st.text_area(
        "News Text (required)",
        height=180,
        placeholder="Paste the news article or headline here..."
    )

with tab_image:
    image_file = st.file_uploader(
        "Upload image (optional)",
        type=["jpg", "jpeg", "png"]
    )

with tab_audio:
    audio_file = st.file_uploader(
        "Upload audio (optional)",
        type=["wav", "mp3"]
    )

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ANALYZE BUTTON
# =========================
analyze = st.button("🔍 Analyze Content", use_container_width=True)

# =========================
# INFERENCE
# =========================
if analyze:
    if not text_input.strip():
        st.warning("⚠️ Text input is required.")
    else:
        with st.spinner("Analyzing multimodal content..."):
            scores = []
            weights = []

            # ---- TEXT
            text_score = predict_text(text_input)
            scores.append(("Text", text_score))
            weights.append(0.6)

            # ---- IMAGE
            if image_file is not None:
                image = Image.open(image_file)
                image_score = predict_image(image)
                scores.append(("Image", image_score))
                weights.append(0.2)

            # ---- AUDIO
            if audio_file is not None:
                suffix = os.path.splitext(audio_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_file.read())
                    audio_path = tmp.name

                audio_score = predict_audio(audio_path)
                scores.append(("Audio", audio_score))
                weights.append(0.2)

            final_score = sum(w * s for (_, s), w in zip(scores, weights)) / sum(weights)
            label = "FAKE" if final_score > 0.5 else "REAL"

        # =========================
        # RESULTS CARD
        # =========================
        box_class = "fake" if label == "FAKE" else "real"

        st.markdown(f'<div class="card metric-box {box_class}">', unsafe_allow_html=True)
        st.subheader(f"🧠 Final Prediction: {label}")
        st.write(f"**Final Confidence Score:** `{final_score:.3f}`")
        st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # BREAKDOWN
        # =========================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔎 Modal Confidence Breakdown")

        for name, score in scores:
            st.write(f"**{name}**")
            st.progress(min(max(score, 0.0), 1.0))
            st.caption(f"Score: {score:.3f}")

        with st.expander("ℹ️ Interpretation Guide"):
            st.write(
                """
                - Scores close to **1.0** indicate higher likelihood of fake or manipulated content  
                - Scores close to **0.0** indicate higher likelihood of authentic content  
                - Final decision is a weighted fusion of all available modalities  
                - Text is the primary signal; image and audio act as supporting evidence
                """
            )

        st.markdown('</div>', unsafe_allow_html=True)
