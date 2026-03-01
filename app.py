import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Gender Vision AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #f0f0f0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #1a0533 0%, #0a0a0f 50%),
                radial-gradient(ellipse at 80% 80%, #0d1a2e 0%, #0a0a0f 50%);
    background-blend-mode: screen;
    min-height: 100vh;
}

[data-testid="stHeader"] { background: transparent; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Main content */
.block-container {
    padding: 2rem 2rem 4rem 2rem !important;
    max-width: 750px !important;
}

/* ── HERO ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem 0;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(139,92,246,0.2), rgba(59,130,246,0.2));
    border: 1px solid rgba(139,92,246,0.4);
    color: #a78bfa;
    font-size: 0.72rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 6vw, 3.8rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    letter-spacing: -0.02em;
}

.hero-sub {
    font-size: 1.05rem;
    color: rgba(240,240,240,0.5);
    font-weight: 300;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── UPLOAD ZONE ── */
.upload-section {
    margin: 2.5rem 0 1.5rem 0;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(139,92,246,0.35) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    transition: border-color 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(139,92,246,0.7) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── MODEL PATH INPUT ── */
.model-section {
    margin-bottom: 1.5rem;
}

[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(139,92,246,0.25) !important;
    border-radius: 12px !important;
    color: #f0f0f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.75rem 1rem !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: rgba(139,92,246,0.6) !important;
    box-shadow: 0 0 0 2px rgba(139,92,246,0.15) !important;
}

/* ── BUTTON ── */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.35) !important;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(124,58,237,0.5) !important;
}

/* ── RESULT CARD ── */
.result-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 2.5rem 2rem;
    margin-top: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    animation: fadeSlideUp 0.5s ease forwards;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-icon {
    font-size: 4rem;
    margin-bottom: 0.8rem;
    display: block;
}

.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}

.result-label.men {
    background: linear-gradient(135deg, #60a5fa, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.result-label.women {
    background: linear-gradient(135deg, #f472b6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.result-confidence {
    font-size: 1rem;
    color: rgba(240,240,240,0.5);
    margin-bottom: 1.5rem;
}

/* Progress bar */
.conf-bar-wrap {
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    height: 8px;
    width: 80%;
    margin: 0 auto 0.5rem auto;
    overflow: hidden;
}

.conf-bar-fill-men {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #3b82f6, #60a5fa);
    transition: width 1s ease;
}

.conf-bar-fill-women {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #a78bfa, #f472b6);
    transition: width 1s ease;
}

.conf-percent {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: rgba(240,240,240,0.8);
}

/* ── ERROR ── */
.err-box {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    color: #fca5a5;
    font-size: 0.9rem;
    margin-top: 1rem;
}

/* ── DIVIDER ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,92,246,0.3), transparent);
    margin: 2rem 0;
}

/* ── FOOTER ── */
.footer {
    text-align: center;
    padding: 2rem 0 0 0;
    color: rgba(240,240,240,0.2);
    font-size: 0.78rem;
    letter-spacing: 0.05em;
}

/* Image display */
[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
}

[data-testid="stImage"] img {
    border-radius: 16px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Label styling */
label, [data-testid="stWidgetLabel"] p {
    color: rgba(240,240,240,0.6) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  MODEL LOADER (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_gender_model(path):
    model = load_model(path)
    return model


# ─────────────────────────────────────────
#  PREDICT FUNCTION
# ─────────────────────────────────────────
def predict_gender(image: Image.Image, model):
    # Pure Pillow se resize — cv2 ki zaroorat nahi
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_input, verbose=0)[0][0]

    if pred >= 0.5:
        label = "Women"
        confidence = float(pred) * 100
    else:
        label = "Men"
        confidence = (1 - float(pred)) * 100

    return label, confidence, float(pred)


# ─────────────────────────────────────────
#  UI
# ─────────────────────────────────────────

# HERO
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ AI Powered · MobileNetV2</div>
    <div class="hero-title">Gender Vision AI</div>
    <div class="hero-sub">Upload any image and let the model instantly classify the gender with high confidence.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── MODEL PATH ──
st.markdown('<div class="model-section">', unsafe_allow_html=True)
model_path = st.text_input(
    "Model File Path (.h5)",
    value="gender_classifier.h5",
    placeholder="e.g.  C:/Users/YourName/Downloads/gender_classifier.h5"
)
st.markdown('</div>', unsafe_allow_html=True)

# ── IMAGE UPLOAD ──
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible"
)
st.markdown('</div>', unsafe_allow_html=True)

# ── PREVIEW + PREDICT ──
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍  Analyze Gender"):
        # Check model path
        if not os.path.exists(model_path):
            st.markdown(f"""
            <div class="err-box">
                ❌ Model file not found at: <code>{model_path}</code><br>
                Please enter the correct full path to your <code>.h5</code> file.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing..."):
                try:
                    model = load_gender_model(model_path)
                    label, confidence, raw_score = predict_gender(image, model)

                    icon = "👨" if label == "Men" else "👩"
                    css_class = "men" if label == "Men" else "women"
                    bar_class = "conf-bar-fill-men" if label == "Men" else "conf-bar-fill-women"

                    st.markdown(f"""
                    <div class="result-card">
                        <span class="result-icon">{icon}</span>
                        <div class="result-label {css_class}">{label}</div>
                        <div class="result-confidence">Model is confident</div>
                        <div class="conf-bar-wrap">
                            <div class="{bar_class}" style="width:{confidence:.1f}%"></div>
                        </div>
                        <div class="conf-percent">{confidence:.2f}% Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(f"""
                    <div class="err-box">
                        ❌ Error loading model or predicting:<br><code>{str(e)}</code>
                    </div>
                    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; color:rgba(240,240,240,0.2); font-size:0.85rem; padding: 1rem 0;">
        ↑ Upload an image to get started
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ──
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    Built with TensorFlow · MobileNetV2 Transfer Learning · Streamlit
</div>
""", unsafe_allow_html=True)
