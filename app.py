"""
Crop Disease Diagnosis — VLM Demo
A deployed, single-agent companion to the VIDA+PANDA multi-agent debate
research framework. Upload a leaf image, get a diagnosis with reasoning.

Repo: https://github.com/YOUR_USERNAME/crop-disease-vlm-demo
Related research: https://github.com/YOUR_USERNAME/vida-panda-vlm-debate
"""

import base64
import io
import json
import os
import time
from pathlib import Path

import streamlit as st
from PIL import Image
from together import Together

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Disease Diagnosis — VLM Demo",
    page_icon="🌿",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_STRING = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
SAMPLE_DIR = Path(__file__).parent / "sample_images"
MAX_IMAGE_DIM = 1024  # downscale large uploads before sending to the API

TAXONOMY = [
    "Apple,Alternaria Blotch", "Apple,Black Rot", "Apple,Brown Spot",
    "Apple,Cedar Apple Rust", "Apple,Frog Eye Leaf Spot", "Apple,Grey Spot",
    "Apple,Healthy", "Apple,Leaf Rust", "Apple,Mosaic Virus",
    "Apple,Powdery Mildew", "Apple,Scab",
    "Bell Pepper,Bacterial Spot", "Bell Pepper,Healthy",
    "Blueberry,Healthy",
    "Cherry,Healthy", "Cherry,Powdery Mildew",
    "Corn,Healthy", "Corn,Leaf Rust", "Corn,Leaf Spot", "Corn,Northern Leaf Blight",
    "Grape,Black Rot", "Grape,Esca", "Grape,Healthy", "Grape,Leaf Blight",
    "Orange,Citrus Greening", "Orange,Healthy",
    "Peach,Bacterial Spot", "Peach,Healthy",
    "Potato,Early Blight", "Potato,Healthy", "Potato,Late Blight",
    "Pumpkin,Powdery Mildew",
    "Raspberry,Healthy",
    "Rice,Bacterial Leaf Blight", "Rice,Blast", "Rice,Brown Spot",
    "Rice,Leaf Blight", "Rice,Leaf Smut", "Rice,Tungro",
    "Soybean,Healthy",
    "Strawberry,Healthy", "Strawberry,Leaf Scorch",
    "Tomato,Bacterial Spot", "Tomato,Early Blight", "Tomato,Healthy",
    "Tomato,Late Blight", "Tomato,Leaf Mold", "Tomato,Mosaic Virus",
    "Tomato,Powdery Mildew", "Tomato,Septoria Leaf Spot", "Tomato,Spider Mites",
    "Tomato,Target Spot", "Tomato,Yellow Leaf Curl Virus",
    "Wheat,Healthy", "Wheat,Leaf Rust", "Wheat,Loose Smut",
    "Wheat,Root Rot", "Wheat,Septoria Leaf Spot", "Wheat,Stem Rust",
    "Wheat,Stripe Rust",
]

SYSTEM_PROMPT = f"""You are a plant pathology assistant analyzing a leaf image for crop disease diagnosis.

You must choose exactly one label from this taxonomy (format is Crop,Condition):
{", ".join(TAXONOMY)}

Respond ONLY with valid JSON in this exact structure, no markdown fences, no extra text:
{{
  "label": "<one exact label from the taxonomy above>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-3 sentences citing specific visual evidence: lesion color, shape, pattern, leaf region affected>"
}}"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_client():
    api_key = st.secrets.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_API_KEY")
    # TEMPORARY DEBUG — remove after diagnosing key issue
    if api_key:
        st.sidebar.caption(f"Debug: key found, length={len(api_key)}, starts='{api_key[:6]}...'")
    else:
        st.sidebar.caption("Debug: no key found at all")
    if not api_key:
        return None
    return Together(api_key=api_key.strip())


def preprocess_image(image: Image.Image) -> Image.Image:
    """Downscale large images to keep API payloads small and fast."""
    image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)))
    return image


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def diagnose(client: Together, image: Image.Image) -> dict:
    b64 = image_to_base64(image)
    response = client.chat.completions.create(
        model=MODEL_STRING,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=400,
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def load_sample_images() -> dict:
    samples = {}
    if SAMPLE_DIR.exists():
        for f in sorted(SAMPLE_DIR.glob("*.jpg")):
            label = f.stem.replace("_", " ")
            samples[label] = f
    return samples


# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #FAF7F0;
    }
    h1, h2, h3 {
        font-family: 'Lora', serif !important;
        color: #1B4332 !important;
    }
    .field-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6B6253;
        margin-bottom: 0.2rem;
    }
    .specimen-card {
        background: #FFFFFF;
        border: 1px solid #E3DCC9;
        border-left: 5px solid #1B4332;
        border-radius: 4px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
    }
    .specimen-card.healthy {
        border-left-color: #74A57F;
    }
    .specimen-card.diseased {
        border-left-color: #BC6C25;
    }
    .diagnosis-title {
        font-family: 'Lora', serif;
        font-style: italic;
        font-size: 1.5rem;
        color: #1A1A1A;
        margin: 0.1rem 0 0.6rem 0;
    }
    .confidence-track {
        background: #EFEAE0;
        border-radius: 3px;
        height: 8px;
        overflow: hidden;
        margin: 0.4rem 0 0.8rem 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 3px;
    }
    .reasoning-text {
        font-size: 0.95rem;
        line-height: 1.55;
        color: #2D2D2D;
    }
    .sample-caption {
        font-size: 0.78rem;
        color: #6B6253;
        text-align: center;
        margin-top: -0.4rem;
    }
    footer, #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="field-label">VISION-LANGUAGE DIAGNOSTIC TOOL</div>',
    unsafe_allow_html=True,
)
st.title("Crop Disease Diagnosis")
st.markdown(
    """
    Upload a leaf photo and a vision-language model will diagnose it against a
    59-class taxonomy spanning 13 crop species — the same taxonomy and
    methodology behind
    [**VIDA+PANDA**](https://github.com/YOUR_USERNAME/vida-panda-vlm-debate),
    a multi-agent debate framework for crop disease diagnosis. This demo runs a
    single open-weight model for a fast, lightweight, publicly deployable version
    of that research.
    """
)

client = get_client()
if client is None:
    st.error(
        "No Together.ai API key found. Set `TOGETHER_API_KEY` in Streamlit "
        "secrets to run this app.",
        icon="🔑",
    )
    st.stop()

st.divider()

# ── Image selection ───────────────────────────────────────────────────────────
st.markdown('<div class="field-label">STEP 1 — CHOOSE A LEAF IMAGE</div>', unsafe_allow_html=True)

tab_sample, tab_upload = st.tabs(["Try a sample", "Upload your own"])

selected_image = None
selected_image_caption = None

with tab_sample:
    samples = load_sample_images()
    if samples:
        cols = st.columns(3)
        sample_choice = st.session_state.get("sample_choice")
        for i, (label, path) in enumerate(samples.items()):
            with cols[i % 3]:
                st.image(str(path), use_container_width=True)
                st.markdown(f'<div class="sample-caption">{label}</div>', unsafe_allow_html=True)
                if st.button("Use this", key=f"sample_{i}", use_container_width=True):
                    st.session_state["sample_choice"] = str(path)
        if st.session_state.get("sample_choice"):
            selected_image = Image.open(st.session_state["sample_choice"])
            selected_image_caption = Path(st.session_state["sample_choice"]).stem.replace("_", " ")
    else:
        st.info("No sample images bundled with this deployment.")

with tab_upload:
    uploaded = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        selected_image = Image.open(uploaded)
        selected_image_caption = "Your upload"
        st.session_state["sample_choice"] = None

# ── Diagnosis ─────────────────────────────────────────────────────────────────
if selected_image is not None:
    st.divider()
    st.markdown('<div class="field-label">STEP 2 — REVIEW</div>', unsafe_allow_html=True)

    col_img, col_action = st.columns([1, 1])
    with col_img:
        st.image(selected_image, caption=selected_image_caption, use_container_width=True)
    with col_action:
        st.write("")
        st.write("")
        run = st.button("Diagnose this leaf →", type="primary", use_container_width=True)
        st.caption(
            f"Powered by {MODEL_STRING.split('/')[-1]} via Together.ai. "
            "Single-pass inference — no multi-agent debate in this lightweight demo."
        )

    if run:
        with st.spinner("Analyzing visual symptoms…"):
            img_for_model = preprocess_image(selected_image)
            try:
                result = diagnose(client, img_for_model)
            except json.JSONDecodeError:
                st.error("The model returned a response that couldn't be parsed. Please try again.")
                st.stop()
            except Exception as e:
                st.error(f"Diagnosis failed: {e}")
                st.stop()

        label = result.get("label", "Unknown")
        confidence = float(result.get("confidence", 0))
        reasoning = result.get("reasoning", "No reasoning provided.")
        is_healthy = "healthy" in label.lower()

        card_class = "healthy" if is_healthy else "diseased"
        bar_color = "#74A57F" if is_healthy else "#BC6C25"

        st.markdown(
            f"""
            <div class="specimen-card {card_class}">
                <div class="field-label">DIAGNOSIS</div>
                <div class="diagnosis-title">{label}</div>
                <div class="field-label">CONFIDENCE — {confidence:.0%}</div>
                <div class="confidence-track">
                    <div class="confidence-fill" style="width:{confidence*100:.0f}%; background:{bar_color};"></div>
                </div>
                <div class="field-label">REASONING</div>
                <div class="reasoning-text">{reasoning}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "⚠️ Research demo only — not for agricultural decision-making. "
            "Single-model inference can be wrong; the full VIDA+PANDA framework "
            "uses multi-agent debate specifically to catch and correct these errors."
        )
else:
    st.info("Choose a sample image or upload your own leaf photo to begin.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style="font-size:0.85rem; color:#6B6253; text-align:center;">
    Built by Zayed Al Aghbari · 
    <a href="https://github.com/YOUR_USERNAME/vida-panda-vlm-debate" style="color:#1B4332;">VIDA+PANDA research repo</a> · 
    <a href="https://github.com/YOUR_USERNAME/crop-disease-vlm-demo" style="color:#1B4332;">source code</a>
    </div>
    """,
    unsafe_allow_html=True,
)
