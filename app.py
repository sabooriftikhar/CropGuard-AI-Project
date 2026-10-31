import json
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.keras"
CLASSES_PATH = APP_DIR / "class_names.json"
METRICS_PATH = APP_DIR / "metrics.json"
IMAGE_SIZE = (224, 224)
TOP_K = 3

# Add your links here later.
PROJECT_REPORT_URL = "CropGuard_AI_Final_Report.docx"   # Word report (.docx)
PRESENTATION_URL = "CropGuard_AI_Final_Presentation.pptx"     # PowerPoint (.pptx)
GITHUB_REPO_URL = "https://github.com/sabooriftikhar/CropGuard-AI-Project"      # GitHub repository


st.set_page_config(
    page_title="CropGuard AI | Plant Leaf Classifier",
    page_icon="🌿",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {font-size: 2.4rem; font-weight: 800; margin-bottom: 0;}
.subhero {color: #8b949e; font-size: 1.05rem; margin-top: .2rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="hero">🌿 CropGuard AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subhero">AI-powered plant leaf disease classification using deep learning</p>',
    unsafe_allow_html=True
)


@st.cache_resource(show_spinner="Loading the CropGuard AI model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.keras is missing. Place it beside app.py.")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_classes():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError("class_names.json is missing. Place it beside app.py.")
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not names:
        raise ValueError("class_names.json must contain a non-empty JSON list.")
    return names


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def prepare_image(image):
    # Matches training: RGB, 224x224, float32, pixel range 0..255.
    image = image.convert("RGB").resize(IMAGE_SIZE)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


def get_predictions(model, image, class_names):
    probabilities = model.predict(prepare_image(image), verbose=0)[0]
    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Model outputs {len(probabilities)} classes, but class_names.json "
            f"contains {len(class_names)}."
        )
    indices = np.argsort(probabilities)[::-1][:min(TOP_K, len(class_names))]
    return [
        {"label": class_names[int(i)], "score": float(probabilities[i])}
        for i in indices
    ]


def pretty_label(label):
    return label.replace("___", " — ").replace("_", " ").strip()


def show_resource(label, url):
    if url.strip():
        st.markdown(f"[{label}]({url})")
    else:
        st.caption(f"{label}: link will be added later.")


try:
    model = load_model()
    class_names = load_classes()
    metrics = load_metrics()
except Exception as error:
    st.error("CropGuard AI could not load its required model files.")
    st.exception(error)
    st.info("Required beside app.py: model.keras and class_names.json.")
    st.stop()


with st.sidebar:
    st.title("🌱 CropGuard AI")
    st.caption("Plant Leaf Disease Classification")
    st.divider()
    st.subheader("Model information")
    st.write("**Architecture:** EfficientNetB0")
    st.write("**Approach:** Transfer learning")
    st.write("**Input:** RGB leaf image")
    st.write("**Image dimensions:** 224 × 224")
    st.write(f"**Number of classes:** {len(class_names)}")
    st.divider()
    st.subheader("Project resources")
    show_resource("Word project report", PROJECT_REPORT_URL)
    show_resource("PowerPoint presentation", PRESENTATION_URL)
    show_resource("GitHub repository", GITHUB_REPO_URL)
    st.divider()
    st.caption("Academic / educational prototype")


with st.expander("Model evaluation on validation dataset", expanded=False):
    if metrics:
        a, b, c, d = st.columns(4)
        accuracy = metrics.get("validation_accuracy")
        loss = metrics.get("validation_loss")
        f1 = metrics.get("macro_f1")
        a.metric("Validation accuracy", f"{float(accuracy)*100:.2f}%" if accuracy is not None else "N/A")
        b.metric("Validation loss", f"{float(loss):.4f}" if loss is not None else "N/A")
        c.metric("Macro F1-score", f"{float(f1):.4f}" if f1 is not None else "N/A")
        d.metric("Classes", str(metrics.get("num_classes", len(class_names))))
        st.caption(
            "Validation metrics measure performance on the project's validation "
            "dataset. They do not guarantee accuracy on real-world field images."
        )
    else:
        st.info("metrics.json was not found; evaluation metrics are unavailable.")


left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1. Upload a leaf image")
    uploaded = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear, well-lit image with the leaf in focus."
    )
    image = None
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded plant leaf", use_container_width=True)
            st.caption(f"Image dimensions: {image.width} × {image.height}")
        except Exception:
            st.error("Could not read this image. Please upload a valid JPG or PNG.")

    if uploaded is None:
        st.info("Upload a plant leaf image to begin.")
        st.markdown(
            "- Keep the leaf clearly visible and in focus.\n"
            "- Prefer natural or bright lighting.\n"
            "- Avoid heavy shadows, filters, and blurry images."
        )

with right:
    st.subheader("2. AI classification")
    if image is not None:
        if st.button("🔍 Analyze leaf", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    st.session_state["cg_results"] = get_predictions(model, image, class_names)
                    st.session_state["cg_filename"] = uploaded.name
                    st.session_state["cg_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                except Exception as error:
                    st.error("Prediction failed.")
                    st.exception(error)

    results = st.session_state.get("cg_results")
    if results:
        best = results[0]
        st.markdown("**Top prediction**")
        st.subheader(pretty_label(best["label"]))
        st.metric("Model confidence", f"{best['score']*100:.2f}%")
        st.progress(float(np.clip(best["score"], 0, 1)))

        st.markdown("**Top predictions**")
        chart = {
            pretty_label(item["label"]): item["score"] * 100
            for item in results
        }
        st.bar_chart(chart, horizontal=True)
        for number, item in enumerate(results, start=1):
            st.write(f"{number}. **{pretty_label(item['label'])}** — {item['score']*100:.2f}%")

        summary = (
            "CropGuard AI - Prediction Summary\n"
            f"Image: {st.session_state.get('cg_filename', 'Uploaded image')}\n"
            f"Date/time: {st.session_state.get('cg_time', 'N/A')}\n\n"
            "Top predictions:\n"
            + "\n".join(
                f"{i}. {pretty_label(item['label'])}: {item['score']*100:.2f}%"
                for i, item in enumerate(results, start=1)
            )
            + "\n\nEducational prototype only; not an agricultural diagnosis."
        )
        st.download_button(
            "⬇️ Download prediction summary",
            data=summary,
            file_name="cropguard_prediction.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.caption("Your prediction will appear here after analysis.")


st.divider()
st.subheader("Important disclaimer")
st.warning(
    "CropGuard AI is an academic and educational image-classification prototype. "
    "It may make incorrect predictions, particularly on field photographs that "
    "differ from its training dataset. It does not confirm a plant disease and "
    "does not provide pesticide, treatment, or crop-management recommendations. "
    "Consult a qualified agricultural expert before making real-world decisions."
)
st.caption("CropGuard AI • EfficientNetB0 • Deep Learning Project")
