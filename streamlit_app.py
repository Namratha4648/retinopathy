import streamlit as st
import streamlit as st
from PIL import Image
import numpy as np
import os
import io
import pandas as pd
import time
import urllib.request
import shutil

from model_utils import load_model, predict_and_explain, CLASS_NAMES
from model_utils import load_model, predict_and_explain, CLASS_NAMES

st.set_page_config(page_title="Diabetic Retinopathy — Demo", layout="wide")

st.title("Diabetic Retinopathy Detection — Demo")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value="models/best_model_finetuned.pth")
    device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
    backend = st.selectbox("Model backend", options=["timm", "efficientnet_pytorch"], index=1)
    # Optional URL to fetch the model if it's missing (useful on Streamlit Cloud)
    default_url = ""
    try:
        default_url = st.secrets.get("MODEL_URL", "")
    except Exception:
        default_url = ""
    model_url = st.text_input("Model URL (used if file missing)", value=default_url, help="Provide a direct download URL to the .pth file or set MODEL_URL in Secrets")
    target_layer_choice = st.selectbox("Target layer for Grad-CAM", options=["conv_head", "last_block"], index=1)
    st.markdown("---")
    st.markdown("Upload an image (PNG/JPG) or choose a sample from `data/train_images`.")

col1, col2 = st.columns([1, 1])

uploaded = col1.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

sample_dir = os.path.join("data", "train_images")
sample_files = []
if os.path.isdir(sample_dir):
    sample_files = sorted(os.listdir(sample_dir))[:200]

sample_choice = None
if sample_files:
    sample_choice = col1.selectbox("Or pick a sample image", options=["—"] + sample_files)

if uploaded is None and (not sample_choice or sample_choice == "—"):
    col1.info("Upload an image or pick a sample to enable prediction.")

# Helper: download model file if missing (for cloud deploys)
def _ensure_model_file(path: str, url: str | None = None):
    if os.path.exists(path):
        return True
    if not url:
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with st.spinner("Downloading model…"):
            with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as f:
                shutil.copyfileobj(r, f)
        return True
    except Exception as e:
        st.sidebar.error(f"Model download failed: {e}")
        return False

# Load model lazily
@st.cache_resource(show_spinner=False)
def _load(path, device_str, backend):
    return load_model(path, device_str, backend=backend)

model, model_device = None, None
try:
    # If model file missing (common on Streamlit Cloud), try to fetch it
    if not os.path.exists(model_path):
        _ = _ensure_model_file(model_path, model_url)
    model, model_device = _load(model_path, device, backend)
except Exception as e:
    st.sidebar.error(f"Model load failed: {e}")

def read_image_from_upload(uploaded_file):
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    return image

image = None
image_name = None
if uploaded is not None:
    image = read_image_from_upload(uploaded)
    image_name = uploaded.name
elif sample_choice and sample_choice != "—":
    image_path = os.path.join(sample_dir, sample_choice)
    try:
        image = Image.open(image_path).convert("RGB")
        image_name = sample_choice
    except Exception as e:
        col1.error(f"Could not open sample image: {e}")

if image is not None:
    # use_column_width deprecated; replaced with use_container_width
    col1.image(image, caption=image_name, use_container_width=True)

    if st.button("Predict & Explain"):
        if model is None:
            st.error("Model not loaded. Check path in sidebar.")
        else:
            st.info("Running model — this may take a few seconds on CPU")
            target_layer = target_layer_choice
            preds, probs, heatmap, overlay = predict_and_explain(image, model, model_device, target_layer=target_layer)

            # Left: probabilities and class
            with col2:
                st.subheader("Prediction")
                pred_idx = int(preds)
                st.metric("Predicted class", f"{CLASS_NAMES[pred_idx]} ({pred_idx})")

                st.subheader("Probabilities")
                prob_list = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
                st.table(prob_list.items())

                # Download overlay
                buf = io.BytesIO()
                overlay.save(buf, format="PNG")
                buf.seek(0)
                st.download_button("Download overlay PNG", data=buf, file_name=f"overlay_{image_name}")

            # Show heatmap + overlay
            st.subheader("Grad-CAM Explainability")
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(heatmap, caption="Heatmap", use_container_width=True)
            with col_b:
                st.image(overlay, caption="Overlay", use_container_width=True)

            st.success("Done — inspect the prediction and the Grad-CAM overlay.")

st.markdown("---")
st.markdown("#### Notes")
st.markdown("---")
st.header("Batch inference")
with st.expander("Run batch inference on a folder of images"):
    batch_folder = st.text_input("Image folder path (local)", value=os.path.join("data", "test_images"))
    out_folder = st.text_input("Output overlay folder", value=os.path.join("models", "overlays"))
    max_images = st.number_input("Max images to process (0 = all)", min_value=0, value=0)

    if st.button("Run batch inference"):
        if model is None:
            st.error("Model not loaded. Check path in sidebar.")
        else:
            if not os.path.isdir(batch_folder):
                st.error(f"Folder not found: {batch_folder}")
            else:
                os.makedirs(out_folder, exist_ok=True)
                files = [f for f in sorted(os.listdir(batch_folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if max_images > 0:
                    files = files[:int(max_images)]
                st.info(f"Found {len(files)} images — processing...")

                progress = st.progress(0)
                results = []
                for i, fname in enumerate(files):
                    p = os.path.join(batch_folder, fname)
                    try:
                        img = Image.open(p).convert('RGB')
                    except Exception as e:
                        results.append({"filename": fname, "error": str(e)})
                        continue

                    pred, probs, heatmap, overlay = predict_and_explain(img, model, model_device, target_layer=target_layer_choice)

                    overlay_name = f"overlay_{fname}"
                    overlay_path = os.path.join(out_folder, overlay_name)
                    overlay.save(overlay_path)

                    row = {"filename": fname, "pred": int(pred)}
                    for k, v in enumerate(probs):
                        row[f"prob_{k}"] = float(v)
                    row["overlay_path"] = overlay_path
                    results.append(row)

                    progress.progress(int((i + 1) / max(1, len(files)) * 100))
                    # small sleep to keep UI responsive for many files
                    time.sleep(0.01)

                df = pd.DataFrame(results)
                st.success("Batch inference complete")
                st.write(df.head(20))

                # CSV download
                csv_buf = io.StringIO()
                df.to_csv(csv_buf, index=False)
                csv_bytes = csv_buf.getvalue().encode('utf-8')
                st.download_button("Download CSV results", data=csv_bytes, file_name="batch_results.csv")

                st.markdown(f"Overlays saved to: `{out_folder}`")
st.markdown("- The app expects a TimM / EfficientNet-B2 style model saved at `models/best_model_finetuned.pth` (same as notebook).\n- Grad-CAM may be noisy for low-res images; try 224x224-sized images.")
