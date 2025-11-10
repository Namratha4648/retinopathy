import streamlit as st
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
import pandas as pd

st.set_page_config(page_title="Diabetic Retinopathy AI", layout="wide", initial_sidebar_state="auto")

# --- STYLING ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #f0f2f6;
    }
    h1, h2 {
        color: #1E3A8A; /* Dark Blue */
        text-align: center;
    }
    .st-emotion-cache-1v0mbdj {
        border-radius: 10px;
        padding: 2rem;
        background-color: #fff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #E0F2FE; /* Light Blue */
        border-left: 5px solid #3B82F6; /* Blue */
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Model and Classes ---
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
DIAGNOSIS_DETAILS = {
    "No DR": "No signs of diabetic retinopathy were detected.",
    "Mild": "Early stage of DR, characterized by small areas of swelling in the retina's blood vessels (microaneurysms).",
    "Moderate": "More advanced than mild DR, with more significant damage to blood vessels, including potential blockages.",
    "Severe": "Significant blockage of blood vessels, leading to decreased blood supply to areas of the retina. The body signals for new blood vessels to grow.",
    "Proliferative DR": "The most advanced stage. New, abnormal blood vessels grow on the retina. These fragile vessels can leak blood, causing severe vision loss or blindness."
}

# --- Build Absolute Path to Model ---
# This makes the app runnable from any directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "models", "best_model_finetuned.pth")
IMG_SIZE = 260 # EfficientNet-B2 expects 260x260

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Image Transformations ---
val_transforms = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Please ensure the training notebook has been run successfully.")
        return None
    try:
        model_instance = EfficientNet.from_pretrained('efficientnet-b2', num_classes=5)
        model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model_instance.to(device)
        model_instance.eval()
        return model_instance
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

# --- Grad-CAM Implementation ---
def get_grad_cam_and_preds(model_instance, img_tensor, original_img, target_layer_name="_conv_head"):
    """Generates Grad-CAM, prediction index, and probabilities."""
    if model_instance is None: return None, None, None
    gradients, activations = [], []

    def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0].detach())
    def forward_hook(module, input, output): activations.append(output.detach())

    try:
        target_layer = dict([*model_instance.named_modules()])[target_layer_name]
    except KeyError:
        st.warning(f"Target layer '{target_layer_name}' not found. Grad-CAM cannot be generated.")
        return None, None, None

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model_instance(img_tensor)
    probabilities = F.softmax(output, dim=1)
    pred_class_idx = torch.argmax(probabilities, dim=1).item()
    
    model_instance.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, pred_class_idx] = 1
    output.backward(gradient=one_hot, retain_graph=True)

    forward_handle.remove(); backward_handle.remove()

    pooled_gradients = torch.mean(gradients[0], dim=[2, 3], keepdim=True)
    cam = torch.sum(pooled_gradients * activations[0], dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, (original_img.shape[0], original_img.shape[1]), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    return overlay, pred_class_idx, probabilities.squeeze().detach().cpu().numpy()

# --- UI Layout ---
st.title("👁️ AI Diabetic Retinopathy Analyzer")

if model is None:
    st.warning("The model is not loaded. Please ensure the training notebook has generated the model file correctly.")
else:
    st.sidebar.title("About the Project")
    st.sidebar.info(
        "This application uses an **EfficientNet-B2** deep learning model to predict the stage of Diabetic Retinopathy from a fundus image. "
        "It also displays a **Grad-CAM heatmap** to visualize which parts of the image were most important for the model's prediction, offering a degree of 'explainability'."
    )
    st.sidebar.title("Instructions")
    st.sidebar.markdown("1. **Upload an Image**: Use the uploader in the main panel.\n2. **Get Prediction**: The model automatically classifies the image.\n3. **View Analysis**: See the diagnosis, confidence scores, and AI heatmap.")
    st.sidebar.title("Model Performance")
    st.sidebar.metric("Final QWK Score", "0.842")


    uploaded_file = st.file_uploader("Upload a Retina Image to Begin Analysis", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        
        with col1:
            st.image(opencv_image, caption='Uploaded Image', use_container_width=True)

        with st.spinner('🧠 AI is analyzing the image...'):
            transformed = val_transforms(image=opencv_image)
            img_tensor = transformed["image"].unsqueeze(0).to(device)
            overlay, pred_class_idx, probabilities = get_grad_cam_and_preds(model, img_tensor, opencv_image)

        if overlay is not None and pred_class_idx is not None:
            prediction = CLASS_NAMES[pred_class_idx]
            
            with col1:
                st.image(overlay, caption='AI Explainability (Grad-CAM)', use_container_width=True)
            
            with col2:
                st.metric("Predicted Diagnosis", prediction)
                
                st.subheader("Diagnosis Details")
                st.info(DIAGNOSIS_DETAILS[prediction])

                st.subheader("Prediction Confidence")
                prob_df = pd.DataFrame(probabilities, index=CLASS_NAMES, columns=["Confidence"])
                prob_df.index.name = "Diagnosis Category"
                st.bar_chart(prob_df)

        else:
            st.error("Could not generate a prediction or heatmap for this image.")
    else:
        st.info("Please upload an image to get started.")