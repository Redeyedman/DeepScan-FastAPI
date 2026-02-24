import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import os

st.set_page_config(
    page_title="DeepScan Pro | Crack Analyzer",
    page_icon="🏗️",
    layout="wide"
)

st.markdown("""
    <style>
    /* MAIN BACKGROUND */
.stApp {
    background: radial-gradient(circle at top,
        #1e293b 0%,
        #0f172a 40%,
        #020617 100%);
    color: #e5e7eb;
}

    /* LIMIT ALL IMAGES SIZE */
img {
    max-height: 260px;      /* 👈 adjust: 220–300 */
    object-fit: contain;
}
    /* Custom Card Style */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #3498db;
    }

    /* Titles */
    h1 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "resnet_sdnet_final.pth"
@st.cache_resource
def load_model():
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE).eval()
        return model
    return None
model = load_model()

def analyze_image(image_pil):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)
        res = "Cracked" if pred.item() == 1 else "Uncracked"
        confidence = conf.item() * 100

    sev, ori, heat_map = "N/A", "N/A", None

    if res == "Cracked":
        img_cv = np.array(image_pil.convert('L'))
        img_blur = cv2.GaussianBlur(img_cv, (5, 5), 0)
        _, thresh = cv2.threshold(img_blur, 120, 255, cv2.THRESH_BINARY_INV)

        pixels = np.sum(thresh == 255)
        ratio = (pixels / thresh.size) * 100
        sev = "High Risk" if ratio > 5 else ("Medium" if ratio > 2 else "Minor")

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            ori = "Horizontal" if w > h else "Vertical"
        heat_map = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)

    return res, sev, ori, confidence, thresh if res == "Cracked" else None

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4342/4342728.png", width=100)
    st.title("Control Panel")
    st.info("System Status: Online 🟢")
    st.write(f"Running on: **{DEVICE}**")
    st.divider()
    st.caption("Instructions: Upload clear top-down images of concrete surfaces for best accuracy.")

st.title("🏗️ DeepScan Pro: Structural Integrity AI")
st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📤 Input Image")
    uploaded_file = st.file_uploader("Drop image here...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Specimen", use_container_width=True)

with col_right:
    st.subheader("📊 Analysis Results")

    if uploaded_file and model:
        with st.spinner('Running Deep Neural Analysis...'):
            res, sev, ori, conf, mask = analyze_image(image)

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Status", res, delta="Danger" if res == "Cracked" else "Safe", delta_color="inverse")
            with m_col2:
                st.metric("Confidence", f"{conf:.2f}%")

            st.divider()

            if res == "Cracked":
                st.warning(f"**Structural Anomaly Detected!**")
                data = {
                    "Parameter": ["Severity Level", "Crack Orientation", "Pixel Density Ratio"],
                    "Value": [sev, ori, "Calculated via CV2"]
                }
                st.table(data)

                st.subheader("🔍 Segmentation Map")
                st.image(mask, caption="Detected Crack Patterns", use_container_width=True, clamp=True)
            else:
                st.success("✅ No significant cracks detected in this specimen.")
                st.balloons()
    else:
        st.info("Waiting for image upload...")

st.markdown("---")
st.caption("© 2026 DeepScan AI - Structural Engineering Division")