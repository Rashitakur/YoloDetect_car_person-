import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

# Page config
st.set_page_config(page_title="YOLOv8 Car & Person Detector", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .title {
        color: #0e1117;
        text-align: center;
        font-size: 40px;
    }
    .info {
        background-color: #eef6ff;
        padding: 15px;
        border-radius: 10px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
st.sidebar.title("🔍 YOLOv8 Info")
st.sidebar.markdown("""
**YOLOv8 (You Only Look Once)** is a real-time object detection model developed by **Ultralytics**.

It can detect multiple classes such as:

- 🚗 Cars
- 🧍 Persons

This app demonstrates detection using a **custom-trained YOLOv8 model** on a Car-Person dataset.
""")

# Title
st.markdown('<div class="title">🚗👤 Car & Person Detection using YOLOv8</div>', unsafe_allow_html=True)

# Description block
st.markdown("""
<div class="info">
This demo lets you upload an image and instantly run object detection on it using a custom-trained YOLOv8 model. The model will identify and highlight objects like <b>cars</b> and <b>persons</b> in the image.
</div>
""", unsafe_allow_html=True)

# Load model
model = YOLO(r"C:\Users\HP\DETECTION\DETECTION\Yolo detect\best.pt")


# Upload section
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Convert to NumPy array
    image_np = np.array(image)

    # Run inference
    results = model(image_np)

    # Get annotated image with boxes
    annotated_img = results[0].plot()

    # Show results
    st.image(annotated_img, caption="✅ Detected Objects", use_container_width=True)

    # Download button
    st.download_button(
        label="⬇️ Download Result Image",
        data=cv2.imencode('.jpg', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))[1].tobytes(),
        file_name='result.jpg',
        mime='image/jpeg'
    )

