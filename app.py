import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import cv2

st.set_page_config(
    page_title="AutoGuard | AI Damage Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .st-emotion-cache-18ni7ap {
        background-color: #F0F2F6;
    }
    .st-emotion-cache-6q9sum {
        background-color: #FFFFFF;
        border-right: 1px solid #E6E6E6;
    }
    h1 {
        font-weight: 700;
        color: #1E1E1E;
    }
    h2, h3 {
        font-weight: 600;
        color: #333333;
    }
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #FF4B4B;
        background-color: #FF4B4B;
        color: white;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        width: 100%;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stButton>button:hover {
        background-color: white;
        color: #FF4B4B;
        border: 1px solid #FF4B4B;
    }
    .stFileUploader {
        border: 2px dashed #CCCCCC;
        border-radius: 8px;
        padding: 2rem;
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

REPAIR_COSTS_PER_INSTANCE = {
    "shatter": 350,
    "scratch": 85,
    "dent": 170,
    "crack": 225,
}

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please ensure the trained model is available at the specified path.")
        st.stop()
    model = YOLO(model_path)
    return model

def analyze_image(model, image_file):
    image = Image.open(image_file)
    image_np = np.array(image)
    results = model(image_np, conf=0.4)
    return image, results

def render_results(analysis_results):
    st.header("Damage Analysis")
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("Damage Detection Overlay")
        annotated_image_bgr = analysis_results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
        st.image(annotated_image_rgb, use_column_width=True)

    with col2:
        st.subheader("Analysis Summary")
        detected_boxes = analysis_results[0].boxes.data.cpu().numpy()
        class_names_map = analysis_results[0].names
        
        detected_classes = [class_names_map[int(obj[5])] for obj in detected_boxes]

        if not detected_classes:
            st.success("No significant damage detected.")
            return

        damage_counts = {cls: detected_classes.count(cls) for cls in set(detected_classes)}
        
        st.markdown("##### Detected Damage Breakdown:")
        total_cost = 0

        for damage_type, count in damage_counts.items():
            cost = REPAIR_COSTS_PER_INSTANCE.get(damage_type, 0) * count
            total_cost += cost
            st.metric(label=f"{damage_type.capitalize()} (x{count})", value=f"${cost:,.2f}")
        
        st.markdown("---")
        st.metric(label="Total Estimated Repair Cost", value=f"${total_cost:,.2f}")

def main_interface():
    with st.sidebar:
        st.title("🛡️ AutoGuard AI")
        st.markdown("Automated vehicle damage assessment using computer vision.")
        st.markdown("---")
        st.info("This tool provides an estimate for repair costs. For an exact quote, please consult a professional body shop.")
    
    st.title("AI-Powered Damage Assessment")
    
    uploaded_file = st.file_uploader(
        "Upload an image of the vehicle damage", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        model = load_model("car_damage_segmentation_v1/yolov8n-seg.pt_e50_b8/weights/best.pt")
        
        with st.spinner('Analyzing image... This may take a moment.'):
            original_image, analysis_results = analyze_image(model, uploaded_file)
        
        render_results(analysis_results)

if __name__ == "__main__":
    main_interface()
