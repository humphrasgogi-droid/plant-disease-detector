import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import json

st.set_page_config(page_title="PlantVillage AI Doctor", page_icon="🌿", layout="centered")

st.title("🌿 Official PlantVillage AI Disease Detector")
st.write("Diagnoses plant diseases across Tomato, Potato, Apple, Corn, Grape, and more.")

# 1. Load labels
@st.cache_data
def get_labels():
    with open("plantvillage_labels.json", "r") as f:
        return json.load(f)

CLASS_NAMES = get_labels()

# 2. Load trained PlantVillage model
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("plantvillage_real_model.keras")

model = load_trained_model()

TREATMENTS = {
    "healthy": "Great news! Your plant is healthy. Maintain regular watering and proper sunlight.",
    "blight": "Prune infected leaves immediately. Avoid overhead watering and treat with a copper fungicide.",
    "scab": "Rake and destroy fallen leaves. Apply protective sulfur spray during early wet periods.",
    "rot": "Prune out diseased parts and ensure soil drains well.",
    "rust": "Remove affected leaves and increase spacing to improve air circulation.",
    "powdery_mildew": "Apply neem oil or a mild baking soda spray (1 tbsp per gallon of water).",
    "bacterial_spot": "Avoid touching foliage while wet. Apply an organic copper-based bactericide.",
    "virus": "Viral diseases cannot be cured chemically. Safely dispose of infected plant to protect others."
}

uploaded_file = st.file_uploader("Upload Leaf Image (Tomato, Potato, Corn, Apple, etc.)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Analyzed Leaf", use_container_width=True)
    
    if st.button("🔍 Diagnose Plant Disease", type="primary"):
        with st.spinner("Analyzing with official PlantVillage Model..."):
            size = (224, 224)
            image_resized = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
            img_array = np.array(image_resized, dtype=np.float32)
            img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            img_batch = np.expand_dims(img_preprocessed, axis=0)
            
            predictions = model.predict(img_batch, verbose=0)
            idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][idx]) * 100
            
            raw_label = CLASS_NAMES[idx]
            
            if "___" in raw_label:
                parts = raw_label.split("___")
                plant = parts[0].replace("_", " ").title()
                disease = parts[1].replace("_", " ").title()
            else:
                plant = raw_label.replace("_", " ").title()
                disease = "Healthy"
                
            st.divider()
            st.subheader("Diagnosis Results")
            st.write(f"### Plant Species: **{plant}**")
            
            if "healthy" in disease.lower():
                st.success(f"🌱 Status: **{disease}** ({confidence:.1f}% confidence)")
            else:
                st.error(f"⚠️ Condition: **{disease}** ({confidence:.1f}% confidence)")
                
            treatment = "Consult local agricultural extension for specialized fungicide recommendations."
            for k, tip in TREATMENTS.items():
                if k in disease.lower():
                    treatment = tip
                    break
                    
            st.info(f"💡 **Recommended Action:** {treatment}")
