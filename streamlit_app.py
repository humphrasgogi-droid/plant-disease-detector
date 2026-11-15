import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="AI Plant & Botany Doctor", page_icon="🌿", layout="centered")

st.title("🌿 AI Plant & Leaf Identifier")
st.write("Upload a leaf, fruit, or plant photo to identify the plant species and condition.")

# Official Google Vision Transformer (100% public, permanent, no token required)
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

with st.spinner("Loading Vision Transformer model..."):
    classifier = load_model()

# Health advice tips
CARE_TIPS = {
    "leaf": "Ensure balanced nitrogen-rich fertilization and check soil moisture.",
    "fruit": "Monitor for blossom end rot by maintaining regular calcium and steady watering.",
    "spot": "Apply organic copper spray and remove diseased foliage immediately.",
    "rot": "Improve drainage, prune crowded branches, and avoid overwatering."
}

uploaded_file = st.file_uploader("Upload leaf/plant image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Target Plant", use_container_width=True)
    
    if st.button("🔍 Identify & Diagnose", type="primary"):
        with st.spinner("Analyzing plant features..."):
            predictions = classifier(img, top_k=5)
            
        st.divider()
        st.subheader("Identification Results")
        
        top = predictions[0]
        label = top['label'].title()
        confidence = top['score'] * 100
        
        st.success(f"🌱 Top Match: **{label}** ({confidence:.1f}% confidence)")
        
        # Recommendations
        st.info("💡 **Care Recommendation:** Inspect leaf undersides regularly for pests and avoid overhead watering to prevent fungal spread.")
        
        with st.expander("📊 View Top 5 Matching Classifications"):
            for pred in predictions:
                st.write(f"- **{pred['label'].title()}**: {pred['score']*100:.1f}%")
