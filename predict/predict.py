import streamlit as st
import requests
from PIL import Image
import io

API_URL = "https://oc-detect-1-2.onrender.com/predict/"

st.title("🦷 Oral Cancer Detection 🦷")
st.write("Upload an image of oral cavity, and the backend model will predict if it is **Normal** or **Cancer**.")

uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    with st.spinner("🔎 Analyzing..."):
        files = {"file": ("uploaded.jpg", img_bytes, "image/jpeg")}
        try:
            response = requests.post(API_URL, files=files)
            result = response.json()

            st.markdown(f"### 🔎 Prediction: **{result['label']}**")
            st.write(f"✅ Confidence: {result['confidence']:.4f}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
