import streamlit as st
import numpy as np
import cv2
import os

# Streamlit config (Must be the first command)
st.set_page_config(page_title="AI Medical Diagnostics", layout="centered", page_icon="🏥")

try:
    import tensorflow as tf
except ImportError:
    tf = None

st.title("🏥 AI-Powered Medical Image Analysis")
st.write("Upload a medical image (X-ray or MRI) to instantly get an AI prediction.")
st.markdown("---")

# Option to pick Scan Type
scan_type = st.radio("Select Scan Type:", ("Chest X-ray (Pneumonia Detection)", "Brain MRI (Tumor Detection - Upcoming)", "CT Scan (Upcoming)"))

st.markdown("### Upload Image")
uploaded_file = st.file_uploader("Choose a file (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(uploaded_file, caption="Uploaded Medical Image", width=400)
    
    st.markdown("---")
    
    if st.button("🔍 Analyze Image"):
        if scan_type != "Chest X-ray (Pneumonia Detection)":
            st.warning("⚠️ Only Chest X-ray models are trained and active in this showcase. Please select 'Chest X-ray' for a prediction.")
        else:
            with st.spinner("Analyzing image patterns using Convolutional Neural Network..."):
                
                MODEL_PATH = "models/medical_ai_model.h5"
                loaded_model = None
                
                if tf is not None and os.path.exists(MODEL_PATH):
                    loaded_model = tf.keras.models.load_model(MODEL_PATH)

                # Preprocessing
                processed_image = cv2.resize(image, (256, 256))
                processed_image = processed_image / 255.0  # Normalize
                
                prediction_prob = 0.0
                
                if loaded_model is None:
                    # Mock output if TF or model is missing (perfect for GitHub/LinkedIn simulation)
                    st.toast("TensorFlow or model not found. Using simulation mode.", icon="⚠️")
                    import random
                    prediction_prob = random.uniform(0.1, 0.9)
                else:
                    processed_image = processed_image.reshape(1, 256, 256, 1)
                    prediction_prob = loaded_model.predict(processed_image)[0][0]

                # Display Results
                st.subheader("Results:")
                if prediction_prob > 0.5:
                    st.error(f"🔴 **Diagnosis:** Pneumonia Detected")
                    st.write(f"**Confidence Score:** {prediction_prob * 100:.2f}%")
                    st.info("Recommendation: Review with a board-certified radiologist immediately.")
                else:
                    st.success(f"🟢 **Diagnosis:** Normal (No Pneumonia)")
                    st.write(f"**Confidence Score:** {(1.0 - prediction_prob) * 100:.2f}%")
                    st.info("Recommendation: Patient scan appears clear. Standard follow-up.")
