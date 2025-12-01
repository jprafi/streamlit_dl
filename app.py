import streamlit as st
import numpy as np
import cv2
import pickle
import json

from utils.face_detector import detect_and_crop
from utils.facenet_embedder import get_embedding

# -----------------------
# LOAD MODEL & METADATA
# -----------------------
with open("model/best_classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("model/metadata.json", "r") as f:
    metadata = json.load(f)

idx_to_class = {int(k): v for k, v in metadata["idx_to_class"].items()}

# -----------------------
# STREAMLIT UI
# -----------------------
st.title("🎓 Student Attendance - Face Recognition")
st.write("Upload image wajah, dan sistem akan mengenali identitas mahasiswa.")

uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded:
    # Convert to cv2 image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(rgb, caption="Input Image", use_container_width=True)

    # ===========================================
    # LOADING UI SAAT PREDIKSI
    # ===========================================
    with st.spinner("🔍 Mendeteksi wajah dan melakukan prediksi..."):
        face = detect_and_crop(rgb)

        if face is None:
            st.error("Tidak ditemukan wajah pada gambar.")
        else:
            embedding = get_embedding(face)
            embedding = embedding.reshape(1, -1)

            pred = classifier.predict(embedding)[0]

            if hasattr(classifier, "predict_proba"):
                prob = classifier.predict_proba(embedding)[0][pred]
            else:
                prob = 1.0

            name = idx_to_class[pred]

            st.subheader("📌 Hasil Prediksi")
            st.write(f"Nama: **{name}**")
            st.write(f"Confidence: **{prob:.2%}**")

            st.image(face, caption="Detected Face", width=200)

    st.success("Prediksi selesai!")
