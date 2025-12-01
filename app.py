import streamlit as st
import numpy as np
import cv2
import pickle
import json

from utils.face_detector import detect_and_crop
from utils.facenet_embedder import get_embedding

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Student Face Recognition",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ================================
# LOAD MODEL & METADATA
# ================================
with open("model/best_classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("model/metadata.json", "r") as f:
    metadata = json.load(f)

idx_to_class = {int(k): v for k, v in metadata["idx_to_class"].items()}

# ================================
# CUSTOM CSS (CARD & STYLE)
# ================================
st.markdown("""
    <style>
    body {
        background-color: #f7f9fc;
    }
    .upload-box {
        border: 2px dashed #6c63ff;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        background-color: white;
    }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.1);
        text-align: center;
    }
    .face-preview {
        border-radius: 12px;
        border: 3px solid #6c63ff;
    }
    </style>
""", unsafe_allow_html=True)


# ================================
# HEADER
# ================================
st.markdown("""
    <h1 style='text-align:center; color:#333;'>
        🎓 Student Attendance — Face Recognition
    </h1>
    <p style='text-align:center; font-size:17px; color:#555;'>
        Unggah foto mahasiswa, dan sistem akan mengidentifikasi wajah secara otomatis.
    </p>
""", unsafe_allow_html=True)

st.write("---")


# ================================
# UPLOADER SECTION
# ================================
st.markdown("<div class='upload-box'>📤 <b>Upload Foto Wajah</b></div>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded:

    # Convert image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.write("### 📸 Foto yang Diunggah")
    st.image(rgb, use_container_width=True)

    # ===========================================
    # LOADING UI SAAT PREDIKSI
    # ===========================================
    with st.spinner("🔍 Mendeteksi wajah & menghitung embedding..."):
        face = detect_and_crop(rgb)

        if face is None:
            st.error("❌ Tidak ditemukan wajah pada gambar.")
        else:
            embedding = get_embedding(face).reshape(1, -1)

            pred = classifier.predict(embedding)[0]
            prob = classifier.predict_proba(embedding)[0][pred] if hasattr(classifier, "predict_proba") else 1.0

            name = idx_to_class[pred]

            st.write("---")
            st.markdown("## 🎯 Hasil Prediksi")

            # ================================
            # RESULT CARDS
            # ================================
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"""
                    <div class='result-card'>
                        <h3>👤 Identitas</h3>
                        <p style='font-size:23px; font-weight:bold; color:#6c63ff;'>{name}</p>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div class='result-card'>
                        <h3>📊 Confidence</h3>
                        <p style='font-size:23px; font-weight:bold; color:#28a745;'>{prob:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
                <h4 style='text-align:center; color:#555;'>🪪 Wajah yang Terdeteksi</h4>
            """, unsafe_allow_html=True)
            st.image(face, caption="Detected Face", width=250, output_format="PNG")

    st.success("✨ Prediksi selesai.")
else:
    st.info("📌 Silakan upload foto untuk mulai melakukan prediksi.")
