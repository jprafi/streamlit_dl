<div align="center">

# 👤 Face Recognition for Class Attendance  
### Deep Learning–based Face Recognition with VGGFace & Streamlit

🔴 **Live Demo:**  
👉 https://rtk-project.streamlit.app/

📦 **Repository:**  
👉 https://github.com/jprafi/streamlit_dl  

---

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Used-orange?logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Deployed-success)
![License](https://img.shields.io/badge/License-Academic-green)

</div>

---

## 👥 Team Members

| Name                     | NIM        | Role (simplified)         |
|--------------------------|-----------:|---------------------------|
| **Alief Fathur Rahman** | 122140027  | Model & Deployment        |
| **JP Rafi Radiktya Arkan** | 122140169 | Backend & Integration     |
| **Desty Ananta Purba**  | 122140076  | Dataset & Evaluation      |

> 🎓 _Mata Kuliah_: Pembelajaran Mendalam (Deep Learning)  
> 🧪 _Tugas_: Proyek akhir – Face Recognition untuk sistem absensi kelas

---

## 🎯 Project Snapshot

Satu kalimat:  
> **Aplikasi web sederhana untuk mengenali wajah mahasiswa dan memprediksi identitasnya secara otomatis berbasis VGGFace ResNet50.**

Apa yang bisa dilakukan aplikasi ini?

- ✅ Upload foto wajah → langsung dikenali (jika termasuk dalam kelas yang dilatih)
- ✅ Deteksi wajah otomatis dengan **MTCNN**
- ✅ Crop wajah + resize → 224×224 piksel
- ✅ Klasifikasi menggunakan **VGGFace ResNet50** yang sudah di-finetune
- ✅ Menampilkan:
  - Nama mahasiswa yang dikenali
  - Confidence score
  - Top-5 prediksi
  - Label **UNKNOWN** bila confidence di bawah threshold

---

## 🧠 Model Variants in This Project

Walaupun yang **dideploy di Streamlit** adalah model VGGFace, proyek ini sebenarnya punya **dua jalur eksperimen**:

### 1️⃣ VGGFace ResNet50 (End-to-End Classifier) – *Deployed Version*

- Backbone: **VGGFace ResNet50** (`keras-vggface`)
- Input: `224 × 224 × 3` (RGB)
- Preprocessing: `preprocess_input(img, version=2)`
- Head classifier:
  - Global Average Pooling (dari backbone)
  - Dense(512, ReLU) + Dropout
  - Dense(num_classes, Softmax)
- Output: probabilitas untuk setiap mahasiswa (70 kelas)

### 2️⃣ FaceNet + SVM / KNN (Two-Stage Pipeline) – *Experimental*

- Feature extractor: **FaceNet / embedding model**  
  👉 Mengubah wajah → vektor fitur berdimensi tetap (misal 512-dim)
- Classifier klasik:
  - **SVM** → decision boundary tajam
  - **KNN** → berbasis kemiripan embedding
- Model klasifikasi disimpan sebagai:
  - `classifier.pkl` (SVM/KNN)
  - `label_map` / metadata terpisah

Pendekatan FaceNet+SVM/KNN ini fleksibel untuk:
- Menambah identitas baru dengan retrain classifier saja
- Eksperimen berbagai algoritma klasik di atas embedding

Namun untuk **deployment sederhana**, kami memilih **VGGFace end-to-end** (satu file `.h5` + metadata) karena:

- Lebih mudah di-load di Streamlit Cloud  
- Struktur pipeline lebih ringkas

---

## 🔍 How the VGGFace Pipeline Works

Urutan proses di aplikasi (versi VGGFace + Streamlit):

1. **Upload Gambar**
   - Pengguna memilih file (`.jpg`, `.png`, dll)
   - Dibaca sebagai `PIL.Image` lalu dikonversi ke NumPy array (RGB)

2. **Face Detection – MTCNN**
   - `detector.detect_faces(image)` mendeteksi bounding box
   - Jika banyak wajah → diambil wajah dengan area terbesar

3. **Crop & Resize**
   - Region wajah dipotong dari gambar
   - Diubah ukurannya menjadi `224 × 224` piksel

4. **Preprocessing (VGGFace)**
   - Menggunakan `preprocess_input(..., version=2)`  
     (sesuai standar VGGFace ResNet50)

5. **Prediction**
   - Dikirim ke `model.predict()` → output: vektor softmax
   - Diambil:
     - Top-1 kelas beserta confidence
     - Top-5 prediksi terbaik untuk visualisasi

6. **Threshold & UNKNOWN**
   - Jika confidence top-1 < threshold (misal 0.5) → label **UNKNOWN**
   - Membantu menghindari “maksa ngaku kenal” untuk wajah yang sebenarnya tidak ada di dataset

7. **Display di Streamlit**
   - Gambar asli dengan bounding box
   - Wajah hasil crop
   - Info nama + confidence
   - Grafik bar **Top-5 Predictions**

---

## 📊 VGGFace Training Summary

Beberapa poin penting dari proses training:

- **Dataset**
  - ~70 identitas (mahasiswa)
  - Beberapa gambar per orang, berpose & kondisi cahaya berbeda

- **Augmentasi**
  - Random horizontal flip
  - Random rotation
  - Random zoom
  - Tujuan: membuat model lebih tahan terhadap variasi pose & pencahayaan

- **Training Setup**
  - Loss: `categorical_crossentropy`
  - Optimizer: `Adam (lr=1e-3 → di-reduce on plateau)`
  - Callbacks:
    - `EarlyStopping` (monitor `val_loss`)
    - `ReduceLROnPlateau`

- **Evaluasi**
  - Akurasi test tinggi (> 98% di subset uji)
  - Menggunakan:
    - `classification_report`
    - confusion matrix
    - per-class accuracy

Model akhir disimpan sebagai:

- `vgg_model.h5` – arsitektur + bobot
- `class_names.npy` – list nama kelas (urutan = indeks softmax)
- `vgg_config.json` – konfigurasi dasar (`image_size`, `version`, dll.)

---

## 🌐 Try the Web App

Tanpa install apa pun, kamu bisa langsung coba di browser:

> 🔗 **https://rtk-project.streamlit.app/**

Langkah singkat:

1. Buka link di atas.
2. Upload foto wajah (sebisa mungkin:
   - 1 orang per gambar
   - wajah menghadap kamera
   - pencahayaan cukup jelas)
3. Klik **“Recognize Face”**
4. Lihat hasil:
   - Nama yang diprediksi
   - Confidence
   - Status **Known / Unknown**
   - Grafik Top-5 Predictions

> ⚠️ Catatan: Model ini dilatih khusus pada dataset satu kelas.  
> Wajah di luar daftar mahasiswa kemungkinan besar akan terdeteksi sebagai **UNKNOWN** atau salah prediksi.

---

## 💻 Run Locally (Development Mode)

Jika ingin menjalankan di laptop sendiri:

```bash
# 1. Clone repository
git clone https://github.com/jprafi/streamlit_dl.git
cd streamlit_dl

# 2. (Opsional) Buat virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan aplikasi Streamlit
streamlit run streamlit_app.py
