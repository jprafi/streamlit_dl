# Face Recognition for Class Attendance (VGGFace + Streamlit)

This repository contains a face recognition system for **student identification and attendance**, built using:

- **VGGFace ResNet50** (transfer learning, end-to-end classifier)
- **MTCNN** for face detection
- **Streamlit** for a simple web-based interface

🔗 **Live demo:** https://rtk-project.streamlit.app/  
🔗 **GitHub repo:** https://github.com/jprafi/streamlit_dl  

---

## 1. Team

- **Alief Fathur Rahman** – 122140027  
- **JP Rafi Radiktya Arkan** – 122140169  
- **Desty Ananta Purba** – 122140076  

Course: **Pembelajaran Mendalam (Deep Learning)** – Tugas Besar

---

## 2. What This Project Does

Singkatnya, aplikasi ini melakukan:

1. Menerima input gambar wajah (upload file).
2. Mencari wajah menggunakan **MTCNN**.
3. Memotong area wajah dan mengubah ukurannya menjadi **224×224** piksel.
4. Melakukan preprocessing dengan fungsi `preprocess_input` khusus **VGGFace (version=2)**.
5. Mengirim hasil preprocessing ke model **VGGFace ResNet50** yang sudah di-finetune pada dataset mahasiswa satu kelas.
6. Menghasilkan:
   - Nama identitas yang diprediksi
   - Nilai confidence
   - Top-5 prediksi terbaik
   - Label **UNKNOWN** jika confidence terlalu rendah

Tujuan utamanya adalah **prototipe absensi otomatis berbasis wajah** untuk satu kelas dengan jumlah mahasiswa terbatas.

---

## 3. Main Features

Beberapa fitur penting:

- **Transfer Learning dengan VGGFace**
  - Menggunakan backbone ResNet50 yang sudah dilatih di dataset wajah besar, lalu di-finetune ke 70 identitas mahasiswa.
  - Keluaran berupa softmax probability untuk setiap kelas.

- **Deteksi Wajah Otomatis (MTCNN)**
  - Sistem tidak bergantung pada gambar yang sudah di-crop.
  - MTCNN digunakan untuk mendeteksi bounding box wajah, kemudian wajah di-crop otomatis.

- **Threshold “UNKNOWN”**
  - Jika confidence tertinggi < threshold (misalnya 0.5), wajah dianggap **tidak dikenal (UNKNOWN)**.
  - Membantu mengurangi salah pengenalan untuk wajah yang bukan bagian dataset.

- **Visualisasi di Web**
  - Menampilkan:
    - Gambar asli dengan bounding box
    - Wajah yang sudah dicrop
    - Top-5 prediksi dengan grafik bar
    - Daftar kelas yang tersedia

---

## 4. Model & Training Summary (VGGFace)

Model utama yang digunakan untuk **versi Streamlit**:

- **Backbone:** VGGFace ResNet50 (`keras-vggface`)
- **Input size:** 224 × 224 × 3 (RGB)
- **Preprocessing:** `preprocess_input(img, version=2)`
- **Head classifier:**
  - Global Average Pooling (dari backbone)
  - Dense(512, ReLU) + Dropout
  - Dense(num_classes, softmax)

Dataset (garis besar):

- **Identitas:** 70 mahasiswa
- **Citra per kelas:** beberapa foto per orang
- **Augmentasi:**
  - Random horizontal flip
  - Random rotation
  - Random zoom
- **Split:**
  - Train set dan test set per identitas
  - Evaluasi menggunakan accuracy, classification report, dan confusion matrix

Model yang sudah jadi disimpan dalam format:

- `vgg_model.h5` – bobot model + arsitektur
- `class_names.npy` – daftar nama kelas (urutan sama dengan output softmax)
- `vgg_config.json` – konfigurasi (misalnya `image_size`, `version`, dll.)

---

## 5. Alternative Approach: FaceNet + SVM / KNN

Selain VGGFace, kami juga melakukan eksperimen dengan pendekatan **dua tahap**:

1. **FaceNet (Embedding)**
   - Menggunakan model face embedding (FaceNet / InceptionResnetV1 atau sejenis) untuk mengubah citra wajah menjadi vektor fitur berdimensi tetap (misalnya 512 dimensi).
   - Setiap gambar wajah → embedding di ruang fitur yang “face-aware”.

2. **Klasifikasi Tradisional (SVM / KNN)**
   - Embedding yang dihasilkan FaceNet digunakan sebagai input ke classifier klasik:
     - **SVM (Support Vector Machine)** untuk decision boundary yang lebih tajam.
     - **KNN (k-Nearest Neighbors)** sebagai baseline sederhana berbasis kemiripan.
   - Model classifier ini disimpan dalam bentuk file **`.pkl`** (pickle), beserta mapping label (misalnya `metadata.json` / `label_map.json`).

Karakteristik pendekatan ini:

- Memisahkan **feature extractor** (FaceNet) dan **classifier** (SVM/KNN).
- Lebih fleksibel jika ingin:
  - Menambah identitas baru tanpa retrain full deep network.
  - Mengganti jenis classifier dengan mudah.

Namun, untuk keperluan deploy sederhana di Streamlit, kami memilih pendekatan **VGGFace end-to-end (hanya satu file .h5 + metadata)** agar integrasi dan loading di server lebih praktis.

---

## 6. How to Use the Online App

Kamu tidak perlu setup environment untuk mencoba. Cukup:

1. Buka: **https://rtk-project.streamlit.app/**
2. Upload foto wajah yang jelas (1 orang dalam satu gambar lebih aman).
3. Klik **"Recognize Face"**.
4. Lihat hasil:
   - Nama mahasiswa yang dikenali (atau **UNKNOWN**)
   - Confidence score
   - Top-5 prediksi
   - Bounding box wajah yang terdeteksi

> Catatan: Sistem ini dilatih khusus pada wajah mahasiswa tertentu. Jika wajah di luar daftar kelas, kemungkinan akan berstatus UNKNOWN atau salah prediksi.

---

## 7. Running Locally (Development)

Jika ingin menjalankan di lokal:

```bash
# 1. Clone repository
git clone https://github.com/jprafi/streamlit_dl.git
cd streamlit_dl

# 2. (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan aplikasi
streamlit run streamlit_app.py
