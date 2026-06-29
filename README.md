# ☕ Coffee Bean Quality Classifier

> **Klasifikasi Mutu Biji Kopi Mentah** menggunakan Ekstraksi Fitur Morfologi & Geometri dengan pendekatan *Rule-Based* dan *Machine Learning (Random Forest)*.

---

## 📌 Tentang Proyek

Aplikasi berbasis **Streamlit** yang mampu menganalisis citra biji kopi mentah (*green bean*) secara otomatis dan mengklasifikasikannya ke dalam **5 grade kualitas** berdasarkan fitur morfologi dan warna. Proyek ini mengimplementasikan pipeline lengkap pemrosesan citra digital — mulai dari sampling, filtering, morfologi, deteksi tepi, segmentasi, hingga *unsupervised learning*.

---

## 🎯 Kelas & Grade yang Didukung

| Grade | Kelas | Deskripsi |
|-------|-------|-----------|
| 1 | **Normal** | Biji sempurna, tidak ada cacat |
| 2 | **Withered** | Biji layu / keriput |
| 3 | **Partial Sour** | Asam sebagian |
| 4 | **Broken** | Biji pecah |
| 4 | **Dry Cherry** | Masih terbungkus kulit kering |
| 5 | **Severe Insect Damage** | Kerusakan parah akibat hama serangga |

---

## 🧠 Dua Metode Klasifikasi

### 1. Rule-Based (Threshold Manual)
Klasifikasi berbasis aturan yang dibuat dari analisis distribusi fitur nyata di dataset. Cocok digunakan tanpa perlu training terlebih dahulu.

### 2. Machine Learning — Random Forest
Model telatih dengan akurasi **~92%** pada test set. Menggunakan 10 fitur numerik yang diekstraksi dari pipeline morfologi.

> Kedua metode dapat dipilih langsung dari sidebar aplikasi.

---

## 🔬 Fitur yang Diekstraksi (10 Fitur)

| # | Fitur | Keterangan |
|---|-------|------------|
| 1 | `area` | Luas area biji (piksel) |
| 2 | `perimeter` | Panjang tepi kontur |
| 3 | `circularity` | Kebulatan bentuk biji |
| 4 | `solidity` | Kepadatan relatif terhadap convex hull |
| 5 | `extent` | Rasio area terhadap bounding box |
| 6 | `aspect_ratio` | Rasio lebar terhadap tinggi |
| 7 | `holes_count` | Jumlah lubang (deteksi dual-track: besar + kecil) |
| 8 | `center_cut_lines` | Garis tepi tengah via Hough Transform |
| 9 | `red_ratio` | Persentase komponen merah |
| 10 | `green_ratio` | Persentase komponen hijau |

---

## 🗂️ Struktur Proyek

```
coffee-bean-classifier/
│
├── app.py                    # Aplikasi Streamlit utama
│
├── extract_features.py       # STEP 1: Ekstraksi fitur dari dataset → CSV
├── train_model.py            # STEP 2: Training Random Forest dari CSV
│
├── utils/
│   ├── preprocessing.py      # M1–M3: Resize, Grayscale, Blur, Threshold
│   ├── morphology.py         # M4: Opening & Closing
│   ├── features.py           # M5–M6: Ekstraksi fitur + deteksi lubang dual-track
│   ├── clustering.py         # M7: K-Means & GMM color clustering
│   ├── visualization.py      # Decision boundary plot
│   └── dataset_loader.py     # Loader & statistik dataset
│
├── data/
│   ├── training dataset/     # Subfolder per kelas (Normal, Withered, dst.)
│   └── features_dataset.csv  # Hasil ekstraksi fitur (dibuat oleh STEP 1)
│
├── models/
│   ├── coffee_bean_rf.pkl    # Model Random Forest (dibuat oleh STEP 2)
│   └── scaler.pkl            # StandardScaler (dibuat oleh STEP 2)
│
└── requirements.txt
```

---

## 🧩 Materi Akademik yang Diimplementasikan

| Kode | Materi | Implementasi |
|------|--------|-------------|
| M1 | Sampling, Quantization, Discretization | `preprocessing.py` — konversi & resize |
| M2 | Interpolation, Geometric Intersections | `preprocessing.py` — 5 metode interpolasi |
| M3 | Convolution, Linear Filter | `preprocessing.py` — Gaussian Blur + Otsu |
| M4 | Morphology (Dilasi, Erosi, Opening, Closing) | `morphology.py` |
| M5 | Feature Detection (Points, Edges, Contours) | `features.py` — kontur + Canny |
| M6 | Segmentation, Hough Transform | `features.py` — HoughLinesP untuk center cut |
| M7 | Unsupervised Learning | `clustering.py` — K-Means & GMM |

---

## 🚀 Cara Setup & Menjalankan

### Prasyarat
- Python 3.8+
- Virtual environment (wajib)

---

### Langkah 1 — Clone & Setup Environment

**Windows (Command Prompt):**
```cmd
git clone <repository-url>
cd coffee-bean-classifier

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

**Linux / macOS:**
```bash
git clone <repository-url>
cd coffee-bean-classifier

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

### Langkah 2 — Siapkan Dataset

Letakkan gambar biji kopi di dalam folder berikut (buat jika belum ada):

```
data/
└── training dataset/
    ├── Normal/
    ├── Withered/
    ├── Partial Sour/
    ├── Broken/
    ├── Dry Cherry/
    └── Severe Insect Damage/
```

Format gambar yang didukung: `.jpg`, `.jpeg`, `.png`

---

### Langkah 3 — Training Model (opsional, untuk metode ML)

Jalankan dua skrip berikut secara berurutan:

```bash
# Ekstraksi fitur dari semua gambar → menghasilkan data/features_dataset.csv
python extract_features.py

# Training Random Forest → menghasilkan models/coffee_bean_rf.pkl dan models/scaler.pkl
python train_model.py
```

> Lewati langkah ini jika hanya ingin menggunakan metode Rule-Based.

---

### Langkah 4 — Jalankan Aplikasi

```bash
streamlit run app.py
```

Buka browser dan akses: `http://localhost:8501`

---

## 🖥️ Tampilan Aplikasi

Setelah upload atau memilih gambar dari dataset, aplikasi menampilkan hasil analisis dalam beberapa tab:

| Tab | Isi |
|-----|-----|
| **Preprocessing** | Langkah resize → grayscale → blur → binary |
| **Morfologi** | Hasil opening & closing |
| **Fitur & Kontur** | Visualisasi kontur + nilai 10 fitur |
| **Klasifikasi** | Hasil prediksi grade + confidence |
| **Analisis Model** *(ML only)* | Confusion matrix + profil fitur + decision boundary |

---

## ⚙️ Parameter yang Dapat Dikustomisasi

Semua parameter di bawah ini dapat diatur langsung dari **sidebar** tanpa perlu mengubah kode:

- **Ukuran resize** — 224×224 (rekomendasi), 256×256, 512×512
- **Metode interpolasi** — Nearest Neighbor, Bilinear, Bicubic, Area-based, Lanczos
- **Kernel Gaussian Blur** — slider 3–11
- **Kernel Opening & Closing** — slider morfologi terpisah
- **Metode klasifikasi** — Rule-Based atau Machine Learning

---

## 📦 Dependencies Utama

```
streamlit
opencv-python
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
tqdm
Pillow
plotly
```

Install semua dengan:
```bash
pip install -r requirements.txt
```

---

## 📝 Catatan Teknis

- **Deteksi lubang dual-track** — `features.py` menggunakan dua pendekatan: Track A untuk rongga besar (via hierarchy kontur) dan Track B untuk titik gelap kecil (via analisis intensitas lokal), dengan deduplication otomatis.
- **K-Means clustering** dilakukan pada piksel biji setelah background hitam dimasking, sehingga warna background tidak ikut terhitung.
- **StandardScaler** disimpan bersamaan dengan model agar prediksi di `app.py` menggunakan skala yang identik dengan saat training.
- Jika model `.pkl` tidak ditemukan, aplikasi otomatis fallback ke mode Rule-Based.

---

