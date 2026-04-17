 
# ☕ Klasifikasi Mutu Biji Kopi Mentah

> Implementasi Handcrafted Feature Extraction untuk Klasifikasi Kualitas Biji Kopi Menggunakan Metode Morfologi Matematika dan Deteksi Tepi Canny

## 📋 Deskripsi Proyek

Aplikasi Streamlit untuk klasifikasi mutu biji kopi mentah (green bean) berdasarkan ekstraksi fitur morfologi dan geometri dengan pendekatan preprocessing citra digital.

**Materi yang Diimplementasikan:**
- **M1**: Sampling, Quantization, Discretization
- **M2**: Interpolation, Geometric Intersections
- **M3**: Convolution, Linear Filter
- **M4**: Morphology (Dilasi, Erosi, Opening, Closing)
- **M5**: Feature Detection (Points, Edges, Contours)
- **M6**: Segmentation, Hough Transform
- **M7**: Unsupervised Learning (K-Means Clustering)


---

## 🚀 Cara Setup & Menjalankan Aplikasi

### ⚠️ PENTING: Setup Virtual Environment (venv)

**JANGAN skip langkah ini!** Virtual environment diperlukan agar dependencies tidak conflict dengan project lain.

#### Windows (Command Prompt)
```cmd
# 1. Clone repository
git clone <repository-url>
cd coffee-bean-classifier

# 2. Buat virtual environment
python -m venv venv

# 3. Aktifkan virtual environment
venv\Scripts\activate

# 4. Pastikan venv aktif (ada tulisan (venv) di prompt)
# (venv) D:\...\coffee-bean-classifier>

# 5. Install dependencies
pip install -r requirements.txt