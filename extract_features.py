"""
STEP 1: Ekstraksi Fitur dari Dataset
Membaca semua gambar di folder data/training dataset
Mengekstrak fitur menggunakan extract_all_features()
Menghasilkan file CSV untuk training model supervised
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm  # Progress bar, install: pip install tqdm

# Import fungsi dari utils yang sudah ada
from utils.preprocessing import preprocess_pipeline
from utils.morphology import apply_morphology
from utils.features import extract_all_features

# Konfigurasi
DATASET_PATH = "data/training dataset"
OUTPUT_CSV = "data/features_dataset.csv"

# Parameter preprocessing (gunakan default yang sudah terbukti bekerja)
TARGET_SIZE = (224, 224)
BLUR_KERNEL = (5, 5)
INTERPOLATION = "Area-based"
OPEN_KERNEL = (3, 3)
CLOSE_KERNEL = (3, 3)

# Mapping folder ke label (sesuai dengan DATASET_INFO di dataset_loader.py)
FOLDER_TO_LABEL = {
    "Normal": "Normal",
    "Withered": "Withered",
    "Partial Sour": "Partial Sour",
    "Broken": "Broken",
    "Dry Cherry": "Dry Cherry",
    "Severe Insect Damage": "Severe Insect Damage"
}

# Label ke angka untuk klasifikasi (0-5)
LABEL_TO_NUM = {
    "Normal": 0,
    "Withered": 1,
    "Partial Sour": 2,
    "Broken": 3,
    "Dry Cherry": 4,
    "Severe Insect Damage": 5
}

# Nama kolom fitur yang akan diekstrak
FEATURE_COLUMNS = [
    'filename',
    'class_name',
    'class_num',
    'area',
    'perimeter',
    'circularity',
    'solidity',
    'extent',
    'aspect_ratio',
    'holes_count',
    'center_cut_lines',
    'mean_intensity',
    'red_ratio',
    'green_ratio',
    'is_valid'
]


def load_image(image_path):
    """Load gambar dari path, return RGB array"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract_features_from_image(image_path, target_size, blur_kernel, interpolation, open_kernel, close_kernel):
    """
    Mengekstrak fitur dari satu gambar
    Return: dict berisi fitur-fitur
    """
    # Load gambar
    img_rgb = load_image(image_path)
    if img_rgb is None:
        return None
    
    # Preprocessing pipeline
    preprocess_results = preprocess_pipeline(
        img_rgb,
        target_size=target_size,
        blur_kernel=blur_kernel,
        interpolation_method=interpolation
    )
    
    # Morphology
    morph_results = apply_morphology(
        preprocess_results['binary'],
        open_kernel=open_kernel,
        close_kernel=close_kernel
    )
    
    # Ekstraksi fitur (dengan hole detection dual-track)
    features = extract_all_features(
        morph_results['closing'],
        preprocess_results['blur'],
        preprocess_results['rgb'],
        preprocess_results['gray']
    )
    
    return features


def main():
    print("=" * 60)
    print("STEP 1: Ekstraksi Fitur dari Dataset")
    print("=" * 60)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print()
    
    # Cek apakah folder dataset ada
    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Folder dataset tidak ditemukan: {DATASET_PATH}")
        print("Pastikan path sudah benar dan folder berisi subfolder kelas.")
        return
    
    # Kumpulkan semua file gambar
    all_images = []
    for class_name, label in FOLDER_TO_LABEL.items():
        folder_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(folder_path):
            print(f"⚠️  Folder tidak ditemukan: {folder_path}")
            continue
        
        # Ambil semua file gambar
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append({
                    'path': os.path.join(folder_path, filename),
                    'filename': filename,
                    'class_name': class_name,
                    'class_num': LABEL_TO_NUM[class_name]
                })
    
    print(f"📊 Total gambar ditemukan: {len(all_images)}")
    print("\nPer kelas:")
    for class_name in FOLDER_TO_LABEL.keys():
        count = sum(1 for img in all_images if img['class_name'] == class_name)
        print(f"   - {class_name}: {count} gambar")
    print()
    
    if len(all_images) == 0:
        print("❌ ERROR: Tidak ada gambar ditemukan.")
        return
    
    # Ekstraksi fitur untuk setiap gambar
    results = []
    failed_images = []
    
    print("🔄 Memulai ekstraksi fitur...")
    for img_info in tqdm(all_images, desc="Processing images"):
        try:
            features = extract_features_from_image(
                img_info['path'],
                TARGET_SIZE,
                BLUR_KERNEL,
                INTERPOLATION,
                OPEN_KERNEL,
                CLOSE_KERNEL
            )
            
            if features is None or not features.get('is_valid', False):
                failed_images.append({
                    'filename': img_info['filename'],
                    'class': img_info['class_name'],
                    'reason': 'Object not detected (is_valid=False)'
                })
                continue
            
            # Simpan hasil
            results.append({
                'filename': img_info['filename'],
                'class_name': img_info['class_name'],
                'class_num': img_info['class_num'],
                'area': features.get('area', 0),
                'perimeter': features.get('perimeter', 0),
                'circularity': features.get('circularity', 0),
                'solidity': features.get('solidity', 0),
                'extent': features.get('extent', 0),
                'aspect_ratio': features.get('aspect_ratio', 0),
                'holes_count': features.get('holes_count', 0),
                'center_cut_lines': features.get('center_cut_lines', 0),
                'mean_intensity': features.get('mean_intensity', 0),
                'red_ratio': features.get('red_ratio', 0),
                'green_ratio': features.get('green_ratio', 0),
                'is_valid': True
            })
            
        except Exception as e:
            failed_images.append({
                'filename': img_info['filename'],
                'class': img_info['class_name'],
                'reason': str(e)
            })
    
    # Buat DataFrame
    df = pd.DataFrame(results)
    
    # Simpan ke CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Laporan hasil
    print("\n" + "=" * 60)
    print("HASIL EKSTRAKSI FITUR")
    print("=" * 60)
    print(f"✅ Berhasil diproses: {len(results)} gambar")
    print(f"❌ Gagal: {len(failed_images)} gambar")
    print(f"📁 CSV disimpan di: {OUTPUT_CSV}")
    
    if failed_images:
        print("\n⚠️  Daftar gambar yang gagal:")
        for fail in failed_images[:10]:  # Tampilkan 10 pertama
            print(f"   - {fail['filename']} ({fail['class']}): {fail['reason']}")
        if len(failed_images) > 10:
            print(f"   ... dan {len(failed_images) - 10} lainnya")
    
    # Statistik per kelas
    print("\n📊 Statistik per kelas:")
    for class_name in FOLDER_TO_LABEL.keys():
        class_df = df[df['class_name'] == class_name]
        if len(class_df) > 0:
            print(f"\n   {class_name} ({len(class_df)} sampel):")
            print(f"      Area: {class_df['area'].mean():.0f} ± {class_df['area'].std():.0f}")
            print(f"      Circularity: {class_df['circularity'].mean():.3f} ± {class_df['circularity'].std():.3f}")
            print(f"      Solidity: {class_df['solidity'].mean():.4f} ± {class_df['solidity'].std():.4f}")
            print(f"      Holes count: {class_df['holes_count'].sum()} lubang terdeteksi")
    
    return df, failed_images


if __name__ == "__main__":
    df, failed = main()