"""
batch_testing.py
Script untuk menguji seluruh dataset dan menyimpan hasil ke CSV.
Jalankan dari terminal: python batch_testing.py
"""

import os
import csv
import cv2
import numpy as np
from utils.preprocessing import preprocess_pipeline
from utils.morphology import apply_morphology
from utils.features import extract_all_features
from utils.clustering import analyze_color_kmeans
from utils.dataset_loader import DATASET_INFO

# ============================================================
# KONFIGURASI — sesuaikan dengan parameter default kamu
# ============================================================
TARGET_SIZE = (224, 224)
BLUR_KERNEL = (5, 5)
OPEN_KERNEL = (3, 3)
CLOSE_KERNEL = (3, 3)
INTERPOLATION = "Area-based"

DATASET_PATH = "data/training dataset"  # sesuaikan path-mu
OUTPUT_CSV = "hasil_pengujian.csv"

# ============================================================
# FUNGSI PENALTI (salin dari app.py, pastikan IDENTIK)
# ============================================================

def run_classification(geom_features, preprocess_results, cluster_stats):
    """Menjalankan logika penalti yang sama dengan app.py Tab 3."""
    logs = []
    triggered = {
        'broken': False,
        'partial_sour': False,
        'withered': False,
        'dry_cherry': False,
        'insect_damage': False
    }
    
    current_score = 100

    # --- 1. BROKEN ---
    aspect_ratio = geom_features.get('aspect_ratio', 1.0)
    extent = geom_features.get('extent', 0.8)
    if aspect_ratio > 1.7 or aspect_ratio < 0.65 or extent < 0.68:
        current_score -= 75
        triggered['broken'] = True
        logs.append(f"Broken (AR:{aspect_ratio:.2f}, Ext:{extent:.2f})")

    # --- 2. PARTIAL SOUR ---
    if cluster_stats:
        valid_ratios = []
        for idx, stat in cluster_stats.items():
            r, g, b = stat['color']
            if r > 185 and g > 185 and b > 185:
                continue
            valid_ratios.append(r / g if g > 0 else 1.0)
        if valid_ratios:
            max_rg = max(valid_ratios)
            if max_rg > 1.28:
                current_score -= 50
                triggered['partial_sour'] = True
                logs.append(f"Partial Sour (R/G:{max_rg:.2f})")

    # --- 3. WITHERED (multi-signal) ---
    intensity_w = geom_features.get('mean_intensity', 127)
    circ_w = geom_features.get('circularity', 1.0)
    sol_w = geom_features.get('solidity', 1.0)
    ar_w = geom_features.get('aspect_ratio', 1.0)

    # Hitung G/B ratio
    _mask_w = np.zeros(preprocess_results['rgb'].shape[:2], dtype=np.uint8)
    if 'contour' in geom_features:
        cv2.drawContours(_mask_w, [geom_features['contour']], -1, 255, -1)
    _rgb_w = preprocess_results['rgb']
    if np.any(_mask_w == 255):
        _g_w = float(np.mean(_rgb_w[:, :, 1][_mask_w == 255]))
        _b_w = float(np.mean(_rgb_w[:, :, 2][_mask_w == 255]))
    else:
        _g_w, _b_w = 130, 90
    gb_ratio = _g_w / _b_w if _b_w > 0 else 1.5

    withered_signals = 0
    if gb_ratio < 1.40 and intensity_w > 140:
        withered_signals += 3
    elif gb_ratio < 1.38 and intensity_w > 132:
        withered_signals += 2
    elif intensity_w > 148:
        withered_signals += 2
    if circ_w < 0.80:
        withered_signals += 1
    if sol_w < 0.985:
        withered_signals += 1
    if ar_w >= 0.88:
        withered_signals += 1

    if withered_signals >= 3 and current_score > 70:
        current_score -= 30
        triggered['withered'] = True
        logs.append(f"Withered (signals:{withered_signals})")

    # --- 4. DRY CHERRY ---
    intensity = geom_features.get('mean_intensity', 127)
    if intensity < 95:
        current_score -= 75
        triggered['dry_cherry'] = True
        logs.append(f"Dry Cherry (Intensity:{intensity:.0f})")

    # --- 5. INSECT DAMAGE ---
    holes = geom_features.get('holes_count', 0)
    if holes > 0:
        if holes >= 2:
            penalty = 90
        else:
            penalty = 85
        current_score -= penalty
        triggered['insect_damage'] = True
        logs.append(f"Insect Damage ({holes} holes)")

    final_score = max(0, current_score)

    # Tentukan grade
    if final_score >= 88:
        grade = 1
    elif final_score >= 60:
        grade = 2
    elif final_score >= 40:
        grade = 3
    elif final_score >= 20:
        grade = 4
    else:
        grade = 5

    return final_score, grade, triggered, ", ".join(logs)


# ============================================================
# LOOP BATCH TESTING
# ============================================================

def main():
    rows = []

    for class_name, info in DATASET_INFO.items():
        folder_path = os.path.join(DATASET_PATH, info['folder'])
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder tidak ditemukan: {folder_path}")
            continue

        images = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()

        print(f"\n📂 Memproses kelas: {class_name} ({len(images)} gambar)")

        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"  ⚠️ Gagal membaca: {img_name}")
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            try:
                # Preprocessing
                preprocess_results = preprocess_pipeline(
                    image_rgb,
                    target_size=TARGET_SIZE,
                    blur_kernel=BLUR_KERNEL,
                    interpolation_method=INTERPOLATION
                )

                # Morphology
                morph_results = apply_morphology(
                    preprocess_results['binary'],
                    open_kernel=OPEN_KERNEL,
                    close_kernel=CLOSE_KERNEL
                )

                # Feature Extraction
                geom_features = extract_all_features(
                    morph_results['closing'],
                    preprocess_results['blur'],
                    preprocess_results['rgb'],
                    preprocess_results['gray']
                )

                # K-Means (hanya jika valid)
                kmeans_img, color_centers, cluster_stats = None, None, None
                if geom_features['is_valid'] and geom_features['cropped_rgb'] is not None:
                    kmeans_img, color_centers, cluster_stats = analyze_color_kmeans(
                        geom_features['cropped_rgb'], k=3
                    )

                # Klasifikasi
                if geom_features['is_valid']:
                    final_score, grade, triggered, log_text = run_classification(
                        geom_features, preprocess_results, cluster_stats
                    )
                else:
                    final_score = 0
                    grade = 0
                    triggered = {'broken': False, 'partial_sour': False,
                                 'withered': False, 'dry_cherry': False,
                                 'insect_damage': False}
                    log_text = "Objek tidak terdeteksi"

                # Simpan baris hasil
                rows.append({
                    'nama_file': img_name,
                    'kelas_asli': class_name,
                    'grade_sebenarnya': info['grade'],
                    'skor_akhir': final_score,
                    'grade_prediksi': grade,
                    'broken_terpicu': triggered['broken'],
                    'sour_terpicu': triggered['partial_sour'],
                    'withered_terpicu': triggered['withered'],
                    'drycherry_terpicu': triggered['dry_cherry'],
                    'insect_terpicu': triggered['insect_damage'],
                    'log_penalti': log_text,
                    'area': geom_features.get('area', 0),
                    'circularity': geom_features.get('circularity', 0),
                    'solidity': geom_features.get('solidity', 0),
                    'aspect_ratio': geom_features.get('aspect_ratio', 0),
                    'extent': geom_features.get('extent', 0),
                    'holes_count': geom_features.get('holes_count', 0),
                    'mean_intensity': geom_features.get('mean_intensity', 0)
                })

                status = "✅" if grade == info['grade'] else "❌"
                print(f"  {status} {img_name} | True: {info['grade']} | Pred: {grade} | Skor: {final_score} | {log_text}")

            except Exception as e:
                print(f"  ❌ Error pada {img_name}: {e}")
                continue

    # ============================================================
    # SIMPAN KE CSV
    # ============================================================
    if rows:
        fieldnames = [
            'nama_file', 'kelas_asli', 'grade_sebenarnya', 'skor_akhir', 'grade_prediksi',
            'broken_terpicu', 'sour_terpicu', 'withered_terpicu', 'drycherry_terpicu', 'insect_terpicu',
            'log_penalti', 'area', 'circularity', 'solidity', 'aspect_ratio', 'extent',
            'holes_count', 'mean_intensity'
        ]
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n✅ Selesai! {len(rows)} sampel berhasil diuji.")
        print(f"📄 Hasil disimpan di: {OUTPUT_CSV}")

        # Hitung akurasi cepat
        benar = sum(1 for r in rows if r['grade_prediksi'] == r['grade_sebenarnya'])
        print(f"🎯 Akurasi sementara: {benar}/{len(rows)} = {benar/len(rows)*100:.1f}%")
    else:
        print("❌ Tidak ada data yang berhasil diuji.")


if __name__ == "__main__":
    main()