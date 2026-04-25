import cv2
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# DETEKSI LUBANG SERANGGA
# Strategi dual-track:
#   Track A – Lubang BESAR (background tembus biji) via binary Otsu hierarchy
#   Track B – Lubang KECIL (titik masuk serangga) via ROI darkspot analysis
#   Proteksi Normal: center cut & bintik alami dibuang lewat filter ketat
# ════════════════════════════════════════════════════════════════════════════

def detect_insect_holes_large(binary_img, contours, hierarchy, main_idx, main_area):
    """
    Track A: Lubang besar yang menembus biji — latar belakang terlihat dari dalam.
    Karakteristik: anak kontur dari binary Otsu, area signifikan (>= 150 px, >= 1.5% biji).
    Threshold diturunkan dari 300 ke 150 agar lubang medium tidak terlewat.
    """
    hole_contours = []
    hole_areas = []

    for i in range(len(contours)):
        if hierarchy[0][i][3] != main_idx:
            continue

        h_area = cv2.contourArea(contours[i])

        # Ukuran: minimal 150 px (lubang medium) sampai 60% luas biji
        if h_area < 150:
            continue
        ratio = h_area / main_area if main_area > 0 else 0
        if not (0.015 <= ratio <= 0.60):
            continue

        # Posisi: centroid harus cukup jauh dari tepi bounding box
        M = cv2.moments(contours[i])
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        bx, by, bw, bh = cv2.boundingRect(contours[main_idx])
        margin = 5
        if not (bx + margin < cx < bx + bw - margin and
                by + margin < cy < by + bh - margin):
            continue

        hole_contours.append(contours[i])
        hole_areas.append(h_area)

    return hole_contours, hole_areas


def detect_insect_holes_small(original_gray, original_rgb, main_contour, main_area, bean_mask):
    """
    Track B: Titik gelap kecil (lubang masuk serangga, titik hitam/coklat tua).

    Algoritma:
    1. Crop area biji saja (pakai bean_mask)
    2. Cari piksel yang jauh lebih gelap dari median biji (outlier gelap)
    3. Cluster piksel gelap → kontur titik
    4. Filter ketat: area 3-80 px, circularitas tinggi, jauh dari tepi
    5. Proteksi Normal: biji normal TIDAK punya outlier gelap terlokalisasi

    Kenapa aman untuk Normal:
    - Center cut = garis tipis memanjang → circularitas sangat rendah, tersaring
    - Bintik alami di Normal: tidak cukup gelap vs median biji (threshold ketat)
    """
    img_h, img_w = original_gray.shape[:2]
    bx, by, bw, bh = cv2.boundingRect(main_contour)

    # Ambil piksel biji (dalam mask)
    bean_pixels = original_gray[bean_mask == 255]
    if bean_pixels.size < 100:
        return [], []

    median_intensity = float(np.median(bean_pixels))
    std_intensity = float(np.std(bean_pixels))

    # Threshold outlier gelap: harus di bawah (median - 2.2 * std) dan <= 100
    # Ini cukup ketat: biji Normal memiliki std rendah dan tidak punya spot sekuat itu
    dark_threshold = max(30, min(100, median_intensity - 2.2 * std_intensity))

    # Bangun binary: piksel gelap di dalam biji
    dark_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    dark_mask[(original_gray < dark_threshold) & (bean_mask == 255)] = 255

    if cv2.countNonZero(dark_mask) == 0:
        return [], []

    # Morfologi kecil: tutup gap kecil dalam spot gelap
    k = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k, iterations=1)

    # Temukan kontur spot gelap
    spot_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hole_contours = []
    hole_areas = []

    for sc in spot_contours:
        s_area = cv2.contourArea(sc)

        # Ukuran: 3 - 80 px (titik masuk serangga, bukan noda besar)
        if not (3 <= s_area <= 80):
            continue

        # Proporsi: tidak lebih dari 3% luas biji (titik serangga kecil)
        if s_area / main_area > 0.03:
            continue

        # Circularitas: spot serangga cenderung bulat/oval (> 0.25)
        # Center cut = sangat rendah (< 0.1), tersaring di sini
        s_perim = cv2.arcLength(sc, True)
        if s_perim == 0:
            continue
        s_circ = (4 * np.pi * s_area) / (s_perim ** 2)
        if s_circ < 0.25:
            continue

        # Posisi: spot harus di dalam area biji, jauh dari tepi
        M = cv2.moments(sc)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        margin = 8
        if not (bx + margin < cx < bx + bw - margin and
                by + margin < cy < by + bh - margin):
            continue

        # Verifikasi: piksel di spot memang gelap secara konsisten
        spot_mask_single = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(spot_mask_single, [sc], -1, 255, -1)
        spot_pixels = original_gray[spot_mask_single == 255]
        if spot_pixels.size == 0:
            continue

        # Rata-rata spot harus <= threshold + 15 (tidak terlalu terang)
        if np.mean(spot_pixels) > dark_threshold + 15:
            continue

        # Verifikasi warna RGB: spot serangga = gelap (hitam/coklat tua)
        # Bukan glare (putih) atau noda kuning alami
        spot_rgb = original_rgb[spot_mask_single == 255]
        if spot_rgb.size > 0:
            mean_rgb = spot_rgb.mean(axis=0)  # [R, G, B]
            # Spot serangga: semua channel rendah (gelap)
            # Bukan: warna kuning/putih/cream (yang sering ada di biji normal)
            if mean_rgb.max() > 120:  # Terlalu terang, bukan lubang serangga
                continue

        hole_contours.append(sc)
        hole_areas.append(s_area)

    return hole_contours, hole_areas


def detect_insect_holes(binary_img, contours, hierarchy, main_idx, main_area,
                        original_gray, original_rgb, main_contour, bean_mask):
    """
    Gabungan Track A (lubang besar) + Track B (titik gelap kecil).
    Deduplication: jika Track B mendeteksi spot di dalam area Track A, skip.
    """
    # Track A: lubang besar dari binary hierarchy
    holes_a, areas_a = detect_insect_holes_large(
        binary_img, contours, hierarchy, main_idx, main_area
    )

    # Track B: titik gelap kecil
    holes_b, areas_b = detect_insect_holes_small(
        original_gray, original_rgb, main_contour, main_area, bean_mask
    )

    # Deduplication: buang Track B yang centroid-nya ada di dalam kontur Track A
    if holes_a:
        filtered_b = []
        filtered_areas_b = []
        for hb, ab in zip(holes_b, areas_b):
            M = cv2.moments(hb)
            if M['m00'] == 0:
                continue
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            inside_a = False
            for ha in holes_a:
                if cv2.pointPolygonTest(ha, (cx, cy), False) >= 0:
                    inside_a = True
                    break
            if not inside_a:
                filtered_b.append(hb)
                filtered_areas_b.append(ab)
        holes_b = filtered_b
        areas_b = filtered_areas_b

    all_holes = holes_a + holes_b
    all_areas = areas_a + areas_b
    return all_holes, all_areas


# ════════════════════════════════════════════════════════════════════════════
# FITUR WITHERED — tambahan untuk membedakan Withered vs Normal
# ════════════════════════════════════════════════════════════════════════════

def compute_withered_features(original_rgb, original_gray, bean_mask):
    """
    Fitur tambahan untuk membedakan Withered dari Normal:
    - HSV saturation mean: Withered lebih pucat (saturation rendah)
    - Color variance (std warna permukaan): Withered belang, tidak merata
    - Local texture variance: Withered permukaan lebih kasar secara lokal
    """
    # Hanya area biji
    roi_rgb = original_rgb.copy()
    roi_rgb[bean_mask == 0] = 0

    # Konversi ke HSV untuk saturation
    hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    sat_channel = hsv[:, :, 1]
    sat_pixels = sat_channel[bean_mask == 255].astype(float)
    mean_saturation = float(np.mean(sat_pixels)) if sat_pixels.size > 0 else 0

    # Std tiap channel warna (ukuran variasi/belang permukaan)
    r_pixels = original_rgb[:, :, 0][bean_mask == 255].astype(float)
    g_pixels = original_rgb[:, :, 1][bean_mask == 255].astype(float)
    b_pixels = original_rgb[:, :, 2][bean_mask == 255].astype(float)
    color_std = float(np.mean([np.std(r_pixels), np.std(g_pixels), np.std(b_pixels)]))

    # Local texture variance (Laplacian variance) — Withered lebih kasar
    gray_roi = original_gray.copy()
    gray_roi[bean_mask == 0] = 0
    lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
    lap_pixels = lap[bean_mask == 255]
    texture_var = float(np.var(lap_pixels)) if lap_pixels.size > 0 else 0

    return {
        'mean_saturation': mean_saturation,
        'color_std': color_std,
        'texture_var': texture_var,
    }


# ════════════════════════════════════════════════════════════════════════════
# FUNGSI UTAMA
# ════════════════════════════════════════════════════════════════════════════

def extract_all_features(binary_img, blurred_img, original_rgb, original_gray):
    """
    Ekstraksi Fitur: Geometri (M5-M6) + Warna (M7)
    Pembaruan:
    - Deteksi lubang: dual-track (besar via hierarchy + kecil via darkspot)
    - Tambahan fitur Withered: saturation, color_std, texture_var
    """
    features = {
        'is_valid': False,
        'area': 0,
        'perimeter': 0,
        'circularity': 0,
        'solidity': 0,
        'extent': 0,
        'aspect_ratio': 0,
        'holes_count': 0,
        'center_cut_lines': 0,
        'red_ratio': 0,
        'green_ratio': 0,
        'mean_intensity': 0,
        'mean_saturation': 0,
        'color_std': 0,
        'texture_var': 0,
        'viz_image': None,
        'cropped_rgb': None
    }

    # 1. Deteksi Kontur Utama & Hierarchy
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or hierarchy is None:
        return features

    main_idx = -1
    max_area = 0
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_area = area
                main_idx = i

    if main_idx == -1 or max_area < 500:
        return features

    cnt = contours[main_idx]
    perimeter = cv2.arcLength(cnt, True)

    # Fitur Geometri
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(max_area) / hull_area if hull_area > 0 else 0

    rect_area = w * h
    extent = float(max_area) / rect_area if rect_area > 0 else 0

    circularity = (4 * np.pi * max_area) / (perimeter ** 2) if perimeter > 0 else 0

    # 2. Bean mask (dibutuhkan oleh detect_insect + withered features)
    img_h, img_w = original_rgb.shape[:2]
    bean_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(bean_mask, [cnt], -1, 255, -1)

    # 3. Deteksi Lubang Hama (dual-track)
    hole_contours, hole_areas = detect_insect_holes(
        binary_img, contours, hierarchy, main_idx, max_area,
        original_gray, original_rgb, cnt, bean_mask
    )
    holes_count = len(hole_contours)

    # 4. Deteksi Garis Tengah/Retakan (Canny + HoughLines — untuk center_cut_lines)
    edges = cv2.Canny(blurred_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=12, minLineLength=8, maxLineGap=4)
    center_cut_lines = len(lines) if lines is not None else 0

    # 5. Analisis Warna & Masking
    mean_val = cv2.mean(original_rgb, mask=bean_mask)
    r_mean, g_mean, b_mean = mean_val[0], mean_val[1], mean_val[2]

    total_rgb = r_mean + g_mean + b_mean
    red_ratio = (r_mean / total_rgb * 100) if total_rgb > 0 else 0
    green_ratio = (g_mean / total_rgb * 100) if total_rgb > 0 else 0

    mean_intensity = np.mean(original_gray[bean_mask == 255]) if np.any(bean_mask == 255) else 0

    # 6. Fitur Withered tambahan
    withered_feats = compute_withered_features(original_rgb, original_gray, bean_mask)

    # 7. Visualisasi
    viz_img = original_rgb.copy()
    cv2.drawContours(viz_img, [cnt], -1, (0, 255, 0), 2)
    for hc in hole_contours:
        cv2.drawContours(viz_img, [hc], -1, (255, 50, 50), 2)

    # 8. Cropping untuk K-Means
    pad = 5
    y1, y2 = max(0, y - pad), min(img_h, y + h + pad)
    x1, x2 = max(0, x - pad), min(img_w, x + w + pad)
    cropped = original_rgb[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = original_rgb[y:y + h, x:x + w]

    features.update({
        'is_valid': True,
        'area': max_area,
        'perimeter': perimeter,
        'circularity': circularity,
        'solidity': solidity,
        'extent': extent,
        'aspect_ratio': aspect_ratio,
        'holes_count': holes_count,
        'center_cut_lines': center_cut_lines,
        'red_ratio': red_ratio,
        'green_ratio': green_ratio,
        'mean_intensity': mean_intensity,
        'mean_saturation': withered_feats['mean_saturation'],
        'color_std': withered_feats['color_std'],
        'texture_var': withered_feats['texture_var'],
        'viz_image': viz_img,
        'cropped_rgb': cropped,
        'contour': cnt,
        'all_contours': contours,
        'hole_contours': hole_contours,
        'hole_areas': hole_areas,
    })

    return features


def classify_coffee_bean(features):
    """
    Logika Penentuan Kelas awal (akan diperkuat di app.py)
    """
    if not features['is_valid']:
        return "Tidak Terdeteksi", 0, 0

    rg_ratio = features['red_ratio'] / features['green_ratio'] if features['green_ratio'] > 0 else 1.0
    intensity = features['mean_intensity']
    holes = features['holes_count']
    circ = features['circularity']
    solidity = features['solidity']
    saturation = features.get('mean_saturation', 100)

    if holes >= 1:
        return "Severe Insect Damage", 5, 95.0
    if intensity < 95:
        return "Dry Cherry", 5, 90.0
    if rg_ratio > 1.25:
        return "Partial Sour", 4, 88.0
    if intensity > 185:
        return "Immature", 2, 85.0
    # Withered: saturation rendah ATAU solidity rendah/circularity rendah
    if (saturation < 55 and intensity > 130) or solidity < 0.94 or circ < 0.75:
        return "Withered", 2, 75.0

    return "Normal", 1, 95.0