import cv2
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# DETEKSI LUBANG SERANGGA — dual-track
# Track A: lubang besar (background tembus) via binary Otsu hierarchy
# Track B: titik gelap kecil (lubang masuk serangga) via darkspot analysis
# ════════════════════════════════════════════════════════════════════════════

def detect_insect_holes_large(binary_img, contours, hierarchy, main_idx, main_area):
    """Track A: rongga besar yang menembus biji (background terlihat dari dalam)."""
    hole_contours, hole_areas = [], []
    for i in range(len(contours)):
        if hierarchy[0][i][3] != main_idx:
            continue
        h_area = cv2.contourArea(contours[i])
        if h_area < 150:
            continue
        ratio = h_area / main_area if main_area > 0 else 0
        if not (0.015 <= ratio <= 0.60):
            continue
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
    Track B: titik gelap kecil (titik masuk serangga).
    Aman untuk Normal: center cut = garis panjang (circularity rendah) -> tersaring.
    Warna kuning/cream Normal tidak akan lolos filter RGB < 120.
    """
    img_h, img_w = original_gray.shape[:2]
    bx, by, bw, bh = cv2.boundingRect(main_contour)
    bean_pixels = original_gray[bean_mask == 255]
    if bean_pixels.size < 100:
        return [], []

    median_intensity = float(np.median(bean_pixels))
    std_intensity = float(np.std(bean_pixels))
    dark_threshold = max(30, min(100, median_intensity - 2.2 * std_intensity))

    dark_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    dark_mask[(original_gray < dark_threshold) & (bean_mask == 255)] = 255
    if cv2.countNonZero(dark_mask) == 0:
        return [], []

    k = np.ones((3, 3), np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k, iterations=1)
    spot_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hole_contours, hole_areas = [], []
    for sc in spot_contours:
        s_area = cv2.contourArea(sc)
        if not (3 <= s_area <= 80):
            continue
        if s_area / main_area > 0.03:
            continue
        s_perim = cv2.arcLength(sc, True)
        if s_perim == 0:
            continue
        s_circ = (4 * np.pi * s_area) / (s_perim ** 2)
        if s_circ < 0.25:
            continue
        M = cv2.moments(sc)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        margin = 8
        if not (bx + margin < cx < bx + bw - margin and
                by + margin < cy < by + bh - margin):
            continue
        spot_mask_single = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(spot_mask_single, [sc], -1, 255, -1)
        spot_pixels = original_gray[spot_mask_single == 255]
        if spot_pixels.size == 0 or np.mean(spot_pixels) > dark_threshold + 15:
            continue
        spot_rgb = original_rgb[spot_mask_single == 255]
        if spot_rgb.size > 0 and spot_rgb.mean(axis=0).max() > 120:
            continue
        hole_contours.append(sc)
        hole_areas.append(s_area)

    return hole_contours, hole_areas


def detect_insect_holes(binary_img, contours, hierarchy, main_idx, main_area,
                        original_gray, original_rgb, main_contour, bean_mask):
    """Gabungan Track A + Track B dengan deduplication."""
    holes_a, areas_a = detect_insect_holes_large(
        binary_img, contours, hierarchy, main_idx, main_area)
    holes_b, areas_b = detect_insect_holes_small(
        original_gray, original_rgb, main_contour, main_area, bean_mask)

    if holes_a:
        filtered_b, filtered_areas_b = [], []
        for hb, ab in zip(holes_b, areas_b):
            M = cv2.moments(hb)
            if M['m00'] == 0:
                continue
            cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
            inside_a = any(cv2.pointPolygonTest(ha, (cx, cy), False) >= 0 for ha in holes_a)
            if not inside_a:
                filtered_b.append(hb)
                filtered_areas_b.append(ab)
        holes_b, areas_b = filtered_b, filtered_areas_b

    return holes_a + holes_b, areas_a + areas_b


# ════════════════════════════════════════════════════════════════════════════
# FUNGSI UTAMA
# ════════════════════════════════════════════════════════════════════════════

def extract_all_features(binary_img, blurred_img, original_rgb, original_gray):
    """
    Ekstraksi Fitur: Geometri + Warna.

    Fitur pembeda Withered vs Normal (dari analisis data nyata):
    - mean_intensity: Withered 135-163 vs Normal ~123  (terkuat)
    - circularity:    Withered 0.74-0.87 vs Normal ~0.856
    - solidity:       Withered 0.957-0.987 vs Normal ~0.990
    - aspect_ratio:   Normal false pos = 0.68-0.87 (landscape), Withered = 0.86-1.45
                      -> AR < 0.86 = guard: jangan hukum biji landscape sebagai Withered
    """
    features = {
        'is_valid': False,
        'area': 0, 'perimeter': 0, 'circularity': 0,
        'solidity': 0, 'extent': 0, 'aspect_ratio': 0,
        'holes_count': 0, 'center_cut_lines': 0,
        'red_ratio': 0, 'green_ratio': 0, 'mean_intensity': 0,
        'viz_image': None, 'cropped_rgb': None
    }

    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return features

    main_idx, max_area = -1, 0
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_area = area; main_idx = i

    if main_idx == -1 or max_area < 500:
        return features

    cnt = contours[main_idx]
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = float(max_area) / hull_area if hull_area > 0 else 0
    extent = float(max_area) / (w * h) if w * h > 0 else 0
    circularity = (4 * np.pi * max_area) / (perimeter ** 2) if perimeter > 0 else 0

    img_h, img_w = original_rgb.shape[:2]
    bean_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(bean_mask, [cnt], -1, 255, -1)

    # Deteksi lubang dual-track
    hole_contours, hole_areas = detect_insect_holes(
        binary_img, contours, hierarchy, main_idx, max_area,
        original_gray, original_rgb, cnt, bean_mask
    )
    holes_count = len(hole_contours)

    # Canny + HoughLines untuk center_cut_lines saja (bukan untuk lubang)
    edges = cv2.Canny(blurred_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=12, minLineLength=8, maxLineGap=4)
    center_cut_lines = len(lines) if lines is not None else 0

    # Analisis warna
    mean_val = cv2.mean(original_rgb, mask=bean_mask)
    r_mean, g_mean = mean_val[0], mean_val[1]
    total_rgb = r_mean + g_mean + mean_val[2]
    red_ratio = (r_mean / total_rgb * 100) if total_rgb > 0 else 0
    green_ratio = (g_mean / total_rgb * 100) if total_rgb > 0 else 0
    mean_intensity = np.mean(original_gray[bean_mask == 255]) if np.any(bean_mask == 255) else 0

    # Visualisasi
    viz_img = original_rgb.copy()
    cv2.drawContours(viz_img, [cnt], -1, (0, 255, 0), 2)
    for hc in hole_contours:
        cv2.drawContours(viz_img, [hc], -1, (255, 50, 50), 2)

    # Crop untuk K-Means
    pad = 5
    y1, y2 = max(0, y - pad), min(img_h, y + h + pad)
    x1, x2 = max(0, x - pad), min(img_w, x + w + pad)
    cropped = original_rgb[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = original_rgb[y:y + h, x:x + w]

    features.update({
        'is_valid': True,
        'area': max_area, 'perimeter': perimeter,
        'circularity': circularity, 'solidity': solidity,
        'extent': extent, 'aspect_ratio': aspect_ratio,
        'holes_count': holes_count, 'center_cut_lines': center_cut_lines,
        'red_ratio': red_ratio, 'green_ratio': green_ratio,
        'mean_intensity': mean_intensity,
        'viz_image': viz_img, 'cropped_rgb': cropped,
        'contour': cnt, 'all_contours': contours,
        'hole_contours': hole_contours, 'hole_areas': hole_areas,
    })
    return features


def classify_coffee_bean(features):
    """Penentuan kelas awal (diperkuat di app.py)."""
    if not features['is_valid']:
        return "Tidak Terdeteksi", 0, 0

    rg_ratio = features['red_ratio'] / features['green_ratio'] if features['green_ratio'] > 0 else 1.0
    intensity = features['mean_intensity']
    holes = features['holes_count']
    circ = features['circularity']
    solidity = features['solidity']
    ar = features['aspect_ratio']

    if holes >= 1:
        return "Severe Insect Damage", 5, 95.0
    if intensity < 95:
        return "Dry Cherry", 5, 90.0
    if rg_ratio > 1.25:
        return "Partial Sour", 4, 88.0
    if intensity > 185:
        return "Immature", 2, 85.0

    # Withered: multi-sinyal, divalidasi 15/15 benar di dataset
    # (10 Withered terdeteksi, 5 Normal tidak false positive)
    ws = 0
    # G/B ratio: hitung dari red_ratio/green_ratio sebagai proxy
    # (G/B asli dihitung di app.py; di sini pakai intensity+AR sebagai fallback)
    if intensity > 148:        ws += 2   # Intensity sangat tinggi
    if intensity > 135 and ar >= 0.88: ws += 2   # Terang + portrait
    if circ < 0.80:            ws += 1
    if solidity < 0.985:       ws += 1
    if ar >= 0.88:             ws += 1   # Orientasi portrait/tegak
    if ws >= 3:
        return "Withered", 2, 75.0

    return "Normal", 1, 95.0