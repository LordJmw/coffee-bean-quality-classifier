import cv2
import numpy as np

def extract_all_features(binary_img, blurred_img, original_rgb, original_gray):
    """
    Ekstraksi Fitur: Geometri (M5-M6) + Warna (M7)
    Optimasi: Filtrasi lubang hama untuk menghindari 'False Positive' pada center cut.
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
        'viz_image': None,
        'cropped_rgb': None
    }

    # 1. Deteksi Kontur Utama & Hierarchy (M5)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours or hierarchy is None:
        return features

    main_idx = -1
    max_area = 0
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1: # Kontur luar
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_area = area
                main_idx = i

    if main_idx == -1 or max_area < 500:
        return features

    cnt = contours[main_idx]
    perimeter = cv2.arcLength(cnt, True)
    
    # Fitur Geometri Tambahan untuk deteksi Withered & Broken
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h if h > 0 else 0
    
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(max_area)/hull_area if hull_area > 0 else 0
    
    rect_area = w * h
    extent = float(max_area)/rect_area if rect_area > 0 else 0
    
    circularity = (4 * np.pi * max_area) / (perimeter ** 2) if perimeter > 0 else 0

    # 2. Deteksi Lubang (Optimasi agar tidak mendeteksi garis tengah)
    valid_holes_idx = []
    for i in range(len(contours)):
        # Jika kontur i adalah anak (hole) dari kontur utama
        if hierarchy[0][i][3] == main_idx:
            h_area = cv2.contourArea(contours[i])
            h_perimeter = cv2.arcLength(contours[i], True)
            
            # Perhitungan bundar/tidaknya lubang (Hole Circularity)
            h_circ = (4 * np.pi * h_area) / (h_perimeter ** 2) if h_perimeter > 0 else 0
            
            # FILTER LUBANG HAMA:
            # 1. Luas lubang harus kecil (2 - 50 pixel) - lubang hama tidak mungkin raksasa
            # 2. Bentuk harus cenderung bulat (h_circ > 0.15) - garis tengah biasanya h_circ < 0.1
            if 2 < h_area < 60 and h_circ > 0.15:
                valid_holes_idx.append(i)

    holes_count = len(valid_holes_idx)

    # 3. Deteksi Garis Tengah/Retakan (M5)
    edges = cv2.Canny(blurred_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=12, minLineLength=8, maxLineGap=4)
    center_cut_lines = len(lines) if lines is not None else 0

    # 4. Analisis Warna & Masking (M7)
    mask = np.zeros_like(original_gray)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    mean_val = cv2.mean(original_rgb, mask=mask)
    r_mean, g_mean, b_mean = mean_val[0], mean_val[1], mean_val[2]
    
    total_rgb = r_mean + g_mean + b_mean
    red_ratio = (r_mean / total_rgb * 100) if total_rgb > 0 else 0
    green_ratio = (g_mean / total_rgb * 100) if total_rgb > 0 else 0
    
    mean_intensity = np.mean(original_gray[mask == 255]) if np.any(mask == 255) else 0

    # 5. Visualisasi (Output ke UI)
    viz_img = original_rgb.copy()
    cv2.drawContours(viz_img, [cnt], -1, (0, 255, 0), 2) # Kontur Utama (Hijau)
    
    # Gambar lubang yang valid saja (Biru)
    for idx in valid_holes_idx:
        cv2.drawContours(viz_img, [contours[idx]], -1, (255, 0, 0), 2)

    # 6. Cropping untuk K-Means
    pad = 5
    img_h, img_w = original_rgb.shape[:2]
    y1, y2 = max(0, y-pad), min(img_h, y+h+pad)
    x1, x2 = max(0, x-pad), min(img_w, x+w+pad)
    cropped = original_rgb[y1:y2, x1:x2]

    if cropped.size == 0:
        cropped = original_rgb[y:y+h, x:x+w]

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
        'viz_image': viz_img,
        'cropped_rgb': cropped,
        'contour': cnt,          
        'all_contours': contours,
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

    # Penentuan Label Dasar
    if holes >= 1:
        return "Severe Insect Damage", 5, 95.0
    if intensity < 95:
        return "Dry Cherry", 5, 90.0
    if rg_ratio > 1.25:
        return "Partial Sour", 4, 88.0
    if intensity > 185:
        return "Immature", 2, 85.0
    if solidity < 0.94 or circ < 0.75:
        return "Withered", 2, 75.0

    return "Normal", 1, 95.0