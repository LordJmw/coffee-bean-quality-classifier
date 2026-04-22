# utils/features.py
import cv2
import numpy as np

def extract_all_features(binary_image, blur_image, original_rgb):
    """
    IMPLEMENTASI W06: FEATURE DETECTION & EXTRACTION
    (Dilengkapi Deteksi Lubang Hama dan Retakan)
    """
    # PERUBAHAN: Gunakan RETR_TREE untuk mendapatkan hirarki (mendeteksi lubang di dalam biji)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    features = {
        'area': 0, 'perimeter': 0, 'circularity': 0, 'aspect_ratio': 0,
        'center_cut_lines': 0,            
        'holes_count': 0,                 # METRIK BARU: Jumlah Lubang
        'viz_image': original_rgb.copy(), 
        'masked_rgb': None,               
        'cropped_rgb': None,              
        'is_valid': False
    }
    
    if contours:
        # Cari indeks kontur terbesar (ini adalah outline luar biji kopi)
        main_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        main_contour = contours[main_idx]
        
        # Ekstraksi Area & Perimeter
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        features['area'] = area
        features['perimeter'] = perimeter
        
        if perimeter > 0:
            features['circularity'] = (4 * np.pi * area) / (perimeter ** 2)
            
        x, y, w, h = cv2.boundingRect(main_contour)
        if h > 0:
            features['aspect_ratio'] = float(w) / h
            
        # --- DETEKSI LUBANG (INSECT DAMAGE) ---
        holes = 0
        if hierarchy is not None:
            for i in range(len(contours)):
                # Jika "parent" dari kontur ini adalah kontur utama (berarti dia ada di dalam biji)
                if hierarchy[0][i][3] == main_idx:
                    # Filter: Abaikan debu super kecil, hanya hitung lubang yang agak besar
                    if cv2.contourArea(contours[i]) > 10: 
                        holes += 1
                        # Gambar lubang tersebut dengan warna MERAH agar terlihat di UI
                        cv2.drawContours(features['viz_image'], contours, i, (255, 0, 0), 2)
        features['holes_count'] = holes

        # Masking untuk W07
        mask = np.zeros(original_rgb.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [main_contour], -1, 255, -1)
        masked_full = cv2.bitwise_and(original_rgb, original_rgb, mask=mask)
        features['masked_rgb'] = masked_full
        features['cropped_rgb'] = masked_full[y:y+h, x:x+w]
        
        # --- DETEKSI RETAKAN (HOUGH TRANSFORM) ---
        edges = cv2.Canny(blur_image, 50, 150)
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=30, minLineLength=15, maxLineGap=10)
        
        if lines is not None:
            features['center_cut_lines'] = len(lines)

        # Visualisasi Utama
        cv2.drawContours(features['viz_image'], [main_contour], -1, (0, 255, 0), 2) # Outline Hijau
        cv2.rectangle(features['viz_image'], (x, y), (x+w, y+h), (255, 0, 255), 2) # Bounding Box Ungu
        
        # Gambar Garis Retakan Kuning
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(features['viz_image'], (x1, y1), (x2, y2), (0, 255, 255), 2)
                
        features['is_valid'] = True
        
    return features