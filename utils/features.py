# utils/features.py
import cv2
import numpy as np

def extract_all_features(binary_image, blur_image, original_rgb):
    """
    IMPLEMENTASI W06: FEATURE DETECTION & EXTRACTION
    Menggabungkan deteksi tepi, pencarian kontur, geometri, dan Hough Transform.
    """
    # =========================================================
    # TAHAP 1: FEATURE EXTRACTION (Mencari Kontur & Bentuk)
    # =========================================================
    # Mencari batas luar objek (kontur) dari gambar biner
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Menyiapkan wadah penyimpanan hasil (Dictionary)
    features = {
        'area': 0, 
        'perimeter': 0, 
        'circularity': 0, 
        'aspect_ratio': 0,
        'center_cut_lines': 0,            # Hasil deteksi garis tengah
        'viz_image': original_rgb.copy(), # Gambar visualisasi UI
        'masked_rgb': None,               # Gambar tanpa background
        'cropped_rgb': None,              # Gambar yang sudah di-zoom
        'is_valid': False
    }
    
    if contours:
        # Ambil kontur paling besar (fokus hanya ke biji kopi utama)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Ekstraksi Data Angka (Numerik)
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        features['area'] = area
        features['perimeter'] = perimeter
        
        # Rumus Kebulatan: (4 * pi * Luas) / (Keliling^2)
        if perimeter > 0:
            features['circularity'] = (4 * np.pi * area) / (perimeter ** 2)
            
        # Kotak Pembatas (Bounding Box) untuk Aspek Rasio & Cropping
        x, y, w, h = cv2.boundingRect(main_contour)
        if h > 0:
            features['aspect_ratio'] = float(w) / h
            
        # Ekstraksi Area Objek (Masking & Cropping untuk W07)
        mask = np.zeros(original_rgb.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [main_contour], -1, 255, -1)
        
        masked_full = cv2.bitwise_and(original_rgb, original_rgb, mask=mask)
        features['masked_rgb'] = masked_full
        features['cropped_rgb'] = masked_full[y:y+h, x:x+w]
        
        # =========================================================
        # TAHAP 2: FEATURE DETECTION (Canny Edge & Hough Transform)
        # =========================================================
        # A. Deteksi Tepi (Canny) pada area biji kopi saja
        edges = cv2.Canny(blur_image, 50, 150)
        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # B. Deteksi Garis (Hough Transform) untuk mencari belahan kopi
        # minLineLength: panjang minimal garis agar dihitung
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=30, minLineLength=15, maxLineGap=10)
        
        if lines is not None:
            features['center_cut_lines'] = len(lines)

        # =========================================================
        # TAHAP 3: VISUALISASI MATA KOMPUTER (Untuk UI Streamlit)
        # =========================================================
        # 1. Gambar Kontur (Hijau Tebal)
        cv2.drawContours(features['viz_image'], [main_contour], -1, (0, 255, 0), 2)
        
        # 2. Gambar Kotak Pembatas (Ungu)
        cv2.rectangle(features['viz_image'], (x, y), (x+w, y+h), (255, 0, 255), 2)
        
        # 3. Gambar Titik Tengah / Center of Mass (Merah)
        M = cv2.moments(main_contour)
        if M['m00'] != 0:
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            cv2.circle(features['viz_image'], (cx, cy), 5, (0, 0, 255), -1) 
            
        # 4. Gambar Garis Belahan Tengah (Kuning)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(features['viz_image'], (x1, y1), (x2, y2), (0, 255, 255), 2)
                
        features['is_valid'] = True
        
    return features