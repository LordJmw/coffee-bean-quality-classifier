import cv2
import numpy as np
import os
import shutil

# --- CONFIGURATION ---
FILE_BESAR = 'Normales.jpg'  # Pastikan file ini ada di folder yang sama
OUTPUT_FOLDER = 'Normal_Cropped'
TARGET_SIZE = (500, 500)
# Nilai Threshold (batas warna) agar gap antar biji terlihat.
# Jika hasil masih gagal, coba ubah angka 140 ini ke 130 atau 150.
TRESHOLD_VALUE = 140 

# --- SCRIPT JALAN ---
# 1. Bersihkan folder output lama agar tidak menumpuk
if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER)

# 2. Load gambar
img = cv2.imread(FILE_BESAR)
if img is None:
    print(f"Gagal memuat gambar! Pastikan nama file '{FILE_BESAR}' benar.")
    exit()

# 3. Preprocessing Lanjutan (Lebih Agresif)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Menggunakan blur yang lebih kuat untuk menghaluskan tekstur biji
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Thresholding yang lebih ketat untuk benar-benar memisahkan biji
_, thresh = cv2.threshold(blurred, TRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)

# Perlu dilakukan Operasi Morfologi (Morphology) untuk:
# Menutup lubang kecil di tengah biji, dan memisahkan gap antar biji
kernel = np.ones((5,5), np.uint8)
# Erosi: Memperkecil objek agar biji yang nempel jadi terpisah
eroded = cv2.erode(thresh, kernel, iterations=2)
# Dilasi: Memperbesar sedikit objek untuk mengembalikan ukuran asli biji
# (setelah gap-nya sudah benar-benar terpisah)
morph = cv2.dilate(eroded, kernel, iterations=1)


# 4. Cari kontur (bentuk biji)
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5. Iterasi dan Cropping ke 500x500
count = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Filter: Abaikan objek yang terlalu kecil (noise) atau terlalu besar (area nampan)
    if w > 40 and h > 40 and w < 400 and h < 400:
        # Ambil potongan biji asli
        crop = img[y:y+h, x:x+w]
        
        # Buat kanvas putih 500x500
        canvas = np.full((TARGET_SIZE[0], TARGET_SIZE[1], 3), 255, dtype=np.uint8)
        
        # Resize biji agar proporsional
        # Mencari skala agar sisi terpanjang biji jadi sekitar 350px
        scale = 350 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        # Cegah resize jadi 0x0
        if new_w <= 0 or new_h <= 0: continue
            
        resized_crop = cv2.resize(crop, (new_w, new_h))
        
        # Tempelkan ke tengah kanvas putih
        y_offset = (TARGET_SIZE[0] - new_h) // 2
        x_offset = (TARGET_SIZE[1] - new_w) // 2
        
        # Amankan agar offset tidak negatif (terjadi jika skala gagal)
        if y_offset < 0 or x_offset < 0: continue
            
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_crop
        
        # Simpan hasil
        cv2.imwrite(f'{OUTPUT_FOLDER}/normal_{count}.jpg', canvas)
        count += 1

print(f"--- SELESAI ---")
print(f"Sukses mendeteksi dan memotong {count} biji kopi ke folder '{OUTPUT_FOLDER}'.")
print(f"Total terdapat {len(contours)} kontur awal (termasuk noise).")