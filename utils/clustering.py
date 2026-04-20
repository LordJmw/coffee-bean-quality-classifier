
# utils/clustering.py
import cv2
import numpy as np

def analyze_color_kmeans(cropped_rgb, k=3):
    """
    IMPLEMENTASI W07: UNSUPERVISED LEARNING (K-MEANS CLUSTERING)
    Menganalisis warna pada biji kopi yang sudah di-crop oleh W06 
    untuk mencari persentase area warna cacat (hitam/jamur).
    """
    # 1. HAPUS BACKGROUND (Penting!)
    # W06 memberikan background hitam pekat [0,0,0]. Kita tidak mau warna hitam meja 
    # ikut terhitung sebagai warna kopi. Jadi kita buat mask untuk mengabaikannya.
    mask_background = (cropped_rgb[:,:,0] == 0) & (cropped_rgb[:,:,1] == 0) & (cropped_rgb[:,:,2] == 0)
    
    # Ambil HANYA piksel yang merupakan bagian dari biji kopi
    bean_pixels = cropped_rgb[~mask_background]
    
    # Jika gambar kosong/error, hentikan proses
    if len(bean_pixels) == 0:
        return cropped_rgb, None, None

    # 2. PERSIAPAN DATA (Ubah ke float32 sesuai syarat algoritma K-Means OpenCV)
    pixel_values = np.float32(bean_pixels)

    # 3. KRITERIA BERHENTI (Algoritma berhenti setelah 100 iterasi atau akurasi mencapai 0.2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 4. JALANKAN K-MEANS CLUSTERING (Unsupervised Learning)
    # Algoritma ini akan secara buta mengelompokkan piksel ke dalam K grup yang paling mirip
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 5. EKSTRAKSI HASIL (Hitung Persentase Luas Penyakit/Warna)
    centers = np.uint8(centers)
    labels = labels.flatten()
    
    cluster_stats = {}
    total_pixels = len(labels)
    
    for i in range(k):
        # Hitung ada berapa piksel yang masuk ke klaster ini
        count = np.sum(labels == i)
        # Jadikan persentase (%)
        percentage = (count / total_pixels) * 100
        
        # Simpan warna RGB asli dari klaster tersebut dan persentasenya
        cluster_stats[i] = {
            'color': centers[i].tolist(),
            'percentage': percentage
        }

    # 6. VISUALISASI MATA KOMPUTER (Rekonstruksi Gambar)
    # Buat kanvas kosong hitam seukuran gambar crop asli
    segmented_image = np.zeros_like(cropped_rgb)
    
    # Warnai kembali area biji kopi HANYA dengan K warna inti hasil clustering
    segmented_image[~mask_background] = centers[labels]

    return segmented_image, centers, cluster_stats