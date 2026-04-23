import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# Import modul internal
from utils.dataset_loader import get_dataset_samples, get_dataset_statistics, load_image_from_dataset
from utils.preprocessing import preprocess_pipeline
from utils.morphology import apply_morphology
from utils.features import classify_coffee_bean, extract_all_features
from utils.clustering import analyze_color_kmeans

# Page config
st.set_page_config(
    page_title="Klasifikasi Biji Kopi",
    page_icon="☕",
    layout="wide"
)

# Header
st.title("☕ Klasifikasi Mutu Biji Kopi Mentah")
st.markdown("### Ekstraksi Fitur Morfologi & Geometri dengan Preprocessing Citra Digital")

# Inisialisasi session state
if 'input_image' not in st.session_state:
    st.session_state['input_image'] = None

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.header("📋 Menu")
    input_mode = st.radio("Pilih sumber gambar:", ["📤 Upload Gambar Sendiri", "📦 Gunakan Dataset Sample"])
    st.divider()
    
    st.subheader("⚙️ Parameter Preprocessing")
    resize_option = st.selectbox("Ukuran Resize", ["224x224 (Rekomendasi)", "256x256", "512x512"], index=0)
    # Parsing target size
    t_size = int(resize_option.split('x')[0])
    target_size = (t_size, t_size)
    
    interpolation_option = st.selectbox("Metode Interpolasi", 
        ["Nearest Neighbor", "Bilinear interpolation", "Bicubic interpolation", "Area-based", "Lanczos"], index=3)

    blur_kernel = st.slider("Gaussian Blur Kernel", 3, 11, 5, step=2)
    open_kernel = st.slider("Opening Kernel", 2, 7, 3)
    close_kernel = st.slider("Closing Kernel", 3, 9, 3) # Default dikecilkan agar lubang tidak tertutup
    
    st.divider()
    
    if input_mode == "📦 Gunakan Dataset Sample":
        st.subheader("📂 Pilih Sampel Dataset")
        dataset_path = "data/training dataset"
        samples = get_dataset_samples(dataset_path)
        
        if samples:
            selected_class = st.selectbox("Pilih kelas cacat:", list(samples.keys()))
            class_info = samples[selected_class]
            st.info(f"Label: {class_info['label']} | Grade: {class_info['grade']}")
            
            selected_image = st.selectbox("Pilih gambar:", class_info['images'])
            if selected_image:
                image_path = os.path.join(class_info['folder'], selected_image)
                st.session_state['sample_image_path'] = image_path
        else:
            st.warning("⚠️ Dataset tidak ditemukan!")

# ============================================
# MAIN CONTENT
# ============================================
col1, col2 = st.columns([1, 1])

with col1:
    if input_mode == "📤 Upload Gambar Sendiri":
        uploaded_file = st.file_uploader("Upload gambar biji kopi", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Input", use_container_width=True)
            st.session_state['input_image'] = np.array(image)
    else:
        if 'sample_image_path' in st.session_state:
            image_rgb = load_image_from_dataset(st.session_state['sample_image_path'])
            if image_rgb is not None:
                st.image(image_rgb, caption=f"Sampel: {os.path.basename(st.session_state['sample_image_path'])}", use_container_width=True)
                st.session_state['input_image'] = image_rgb

with col2:
    if st.session_state['input_image'] is not None:
        img = st.session_state['input_image']
        with st.spinner("Memproses analisis..."):
            try:
                # --- STRATEGI SINKRONISASI ---
                # M1-M3: Preprocessing (Semua dikerjakan pada skala target_size)
                preprocess_results = preprocess_pipeline(
                    img, 
                    target_size=target_size, 
                    blur_kernel=(blur_kernel, blur_kernel), 
                    interpolation_method=interpolation_option
                )
                
                # M4: Morphology (Gunakan binary dari hasil resize)
                morph_results = apply_morphology(
                    preprocess_results['binary'], 
                    open_kernel=(open_kernel, open_kernel), 
                    close_kernel=(close_kernel, close_kernel)
                )
                
                # M6: Feature Extraction 
                # PENTING: Gunakan preprocess_results['rgb'] (BUKAN original_rgb) 
                # agar koordinat kontur matching dengan gambar warnanya.
                geom_features = extract_all_features(
                    morph_results['closing'], 
                    preprocess_results['blur'], 
                    preprocess_results['rgb'],    # Koordinat sinkron
                    preprocess_results['gray']   # Koordinat sinkron
                )
                geo_overlay = preprocess_results['rgb'].copy()
                detection_view = preprocess_results['rgb'].copy()
                if geom_features['is_valid']:
                    # 1. Gambar Kontur Utama (Garis Hijau)
                    if 'contour' in geom_features:
                        cv2.drawContours(geo_overlay, [geom_features['contour']], -1, (0, 255, 0), 2)
                    
                    # 2. Gambar Kontur Lubang (Garis Merah/Biru)
                    # Pastikan extract_all_features mengembalikan 'all_contours'
                    if 'all_contours' in geom_features:
                        # Mulai dari indeks 1 karena indeks 0 biasanya kontur luar
                        for i in range(1, len(geom_features['all_contours'])):
                            cv2.drawContours(geo_overlay, geom_features['all_contours'], i, (255, 0, 0), 2)
                    
                    # 3. Bounding Box (Opsional, garis tipis)
                    cnt = geom_features['contour']
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(geo_overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)

                # M7: K-Means
                kmeans_img, color_centers, cluster_stats = analyze_color_kmeans(
                    geom_features['cropped_rgb'], 
                    k=3
                ) if geom_features['is_valid'] and geom_features['cropped_rgb'] is not None else (None, None, None)

                tabs = st.tabs(["Proses Citra", "Hasil Ekstraksi", "Analisis Warna", "Penilaian Akhir"])
                
                with tabs[0]:
                    st.subheader(f"🔍 Deteksi Geometri & Morfologi")
                    c1, c2 = st.columns(2)
                    with c1:
                        # Gambar ini sekarang menampilkan representasi geometris (tepi & lubang)
                        st.image(geo_overlay, 
                                 caption=f"Representasi Geometri (Tepi & Lubang) - {interpolation_option}", 
                                 use_container_width=True)
                    with c2:
                        st.image(morph_results['closing'], 
                                 caption="Binary Mask (Hasil Morfologi)", 
                                 use_container_width=True)
                    
                    st.info("💡 **Keterangan:** Garis **Hijau** (Tepi Luar), Garis **Merah** (Deteksi Lubang/Cacat).")
                    
                    st.divider()
                    st.markdown("**Langkah-Langkah Intermediate:**")
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.image(preprocess_results['gray'], caption="1. Grayscale")
                    c_m2.image(preprocess_results['blur'], caption="2. Noise Reduction")
                    if geom_features['is_valid']:
                        c_m3.image(geom_features['cropped_rgb'], caption="3. Hasil Crop (ROI)")
                
                with tabs[1]:
                    # (Kode Tab 1 Hasil Ekstraksi tetap menggunakan versi st.metric yang sudah kita buat)
                    st.subheader("📊 Parameter Geometri Biji")
                    if geom_features['is_valid']:
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric(label="Luas Biji (Area)", value=f"{geom_features['area']:.0f} px")
                            st.caption("Total piksel tubuh biji.")
                        with m2:
                            sol_pct = geom_features['solidity'] * 100
                            st.metric(label="Kepadatan (Solidity)", value=f"{sol_pct:.1f}%")
                            st.caption("Mendeteksi kerutan/withered.")
                        with m3:
                            st.metric(label="Bentuk (Circularity)", value=f"{geom_features['circularity']:.2f}")
                            st.caption("Nilai 1.0 = Bulat Sempurna.")
                        
                        st.divider()
                        m4, m5, m6 = st.columns(3)
                        with m4:
                            st.metric(label="Rasio Panjang (AR)", value=f"{geom_features['aspect_ratio']:.2f}")
                            st.caption("Panjang vs Lebar (Broken detection).")
                        with m5:
                            ext_pct = geom_features['extent'] * 100
                            st.metric(label="Kepenuhan (Extent)", value=f"{ext_pct:.1f}%")
                            st.caption("Persentase pengisian bounding box.")
                        with m6:
                            hole_count = geom_features.get('holes_count', 0)
                            st.metric(label="Lubang Serangga", value=f"{hole_count} Titik", 
                                      delta="Terdeteksi" if hole_count > 0 else "Aman", 
                                      delta_color="inverse" if hole_count > 0 else "normal")
                    else:
                        st.error("⚠️ Objek tidak terdeteksi dengan jelas.")

                with tabs[2]:
                    if kmeans_img is not None:
                        st.image(kmeans_img, caption="Segmentasi Warna K-Means", use_container_width=True)
                        cols = st.columns(len(cluster_stats))
                        
                        for i, (idx, stat) in enumerate(cluster_stats.items()):
                            with cols[i]:
                                # 1. Ambil data warna
                                r, g, b = stat['color']
                                
                                # 2. Logika Penentuan Label (Keterangan)
                                if r > 185 and g > 185 and b > 185:
                                    label = "⚪ Glare"
                                    color_theme = "gray"
                                elif r / (g if g > 0 else 1) > 1.28:
                                    label = "🟤 Sour"
                                    color_theme = "red"
                                elif np.std([r, g, b]) < 22:
                                    label = "🔘 Pucat"
                                    color_theme = "orange"
                                else:
                                    label = "🟢 Normal"
                                    color_theme = "green"

                                # 3. Tampilkan kotak warna
                                color_box = np.zeros((50, 100, 3), dtype=np.uint8)
                                color_box[:] = [r, g, b]
                                st.image(color_box, caption=f"Klaster {idx+1}")
                                
                                # 4. Tampilkan Statistik dan Label
                                st.write(f"**{stat['percentage']:.1f}%** area")
                                st.caption(f"RGB: {list(stat['color'])}")
                                
                                # Menampilkan label dengan st.code atau st.success agar menonjol
                                st.markdown(f":{color_theme}[**{label}**]")
                    else:
                        st.warning("⚠️ Klaster warna tidak muncul. Pastikan kontur biji terdeteksi di Tab 2.")

                with tabs[3]:
                    if geom_features['is_valid']:
                        current_score = 100
                        logs = []

                        # --- 1. DETEKSI FISIK: BROKEN (Grade 4) ---
                        aspect_ratio = geom_features.get('aspect_ratio', 1.0)
                        extent = geom_features.get('extent', 0.8)
                        
                        if aspect_ratio > 1.7 or aspect_ratio < 0.65 or extent < 0.68:
                            penalty = 75 # Skor jadi 25
                            current_score -= penalty
                            logs.append(f"❌ **Fisik:** Broken/Pecah (AR: {aspect_ratio:.2f}) [-{penalty}]")

                        # --- 2. ANALISIS WARNA: WITHERED (G2) & PARTIAL SOUR (G3) ---
                        if cluster_stats:
                            valid_ratios = []
                            avg_saturation = []
                            for idx, stat in cluster_stats.items():
                                r, g, b = stat['color']
                                if r > 185 and g > 185 and b > 185: continue
                                valid_ratios.append(r / g if g > 0 else 1.0)
                                avg_saturation.append(np.std([r, g, b]))
                            
                            if valid_ratios:
                                max_rg = max(valid_ratios)
                                mean_sat = np.mean(avg_saturation) if avg_saturation else 50
                                intensity = geom_features.get('mean_intensity', 127)

                                # A. Partial Sour -> Target Grade 3 (Skor 40-59)
                                if max_rg > 1.28: 
                                    penalty = 50
                                    current_score -= penalty
                                    logs.append(f"⚠️ **Warna:** Partial Sour (R/G: {max_rg:.2f}) [-{penalty}]")
                                
                                # B. Withered -> Target Grade 2 (Skor 60-87)
                                elif mean_sat < 22 and intensity > 130:
                                    penalty = 20
                                    current_score -= penalty
                                    logs.append(f"❌ **Warna:** Withered/Layu [-{penalty}]")

                        # --- 3. DETEKSI INTENSITAS: DRY CHERRY (Grade 4) ---
                        intensity = geom_features.get('mean_intensity', 127)
                        if intensity < 95:
                            penalty = 75 # Skor jadi 25
                            current_score -= penalty
                            logs.append(f"❌ **Intensitas:** Dry Cherry/Hitam [-{penalty}]")

                        # --- 4. DETEKSI FISIK TAMBAHAN: WITHERED (G2) & INSECT (G5) ---
                        solidity = geom_features.get('solidity', 1.0)
                        if solidity < 0.94 and current_score > 80: 
                            penalty = 20
                            current_score -= penalty
                            logs.append(f"⚠️ **Fisik:** Mengerut (Sol: {solidity:.3f}) [-{penalty}]")

                        # Severe Insect Damage (Grade 5)
                        holes = geom_features.get('holes_count', 0)
                        if holes > 0:
                            if holes >= 2:
                                penalty = 90 # Skor jadi 10 (G5)
                                logs.append(f"❌ **Hama:** Severe Insect Damage ({holes} lubed) [-{penalty}]")
                            else:
                                penalty = 15 # 1 lubang hanya dikurangi sedikit (Tetap G1/G2 tergantung kondisi lain)
                                logs.append(f"⚠️ **Hama:** Lubang Serangga Ringan (1 lubang) [-{penalty}]")
                            current_score -= penalty

                        # --- FINALISASI 5 GRADE ---
                        final_score = max(0, current_score)
                        st.subheader(f"Total Skor Akhir: {final_score}")

                        if final_score >= 88:
                            st.success("### HASIL: GRADE 1 (NORMAL)")
                        elif final_score >= 60:
                            st.info("### HASIL: GRADE 2 (WITHERED / DEFECT RINGAN)")
                        elif final_score >= 40:
                            st.warning("### HASIL: GRADE 3 (PARTIAL SOUR)")
                        elif final_score >= 20:
                            st.error("### HASIL: GRADE 4 (BROKEN / DRY CHERRY)")
                        else:
                            st.error("### HASIL: GRADE 5 (SEVERE INSECT DAMAGE)")

                        for log in logs: st.write(log)

            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {e}")

st.divider()
