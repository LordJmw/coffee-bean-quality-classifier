import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd # Pindahkan import ke atas agar lebih bersih

# Import modul yang sudah dibuat
from utils.dataset_loader import get_dataset_samples, get_dataset_statistics, load_image_from_dataset, DATASET_INFO
from utils.preprocessing import preprocess_pipeline
from utils.morphology import apply_morphology
from utils.features import extract_all_features
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
if 'input_mode' not in st.session_state:
    st.session_state['input_mode'] = None

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.header("📋 Menu")
    
    # Opsi input gambar
    input_mode = st.radio(
        "Pilih sumber gambar:",
        ["📤 Upload Gambar Sendiri", "📦 Gunakan Dataset Sample"]
    )
    
    st.divider()
    
    # ===== PARAMETER PREPROCESSING =====
    st.subheader("⚙️ Parameter Preprocessing")
    
    # Opsi resize
    resize_option = st.selectbox(
        "Ukuran Resize",
        ["224x224 (Rekomendasi)", "256x256", "512x512"],
        index=0
    )
    
    # Parse ukuran
    if resize_option == "224x224 (Rekomendasi)":
        target_size = (224, 224)
    elif resize_option == "256x256":
        target_size = (256, 256)
    else:
        target_size = (512, 512)
    
    interpolation_option = st.selectbox(
        "Metode Interpolasi",
        [
            "Nearest Neighbor", 
            "Bilinear interpolation", 
            "Bicubic interpolation", 
            "Area-based", 
            "Lanczos", 
            "Exact Bilinear interpolation", 
            "Exact Nearest Neighbor"
        ],
        index=3 # Default: Area-based
    )

    blur_kernel = st.slider("Gaussian Blur Kernel", 3, 11, 5, step=2)
    open_kernel = st.slider("Opening Kernel", 2, 7, 3)
    close_kernel = st.slider("Closing Kernel", 3, 9, 5)
    
    st.divider()
    
    # Jika pilih dataset sample
    if input_mode == "📦 Gunakan Dataset Sample":
        st.subheader("📂 Pilih Sampel Dataset")
        
        dataset_path = "data/training dataset"
        samples = get_dataset_samples(dataset_path)
        
        if samples:
            selected_class = st.selectbox(
                "Pilih kelas cacat:",
                list(samples.keys())
            )
            
            if selected_class:
                class_info = samples[selected_class]
                grade_emoji = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "⚫"}
                st.info(f"{grade_emoji.get(class_info['grade'], '')} Grade {class_info['grade']}: {class_info['description']}")
                st.caption(f"Total: {class_info['total_images']} gambar | Label: {class_info['label']}")
                
                selected_image = st.selectbox(
                    "Pilih gambar:",
                    class_info['images']
                )
                
                if selected_image:
                    image_path = os.path.join(class_info['folder'], selected_image)
                    st.session_state['sample_image_path'] = image_path
                    st.session_state['expected_grade'] = class_info['grade']
                    st.session_state['expected_class'] = selected_class
                    st.success(f"✅ {selected_image}")
        else:
            st.warning("⚠️ Dataset belum di-download!")

# Statistik Dataset
dataset_path = "data/training dataset"
stats = get_dataset_statistics(dataset_path)
if stats['total_images'] > 0:
    with st.sidebar:
        st.success(f"✅ Dataset terdeteksi: {stats['total_images']} gambar")

# ============================================
# MAIN CONTENT
# ============================================

col1, col2 = st.columns([1, 1])

with col1:
    if input_mode == "📤 Upload Gambar Sendiri":
        uploaded_file = st.file_uploader(
            "Upload gambar biji kopi",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Input", use_container_width=True)
            st.success("✅ Gambar berhasil diupload!")
            st.session_state['input_image'] = np.array(image)
            st.session_state['input_mode'] = 'upload'
        else:
            st.warning("⚠️ Silakan upload gambar terlebih dahulu")

    elif input_mode == "📦 Gunakan Dataset Sample":
        if 'sample_image_path' in st.session_state:
            image_path = st.session_state['sample_image_path']
            if os.path.exists(image_path):
                image_rgb = load_image_from_dataset(image_path)
                if image_rgb is not None:
                    st.image(image_rgb, caption=f"Sampel: {os.path.basename(image_path)}", use_container_width=True)
                    st.success(f"✅ Gambar dari dataset berhasil dimuat!")
                    st.session_state['input_image'] = image_rgb
                    st.session_state['input_mode'] = 'sample'
                else:
                    st.error("Gagal membaca gambar!")
            else:
                st.error(f"File tidak ditemukan: {image_path}")
        else:
            st.info("👈 Silakan pilih gambar dari sidebar")

with col2:
    if st.session_state['input_image'] is not None:
        st.subheader("📊 Informasi Gambar")
        img = st.session_state['input_image']

        original_h, original_w = img.shape[:2]
        st.write(f"**Dimensi Asli:** {original_w} x {original_h} piksel")
        st.write(f"**Dimensi Preprocessing:** {target_size[0]} x {target_size[1]} piksel")

        if len(img.shape) == 3:
            st.write(f"**Channel:** {img.shape[2]} (RGB)")

        if original_w != target_size[0] or original_h != target_size[1]:
            if original_w > target_size[0]:
                st.caption(f"⬇️ Downscale: {original_w}x{original_h} → {target_size[0]}x{target_size[1]}")
            else:
                st.caption(f"⬆️ Upscale: {original_w}x{original_h} → {target_size[0]}x{target_size[1]}")

        if input_mode == "📦 Gunakan Dataset Sample" and 'expected_grade' in st.session_state:
            grade_emoji = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "⚫"}
            st.write(f"**Expected Grade:** {grade_emoji.get(st.session_state['expected_grade'], '')} {st.session_state['expected_grade']}")
            if 'expected_class' in st.session_state:
                st.caption(f"Kelas: {st.session_state['expected_class']}")

        st.divider()
        st.subheader("🔬 Hasil Preprocessing (M1-M7)")

        with st.spinner("Memproses gambar..."):
            try:
                # M1 & M3: Preprocessing
                preprocess_results = preprocess_pipeline(
                    img, 
                    target_size=target_size,
                    blur_kernel=(blur_kernel, blur_kernel),
                    interpolation_method=interpolation_option
                )
                
                # M4: Morfologi
                morph_results = apply_morphology(
                    preprocess_results['binary'],
                    open_kernel=(open_kernel, open_kernel),
                    close_kernel=(close_kernel, close_kernel)
                )
                
                # M6: Feature Extraction
                geom_features = extract_all_features(morph_results['closing'], preprocess_results['blur'], preprocess_results['rgb'])
                
                # M7: K-Means Clustering
                if geom_features['is_valid'] and geom_features['cropped_rgb'] is not None:
                    kmeans_img, color_centers, cluster_stats = analyze_color_kmeans(geom_features['cropped_rgb'], k=3)
                else:
                    kmeans_img, color_centers, cluster_stats = None, None, None

                # Setup Tabs (FIXED SYNTAX)
                tabs = st.tabs([
                    "M2: Interpolation",
                    "M1: Grayscale", 
                    "M3: Gaussian Blur", 
                    "M4: Opening", 
                    "M4: Closing",
                    "M5: Edge & Contour",
                    "M6: Geometric Features",
                    "M7: K-Means Clustering",
                    "Tabel Fitur"
                ])
                
                tab_interp, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = tabs
                
                with tab_interp:
                    st.image(preprocess_results['rgb'], caption=f"{interpolation_option} Interpolation", use_container_width=True, clamp=True)
                    st.caption("M2: Interpolasi: Mengisi atau meringkas pixel untuk resizing.")

                with tab1:
                    st.image(preprocess_results['gray'], caption="Grayscale (8-bit)", use_container_width=True, clamp=True)
                    with st.expander("📊 Lihat Representasi Numerik"):
                        st.dataframe(preprocess_results['gray'][:10, :10])
                    
                with tab2:
                    st.image(preprocess_results['blur'], caption="Gaussian Blur", use_container_width=True, clamp=True)
                    
                with tab3:
                    st.image(morph_results['opening'], caption="Opening", use_container_width=True, clamp=True)
                    
                with tab4:
                    st.image(morph_results['closing'], caption="Closing", use_container_width=True, clamp=True)

                with tab5:
                    st.markdown("### M5: Feature Detection (Edges & Contours)")
                    edges = cv2.Canny(preprocess_results['blur'], 50, 150)
                    contour_img = preprocess_results['rgb'].copy()
                    contours, _ = cv2.findContours(morph_results['closing'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
                    
                    c_edge, c_contour = st.columns(2)
                    c_edge.image(edges, caption="Canny Edge Detection", use_container_width=True, clamp=True)
                    c_contour.image(contour_img, caption="Contour Detection", use_container_width=True)
                    
                    if contours:
                        main_contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(main_contour)
                        st.markdown("---")
                        st.metric("Jumlah Kontur", len(contours))
                        st.metric("Luas Kontur Utama", f"{area:.0f} px")

                with tab6:
                    if geom_features['is_valid']:
                        st.image(geom_features['viz_image'], caption="M6: Visualisasi Ekstraksi Ciri", use_container_width=True)
                        c1, c2 = st.columns(2)
                        c1.metric("Area (Luas)", f"{geom_features['area']:.0f} px")
                        c1.metric("Perimeter (Keliling)", f"{geom_features['perimeter']:.2f} px")
                        c2.metric("Circularity", f"{geom_features['circularity']:.3f}")
                        c2.metric("Hough Lines", f"{geom_features['center_cut_lines']}")
                    else:
                        st.error("Gagal mendeteksi ciri fisik objek.")

                with tab7:
                    if kmeans_img is not None:
                        st.image(kmeans_img, caption="Hasil Segmentasi Warna (M7)", use_container_width=True, clamp=True)
                        if cluster_stats:
                            for idx, stat in cluster_stats.items():
                                r, g, b = stat['color']
                                st.markdown(
                                    f"<div style='display:flex; align-items:center; margin-bottom:10px;'>"
                                    f"<div style='width:30px; height:30px; background-color:rgb({r},{g},{b}); border:1px solid #fff; margin-right:15px;'></div>"
                                    f"<b>Klaster {idx+1} :</b> &nbsp; {stat['percentage']:.1f}% area"
                                    f"</div>", 
                                    unsafe_allow_html=True
                                )
                    else:
                        st.error("Gagal melakukan segmentasi warna.")
                
                with tab8:
                    st.markdown("### 🔍 Analisis Fitur & Standar Mutu")
                    
                    if geom_features['is_valid'] and kmeans_img is not None:
                        # --- BAGIAN 1: VISUALISASI ---
                        st.markdown("#### 1. Visualisasi Analisis")
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        with m_col1: st.image(geom_features['viz_image'], caption="Geometri", use_container_width=True)
                        with m_col2: st.image(morph_results['closing'], caption="Pola Retakan", use_container_width=True)
                        with m_col3: st.image(kmeans_img, caption="Klaster Warna", use_container_width=True)
                        with m_col4:
                            edges = cv2.Canny(preprocess_results['blur'], 50, 150)
                            st.image(edges, caption="Tekstur Permukaan", use_container_width=True)

                        st.divider()

                        # --- BAGIAN 2: LOGIKA PENILAIAN (PUSAT DATA) ---
                        total_score = 100
                        # CRITICAL: Inisialisasi list ini hanya SEKALI di awal
                        analysis_details = []

                        # A. ANALISIS FISIK (Circularity)
                        if geom_features['circularity'] < 0.60:
                            p = 25
                            total_score -= p
                            analysis_details.append(("error", f"❌ **Bentuk Tidak Utuh:** Circularity rendah ({geom_features['circularity']:.2f}). Indikasi Biji Pecah (Broken). (-{p} pts)"))
                        else:
                            analysis_details.append(("info", f"✅ **Bentuk Normal:** Circularity ideal ({geom_features['circularity']:.2f}). (+0 pts)"))
                        
                        # B. ANALISIS RETAKAN (Hough Lines)
                        if geom_features['center_cut_lines'] > 6:
                            p = 30
                            total_score -= p
                            analysis_details.append(("error", f"❌ **Retakan Parah:** Terdeteksi {geom_features['center_cut_lines']} jalur retakan mendalam. (-{p} pts)"))
                        elif geom_features['center_cut_lines'] > 2:
                            p = 10
                            total_score -= p
                            analysis_details.append(("warning", f"⚠️ **Cacat Tekstur:** Terdeteksi {geom_features['center_cut_lines']} retakan halus. (-{p} pts)"))
                        else:
                            analysis_details.append(("info", f"✅ **Tekstur Utuh:** Tidak ditemukan retakan signifikan. (+0 pts)"))

                        # C. ANALISIS WARNA (Logika Rasio)
                        black_area = 0
                        sour_area = 0
                        for idx, stat in cluster_stats.items():
                            r, g, b = stat['color']
                            total_rgb = r + g + b
                            if total_rgb == 0: continue 
                            g_ratio = g / total_rgb
                            r_to_g_ratio = r / g if g > 0 else 2.0
                            
                            if total_rgb < 130 and r < 50:
                                black_area += stat['percentage']
                            elif g_ratio < 0.35 and r_to_g_ratio > 1.05:
                                sour_area += stat['percentage']

                        if sour_area > 10:
                            p = 25
                            total_score -= p
                            analysis_details.append(("warning", f"⚠️ **Cacat Partial Sour:** Area cokelat/keruh {sour_area:.1f}%. (-{p} pts)"))
                        
                        if black_area > 5:
                            p = 40
                            total_score -= p
                            analysis_details.append(("error", f"❌ **Cacat Full Black:** Area hitam pekat {black_area:.1f}%. (-{p} pts)"))
                        
                        # Hanya tambah pesan "Warna Normal" jika tidak ada cacat warna
                        if sour_area <= 10 and black_area <= 5:
                            analysis_details.append(("info", "✅ **Warna Normal:** Komponen Hijau-Zaitun dominan dan stabil. (+0 pts)"))

                        # --- BAGIAN RENDER (DI SINI SEMUA DETAIL DITAMPILKAN) ---
                        st.markdown("#### 2. Rincian Penilaian")
                        # Pastikan loop ini berada di luar blok warna agar merender SEMUA isi analysis_details
                        for style, message in analysis_details:
                            if style == "info": st.info(message)
                            elif style == "warning": st.warning(message)
                            else: st.error(message)

                        st.divider()

                        # --- BAGIAN 3: SKOR AKHIR ---
                        final_score = max(total_score, 0)
                        st.markdown(f"## **Skor Akhir Mutu: {final_score}**")
                        
                        if final_score >= 80: st.success("🟢 **HASIL: GRADE 1-2 (MUTU LAYAK)**")
                        elif final_score >= 60: st.warning("🟡 **HASIL: GRADE 3 (MUTU RENDAH)**")
                        else: st.error("🔴 **HASIL: GRADE 4-5 (REJECT/CACAT TOTAL)**")

                        # Export CSV
                        summary_df = pd.DataFrame({
                            'Metrik': ['Circularity', 'Hough Lines', 'Black %', 'Sour %', 'Total Skor'],
                            'Nilai': [f"{geom_features['circularity']:.3f}", geom_features['center_cut_lines'], f"{black_area:.1f}%", f"{sour_area:.1f}%", final_score]
                        })
                        st.download_button("📥 Ekspor Data Analisis", summary_df.to_csv(index=False), "analisis_kopi.csv")
             

            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
    else:
        st.info("Gambar akan ditampilkan di sini setelah diupload/dipilih")

# Footer
st.divider()
st.caption("🎓 Tugas Computer Vision - Implementasi M1-M7")