import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import joblib
import pandas as pd

# Import modul internal
from utils.dataset_loader import get_dataset_samples, get_dataset_statistics, load_image_from_dataset
from utils.preprocessing import preprocess_pipeline
from utils.morphology import apply_morphology
from utils.features import classify_coffee_bean, extract_all_features
from utils.clustering import analyze_color_kmeans
from utils.visualization import create_decision_boundary_plot

# Page config
st.set_page_config(
    page_title="Klasifikasi Biji Kopi",
    page_icon="☕",
    layout="wide"
)

@st.cache_resource
def load_ml_model():
    """Load Random Forest model dan scaler"""
    try:
        model = joblib.load("models/coffee_bean_rf.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

ml_model, ml_scaler = load_ml_model()

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

    
    # ========== TAMBAHAN: Mode Klasifikasi ==========
    st.subheader("Metode Klasifikasi")
    if ml_model is not None:
        classification_mode = st.radio(
            "Pilih metode:",
            ["Rule-based", "Machine Learning"],
            help="Rule-based: threshold manual dari makalah | ML: model dengan akurasi 92%"
        )
    else:
        classification_mode = "Rule-based"
        st.warning("⚠️ Model ML belum tersedia. Jalankan train_model.py dulu.")

    st.subheader("⚙️ Parameter Preprocessing")
    resize_option = st.selectbox("Ukuran Resize", ["224x224 (Rekomendasi)", "256x256", "512x512"], index=0)
    t_size = int(resize_option.split('x')[0])
    target_size = (t_size, t_size)

    interpolation_option = st.selectbox("Metode Interpolasi",
        ["Nearest Neighbor", "Bilinear interpolation", "Bicubic interpolation", "Area-based", "Lanczos"], index=3)

    blur_kernel = st.slider("Gaussian Blur Kernel", 3, 11, 5, step=2)
    open_kernel = st.slider("Opening Kernel", 2, 7, 3)
    close_kernel = st.slider("Closing Kernel", 3, 9, 3)

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
# FUNGSI PREDIKSI ML
# ============================================
def predict_with_ml(features, model, scaler):
    """Prediksi menggunakan Random Forest"""
    feature_names = ['area', 'circularity', 'solidity', 'extent', 'aspect_ratio', 
                     'holes_count', 'center_cut_lines', 'mean_intensity', 'red_ratio', 'green_ratio']
    
    X = np.array([[
        features.get('area', 0),
        features.get('circularity', 0),
        features.get('solidity', 0),
        features.get('extent', 0),
        features.get('aspect_ratio', 0),
        features.get('holes_count', 0),
        features.get('center_cut_lines', 0),
        features.get('mean_intensity', 0),
        features.get('red_ratio', 0),
        features.get('green_ratio', 0)
    ]])
    
    X_scaled = scaler.transform(X)
    pred_num = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    confidence = max(proba) * 100
    
    class_names = ['Normal', 'Withered', 'Partial Sour', 'Broken', 'Dry Cherry', 'Severe Insect Damage']
    grade_map = {'Normal': 1, 'Withered': 2, 'Partial Sour': 3, 'Broken': 4, 'Dry Cherry': 4, 'Severe Insect Damage': 5}
    
    return class_names[pred_num], confidence, proba, grade_map[class_names[pred_num]]

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
                # M1-M3: Preprocessing
                preprocess_results = preprocess_pipeline(
                    img,
                    target_size=target_size,
                    blur_kernel=(blur_kernel, blur_kernel),
                    interpolation_method=interpolation_option
                )

                # M4: Morphology
                morph_results = apply_morphology(
                    preprocess_results['binary'],
                    open_kernel=(open_kernel, open_kernel),
                    close_kernel=(close_kernel, close_kernel)
                )

                # M6: Feature Extraction (dengan Canny-based hole detection)
                geom_features = extract_all_features(
                    morph_results['closing'],
                    preprocess_results['blur'],
                    preprocess_results['rgb'],
                    preprocess_results['gray']
                )

                # Visualisasi overlay geometri
                geo_overlay = preprocess_results['rgb'].copy()
                if geom_features['is_valid']:
                    # Kontur utama (Hijau)
                    if 'contour' in geom_features:
                        cv2.drawContours(geo_overlay, [geom_features['contour']], -1, (0, 255, 0), 2)

                    # Lubang Canny (Merah) — lebih akurat, tidak tergantung closing
                    if 'hole_contours' in geom_features:
                        for hc in geom_features['hole_contours']:
                            cv2.drawContours(geo_overlay, [hc], -1, (255, 50, 50), 2)

                    # Bounding Box
                    cnt = geom_features['contour']
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(geo_overlay, (x, y), (x + w, y + h), (0, 200, 200), 1)

                # M7: K-Means
                kmeans_img, color_centers, cluster_stats = analyze_color_kmeans(
                    geom_features['cropped_rgb'],
                    k=3
                ) if geom_features['is_valid'] and geom_features['cropped_rgb'] is not None else (None, None, None)

                if classification_mode == "Machine Learning":
                    tabs = st.tabs(["Proses Citra", "Hasil Ekstraksi", "Analisis Warna", "Penilaian Akhir", "Penilaian Model"])
                else:
                    tabs = st.tabs(["Proses Citra", "Hasil Ekstraksi", "Analisis Warna", "Penilaian Akhir"])

                # ─── TAB 0: Proses Citra ───────────────────────────────────────────
                with tabs[0]:
                    st.subheader("🔍 Deteksi Geometri & Morfologi")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(geo_overlay,
                                 caption=f"Representasi Geometri (Tepi & Lubang) — {interpolation_option}",
                                 use_container_width=True)
                    with c2:
                        st.image(morph_results['closing'],
                                 caption="Binary Mask (Hasil Morfologi)",
                                 use_container_width=True)

                    st.info(
                        "💡 **Keterangan:** Garis **Hijau** = Tepi Luar Biji | "
                        "Garis **Merah** = Lubang/Cacat (Canny Edge Detection) | "
                        "Kotak **Cyan** = Bounding Box"
                    )

                    # Debug: tampilkan info lubang jika terdeteksi
                    if geom_features['is_valid'] and geom_features.get('hole_areas'):
                        with st.expander("🔬 Debug: Info Lubang Terdeteksi"):
                            for i, ha in enumerate(geom_features['hole_areas']):
                                pct = ha / geom_features['area'] * 100 if geom_features['area'] > 0 else 0
                                st.caption(f"Lubang {i+1}: {ha:.0f} px² ({pct:.1f}% luas biji)")

                    st.divider()
                    st.markdown("**Langkah-Langkah Intermediate:**")
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.image(preprocess_results['gray'], caption="1. Grayscale")
                    c_m2.image(preprocess_results['blur'], caption="2. Noise Reduction")
                    if geom_features['is_valid']:
                        c_m3.image(geom_features['cropped_rgb'], caption="3. Hasil Crop (ROI)")

                # ─── TAB 1: Hasil Ekstraksi ────────────────────────────────────────
                with tabs[1]:
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
                            st.metric(
                                label="Lubang Serangga",
                                value=f"{hole_count} Titik",
                                delta="Terdeteksi" if hole_count > 0 else "Aman",
                                delta_color="inverse" if hole_count > 0 else "normal"
                            )
                            st.caption("Dual-track: rongga besar + titik gelap.")

                        st.divider()
                        st.markdown("**Fitur Withered (dari data nyata):**")
                        w1, w2, w3 = st.columns(3)
                        with w1:
                            inten = geom_features.get('mean_intensity', 0)
                            st.metric(label="Mean Intensity", value=f"{inten:.1f}",
                                      delta="Withered jika > 130" if inten > 130 else "Normal range",
                                      delta_color="inverse" if inten > 130 else "normal")
                            st.caption("Normal ~123 | Withered 135–163")
                        with w2:
                            circ_v = geom_features.get('circularity', 0)
                            st.metric(label="Circularity", value=f"{circ_v:.3f}",
                                      delta="Kerut" if circ_v < 0.83 else "Normal",
                                      delta_color="inverse" if circ_v < 0.83 else "normal")
                            st.caption("Normal ~0.856 | Withered 0.74–0.87")
                        with w3:
                            sol_v = geom_features.get('solidity', 0)
                            st.metric(label="Solidity", value=f"{sol_v:.3f}",
                                      delta="Cekung" if sol_v < 0.988 else "Normal",
                                      delta_color="inverse" if sol_v < 0.988 else "normal")
                            st.caption("Normal ~0.990 | Withered 0.957–0.987")
                    else:
                        st.error("⚠️ Objek tidak terdeteksi dengan jelas.")

                # ─── TAB 2: Analisis Warna ─────────────────────────────────────────
                with tabs[2]:
                    if kmeans_img is not None:
                        st.image(kmeans_img, caption="Segmentasi Warna K-Means", use_container_width=True)
                        cols = st.columns(len(cluster_stats))

                        for i, (idx, stat) in enumerate(cluster_stats.items()):
                            with cols[i]:
                                r, g, b = stat['color']

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

                                color_box = np.zeros((50, 100, 3), dtype=np.uint8)
                                color_box[:] = [r, g, b]
                                st.image(color_box, caption=f"Klaster {idx + 1}")
                                st.write(f"**{stat['percentage']:.1f}%** area")
                                st.caption(f"RGB: {list(stat['color'])}")
                                st.markdown(f":{color_theme}[**{label}**]")
                    else:
                        st.warning("⚠️ Klaster warna tidak muncul. Pastikan kontur biji terdeteksi di Tab 2.")

                # ─── TAB 3: Penilaian Akhir ────────────────────────────────────────
                with tabs[3]:
                    if geom_features['is_valid']:

                        # ==========================================
                        # 1. JALANKAN KEDUA METODE (BACKEND)
                        # ==========================================
                        
                        # --- A. Prediksi Machine Learning ---
                        if ml_model is not None:
                            pred_class, confidence, proba, grade = predict_with_ml(geom_features, ml_model, ml_scaler)

                        # --- B. Prediksi Rule-Based ---
                        current_score = 100
                        logs = []
                        detected_class = None
                        
                        # ── 1. DRY CHERRY ──────────────────────────────
                        intensity = geom_features.get('mean_intensity', 127)
                        if intensity < 95:
                            penalty = 75
                            current_score -= penalty
                            detected_class = "DRY_CHERRY"
                            logs.append(f"❌ **Intensitas:** Dry Cherry/Hitam (Inten: {intensity:.0f}) [-{penalty}]")
                        
                        # ── 2. BROKEN ──────────────────────────────────
                        if detected_class is None:
                            aspect_ratio = geom_features.get('aspect_ratio', 1.0)
                            extent = geom_features.get('extent', 0.8)
                            if aspect_ratio > 1.7 or aspect_ratio < 0.65 or extent < 0.68:
                                penalty = 75
                                current_score -= penalty
                                detected_class = "BROKEN"
                                logs.append(f"❌ **Fisik:** Broken/Pecah (AR: {aspect_ratio:.2f}) [-{penalty}]")
                        
                        # ── 3. PARTIAL SOUR ────────────────────────────
                        if detected_class is None and cluster_stats is not None:
                            valid_ratios = []
                            for idx, stat in cluster_stats.items():
                                r, g, b = stat['color']
                                if r > 185 and g > 185 and b > 185:
                                    continue
                                valid_ratios.append(r / g if g > 0 else 1.0)
                            if valid_ratios:
                                max_rg = max(valid_ratios)
                                if max_rg > 1.28:
                                    penalty = 50
                                    current_score -= penalty
                                    detected_class = "PARTIAL_SOUR"
                                    logs.append(f"⚠️ **Warna:** Partial Sour (R/G: {max_rg:.2f}) [-{penalty}]")
                        
                        # ── 4. WITHERED ────────────────────────────────
                        if detected_class is None:
                            intensity_w = geom_features.get('mean_intensity', 127)
                            circ_w = geom_features.get('circularity', 1.0)
                            sol_w = geom_features.get('solidity', 1.0)
                            ar_w = geom_features.get('aspect_ratio', 1.0)
                            
                            _mask_w = np.zeros(preprocess_results['rgb'].shape[:2], dtype=np.uint8)
                            if 'contour' in geom_features:
                                cv2.drawContours(_mask_w, [geom_features['contour']], -1, 255, -1)
                            _rgb_w = preprocess_results['rgb']
                            _g_w = float(np.mean(_rgb_w[:,:,1][_mask_w==255])) if np.any(_mask_w==255) else 130
                            _b_w = float(np.mean(_rgb_w[:,:,2][_mask_w==255])) if np.any(_mask_w==255) else 90
                            gb_ratio = _g_w / _b_w if _b_w > 0 else 1.5
                            
                            withered_signals = 0
                            withered_detail = []
                            
                            if gb_ratio < 1.40 and intensity_w > 140:
                                withered_signals += 3
                                withered_detail.append(f"G/B={gb_ratio:.3f} & I={intensity_w:.0f}")
                            elif gb_ratio < 1.38 and intensity_w > 132:
                                withered_signals += 2
                                withered_detail.append(f"G/B={gb_ratio:.3f} & I={intensity_w:.0f}(border)")
                            elif intensity_w > 148:
                                withered_signals += 2
                                withered_detail.append(f"I={intensity_w:.0f}>148")
                            
                            if circ_w < 0.80:
                                withered_signals += 1
                                withered_detail.append(f"Circ={circ_w:.3f}")
                            if sol_w < 0.985:
                                withered_signals += 1
                                withered_detail.append(f"Sol={sol_w:.3f}")
                            if ar_w >= 0.88:
                                withered_signals += 1
                                withered_detail.append(f"AR={ar_w:.2f}(port)")
                            
                            if withered_signals >= 3 and current_score > 70:
                                penalty = 30
                                current_score -= penalty
                                detected_class = "WITHERED"
                                logs.append(f"⚠️ **Fisik+Warna:** Withered ({', '.join(withered_detail)}) [-{penalty}]")
                        
                        # ── 5. INSECT HOLES ────────────────────────────
                        holes = geom_features.get('holes_count', 0)
                        if holes > 0:
                            if holes >= 2:
                                penalty = 90
                                logs.append(f"❌ **Hama:** Severe Insect Damage ({holes} lubang) [-{penalty}]")
                            else:
                                penalty = 85
                                logs.append(f"❌ **Hama:** Insect Damage (1 lubang terdeteksi) [-{penalty}]")
                            current_score -= penalty
                        
                        # ── Finalisasi ─────────────────────────────────
                        final_score = max(0, current_score)

                        # Tetapkan Konversi Grade Rule-Based
                        if final_score >= 88:
                            rule_grade_text, rule_grade_num = "NORMAL", 1
                        elif final_score >= 60:
                            rule_grade_text, rule_grade_num = "WITHERED / DEFECT RINGAN", 2
                        elif final_score >= 40:
                            rule_grade_text, rule_grade_num = "PARTIAL SOUR", 3
                        elif final_score >= 20:
                            rule_grade_text, rule_grade_num = "BROKEN / DRY CHERRY", 4
                        else:
                            rule_grade_text, rule_grade_num = "SEVERE INSECT DAMAGE", 5

                        class_names = ['Normal', 'Withered', 'Partial Sour', 'Broken', 'Dry Cherry', 'Severe Insect Damage']

                        # ==========================================
                        # 2. TAMPILAN UTAMA (Sesuai Pilihan Sidebar)
                        # ==========================================
                        if classification_mode == "Machine Learning" and ml_model is not None:
                            st.subheader(f"###Hasil Klasifikasi ML(Random Forest): **{pred_class}**")

                            col_score, col_grade = st.columns(2)
                            with col_score:
                                st.metric("Confidence", f"{confidence:.1f}%")
                            with col_grade:
                                st.metric("Grade", grade)
                            
                            # Tampilkan probabilitas per kelas
                            st.caption("📊 Distribusi Probabilitas per Kelas:")
                            for name, prob in zip(class_names, proba):
                                bar_length = int(prob * 30)
                                bar = "█" * bar_length + "░" * (30 - bar_length)
                                st.caption(f"   {name:<18}: {bar} {prob*100:.1f}%")

                            st.info(f"💡 Model ini dilatih dengan 378 sampel dan mencapai akurasi 92.1% pada data uji.")

                            # VISUALISASI DECISION BOUNDARY
                            with st.expander("📈 Visualisasi Decision Boundary Random Forest"):
                                
                                st.caption(
                                    "Visualisasi area keputusan model berdasarkan "
                                    "fitur Green Ratio dan Area."
                                )

                                fig = create_decision_boundary_plot(
                                    ml_model,
                                    ml_scaler,
                                    user_features=geom_features
                                )

                                st.pyplot(fig)
                                
                                # Menambahkan catatan kaki/edukasi untuk pengguna dashboard
                                st.info(
                                    "💡 **Catatan Grafik:**\n"
                                    "- **Titik-titik di atas adalah Data Testing (20% dari total dataset)** yang digunakan untuk menguji akurasi model pada data baru.\n"
                                    "- **Bintang Hitam (*)** merepresentasikan posisi karakteristik sampel biji kopi yang Anda masukkan saat ini.\n"
                                    "- Karena model ini bekerja pada ruang 10-Dimensi dan grafik ini hanya memproyeksikan 2-Dimensi utama, "
                                    "beberapa titik mungkin terlihat 'salah tempat' secara visual karena dipengaruhi oleh 8 fitur lainnya yang tidak terlihat di sini."
                                )
                        
                        else:
                            st.subheader(f"Total Skor Akhir: {final_score}")

                            text = f"### HASIL: GRADE {rule_grade_num} ({rule_grade_text})"
                            if rule_grade_num == 1:
                                st.success(text)
                            elif rule_grade_num == 2:
                                st.info(text)
                            elif rule_grade_num == 3:
                                st.warning(text)
                            else:
                                st.error(text)

                            for log in logs:
                                st.write(log)


                        # ==========================================
                        # 3. EXPANDER KOMPARASI (SIDE-BY-SIDE)
                        # ==========================================
                        with st.expander("🔍 Analisis: Bandingkan Rule-Based vs Machine Learning"):
                            # Layout Side-by-Side
                            c_rule, c_ml = st.columns(2)
                            
                            with c_rule:
                                st.markdown("#### 📐 Rule-Based")
                                st.metric("Grade", rule_grade_num)
                                st.write(f"**Kelas:** {rule_grade_text.title()}")
                                st.write(f"**Skor:** {final_score}/100")
                                st.markdown("**Jejak Penalti:**")
                                if len(logs) == 0:
                                    st.caption("- Tidak ada penalti")
                                else:
                                    for log in logs:
                                        st.caption(f"- {log}")
                                        
                            with c_ml:
                                st.markdown("#### 🤖 Machine Learning")
                                st.metric("Grade", grade)
                                st.write(f"**Kelas:** {pred_class}")
                                st.write(f"**Confidence:** {confidence:.1f}%")
                                st.markdown("**Top Probabilitas:**")
                                # Mengurutkan dan mengambil 3 probabilitas tertinggi agar tidak terlalu penuh
                                prob_dict = {name: p for name, p in zip(class_names, proba)}
                                sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)[:3])
                                for name, p in sorted_probs.items():
                                    st.caption(f"- {name}: {p*100:.1f}%")

                    else:
                        st.error("⚠️ Objek tidak terdeteksi dengan jelas.")

                if classification_mode == "Machine Learning":
                    with tabs[4]:
                        # st.subheader("📈 Analisis Model Dinamis")
                        # st.write("Visualisasi evaluasi model menggunakan 30 sampel dataset per kelas.")
                        
                        import pandas as pd
                        import numpy as np
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        import plotly.express as px
                        from sklearn.metrics import confusion_matrix
                        import os

                        class_names = ['Normal', 'Withered', 'Partial Sour', 'Broken', 'Dry Cherry', 'Severe Insect Damage']
                        features_list = ['area', 'circularity', 'solidity', 'extent', 'aspect_ratio', 'holes_count', 'center_cut_lines', 'mean_intensity', 'red_ratio', 'green_ratio']
                        dataset_path = "data/features_dataset.csv"

                        # ─── TAB 4: Penilaian Model ────────────────────────────────────────
                if classification_mode == "Machine Learning":
                    with tabs[4]:
                        st.subheader("📈 Analisis Model Dinamis")
                        st.write("Visualisasi evaluasi model menggunakan 30 sampel dataset per kelas.")
                        
                        import pandas as pd
                        import numpy as np
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        from sklearn.metrics import confusion_matrix
                        import os

                        class_names = ['Normal', 'Withered', 'Partial Sour', 'Broken', 'Dry Cherry', 'Severe Insect Damage']
                        features_list = ['area', 'circularity', 'solidity', 'extent', 'aspect_ratio', 'holes_count', 'center_cut_lines', 'mean_intensity', 'red_ratio', 'green_ratio']
                        dataset_path = "data/features_dataset.csv"

                        # Pastikan model, scaler, dan data asli tersedia
                        if ml_model is not None and ml_scaler is not None and os.path.exists(dataset_path):
                            
                            # ==========================================
                            # 1. TARIK TEPAT 30 DATA PER KELAS DARI CSV ANDA
                            # ==========================================
                            df_dataset = pd.read_csv(dataset_path)
                            
                            # Mengambil maksimal 30 baris untuk setiap kelas (class_num)
                            df_30_per_class = df_dataset.groupby('class_num').head(30)
                            
                            X_eval = df_30_per_class[features_list].values
                            y_eval = df_30_per_class['class_num'].values
                            
                            # Model memprediksi 180 data tersebut
                            X_eval_scaled = ml_scaler.transform(X_eval)
                            y_pred_eval = ml_model.predict(X_eval_scaled)
                            
                            # Generate Matriks NxN (6x6) dari hasil 30 data/kelas
                            cm = confusion_matrix(y_eval, y_pred_eval)

                            # ==========================================
                            # 2. GAMBAR HEATMAP 6x6 (Style Klasik Terang & Bergaris)
                            # ==========================================
                            st.markdown("#### 🧱 Confusion Matrix (30 Dataset/Kelas)")
                            
                            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                            sns.heatmap(
                                cm, 
                                annot=True, 
                                fmt='d', 
                                cmap='Blues', 
                                ax=ax_cm, 
                                cbar=True,
                                square=True,
                                linewidths=1,
                                linecolor='gray',
                                xticklabels=class_names,
                                yticklabels=class_names
                            )
                            
                            ax_cm.set_title('Confusion Matrix - Random Forest\n', fontsize=14)
                            ax_cm.set_xlabel('Predicted Label', fontsize=10)
                            ax_cm.set_ylabel('True Label', fontsize=10)
                            
                            ax_cm.tick_params(axis='x', rotation=45)
                            ax_cm.tick_params(axis='y', rotation=0)
                            
                            st.pyplot(fig_cm)

                            st.divider()

                            # ==========================================
                            # 3. FEATURE PROFILE (Style Matplotlib Klasik)
                            # ==========================================
                            st.markdown("#### 📊 Profil Fitur Biji Kopi Saat Ini")
                            
                            if 'geom_features' in locals():
                                # Menarik ke-10 fitur sesuai urutan dataset yang diminta
                                current_features = {
                                    'area': geom_features.get('area', 0),
                                    'circularity': geom_features.get('circularity', 0),
                                    'solidity': geom_features.get('solidity', 0),
                                    'extent': geom_features.get('extent', 0),
                                    'aspect_ratio': geom_features.get('aspect_ratio', 0),
                                    'holes_count': geom_features.get('holes_count', 0),
                                    'center_cut_lines': geom_features.get('center_cut_lines', 0),
                                    'mean_intensity': geom_features.get('mean_intensity', 0),
                                    'red_ratio': geom_features.get('red_ratio', 0),
                                    'green_ratio': geom_features.get('green_ratio', 0)
                                }
                                
                                df_feat = pd.DataFrame({
                                    'Fitur': list(current_features.keys()), 
                                    'Nilai': list(current_features.values())
                                }).sort_values(by='Nilai', ascending=True)
                                
                                # Membuat grafik bar horizontal menggunakan Matplotlib
                                fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
                                
                                # Warna biru default Matplotlib (#1f77b4) persis seperti contoh gambar
                                ax_fi.barh(df_feat['Fitur'], df_feat['Nilai'], color='#1f77b4', align='center')
                                
                                ax_fi.set_xlabel('Nilai Kuantitatif (Skala Logaritmik)')
                                ax_fi.set_title('Profil Fitur Biji Kopi Saat Ini')
                                
                                # Mengatur batas/bingkai grafik agar berwarna hitam solid
                                for spine in ax_fi.spines.values():
                                    spine.set_color('black')
                                    spine.set_linewidth(1)
                                
                                # Menggunakan skala logaritma agar fitur dengan nilai besar tidak menutupi yang kecil
                                ax_fi.set_xscale('log')
                                
                                plt.tight_layout()
                                st.pyplot(fig_fi)
                                
                        else:
                            st.error("⚠️ Data `features_dataset.csv` atau Model `.pkl` tidak ditemukan untuk melakukan perhitungan.")
                        
            except Exception as e:
                st.error(f"Terjadi kesalahan teknis: {e}")

st.divider()