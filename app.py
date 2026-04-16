import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image

# Import modul yang sudah dibuat
# Pastikan file utils/preprocessing.py dan utils/morphology.py tersedia
try:
    from utils.preprocessing import preprocess_pipeline
    from utils.morphology import apply_morphology
except ImportErrors:
    st.error("Gagal mengimpor modul utils. Pastikan folder 'utils' tersedia.")

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Klasifikasi Biji Kopi",
    page_icon="☕",
    layout="wide"
)

# ============================================
# FUNGSI HELPER
# ============================================

def get_dataset_samples(base_path="data/raw"):
    """
    Mendapatkan daftar sampel dataset yang tersedia
    """
    dataset_info = {
        'Normales.jpg': {'label': 'Good', 'grade': 1, 'description': 'Biji normal'},
        'PMordidoCortado.jpg': {'label': 'Broken', 'grade': 4, 'description': 'Pecah/terpotong'},
        'BrocadoLeve.jpg': {'label': 'Slight Insect', 'grade': 4, 'description': 'Lubang ringan'},
        'BrocadoSevero.jpg': {'label': 'Severe Insect', 'grade': 5, 'description': 'Lubang parah'},
        'DXHongo.jpg': {'label': 'Fungus', 'grade': 5, 'description': 'Jamur'},
        'Negros.jpg': {'label': 'Full Black', 'grade': 5, 'description': 'Hitam total'},
        'MarronAVinagre.jpg': {'label': 'Sour', 'grade': 5, 'description': 'Asam/fermentasi'},
        'CerezaSeca.jpg': {'label': 'Dried Cherry', 'grade': 5, 'description': 'Kulit kering'},
        'Pergamino.jpg': {'label': 'Parchment', 'grade': 5, 'description': 'Kulit tanduk'},
        'Concha.jpg': {'label': 'Shell', 'grade': 3, 'description': 'Cangkang kosong'},
        'Inmaduro.jpg': {'label': 'Immature', 'grade': 3, 'description': 'Biji muda'}
    }
    
    samples = {}
    
    if os.path.exists(base_path):
        for filename, info in dataset_info.items():
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                samples[info['description']] = {
                    'path': filepath,
                    'grade': info['grade'],
                    'label': info['label']
                }
    else:
        os.makedirs(base_path, exist_ok=True)
    
    return samples

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
    
    blur_kernel = st.slider("Gaussian Blur Kernel", 3, 11, 5, step=2)
    open_kernel = st.slider("Opening Kernel", 2, 7, 3)
    close_kernel = st.slider("Closing Kernel", 3, 9, 5)
    
    st.divider()
    
    # Jika pilih dataset sample
    if input_mode == "📦 Gunakan Dataset Sample":
        st.subheader("📂 Pilih Sampel Dataset")
        
        samples = get_dataset_samples("data/raw")
        
        if samples:
            selected_desc = st.selectbox(
                "Pilih jenis sampel:",
                list(samples.keys())
            )
            
            if selected_desc:
                sample_info = samples[selected_desc]
                st.session_state['sample_image_path'] = sample_info['path']
                st.session_state['expected_grade'] = sample_info['grade']
                
                st.info(f"📊 Expected Grade: {sample_info['grade']}")
                st.caption(f"Label: {sample_info['label']}")
        else:
            st.warning("⚠️ Dataset belum di-download!")
            st.markdown("""
            **Struktur folder yang diharapkan:**
            `data/raw/Normales.jpg`, dsb.
            """)

# ============================================
# MAIN CONTENT HEADER
# ============================================

st.title("☕ Klasifikasi Mutu Biji Kopi Mentah")
st.markdown("### Ekstraksi Fitur Morfologi & Geometri dengan Preprocessing Citra Digital")

st.divider()
st.caption("M1-M7: Preprocessing → Morfologi → Fitur → Clustering")

# ============================================
# MAIN CONTENT COLUMNS
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
            st.image(image, caption="Gambar Input", use_column_width=True)
            st.success("✅ Gambar berhasil diupload!")
            st.session_state['input_image'] = np.array(image)
            st.session_state['input_mode'] = 'upload'
        else:
            st.warning("⚠️ Silakan upload gambar terlebih dahulu")

    elif input_mode == "📦 Gunakan Dataset Sample":
        if 'sample_image_path' in st.session_state:
            image_path = st.session_state['sample_image_path']

            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                st.image(image_rgb, caption=f"Sampel: {os.path.basename(image_path)}", use_column_width=True)
                st.success(f"✅ Gambar dari dataset berhasil dimuat!")
                
                st.session_state['input_image'] = image_rgb
                st.session_state['input_mode'] = 'sample'
            else:
                st.error(f"File tidak ditemukan: {image_path}")
        else:
            st.info("👈 Silakan pilih gambar dari sidebar")

with col2:
    if 'input_image' in st.session_state:
        st.subheader("📊 Informasi Gambar")
        img = st.session_state['input_image']

        # Tampilkan info dimensi
        if len(img.shape) == 3:
            st.write(f"**Dimensi:** {img.shape[1]} x {img.shape[0]} piksel")
            st.write(f"**Channel:** {img.shape[2]} (RGB)")
        else:
            st.write(f"**Dimensi:** {img.shape[1]} x {img.shape[0]} piksel")
            st.write(f"**Channel:** 1 (Grayscale)")

        st.write(f"**Mode Input:** {st.session_state.get('input_mode', 'N/A')}")

        st.divider()
        st.subheader("🔬 Hasil Preprocessing (M1-M4)")

        # Jalankan preprocessing
        with st.spinner("Memproses gambar..."):
            try:
                # M1 & M3: Preprocessing
                preprocess_results = preprocess_pipeline(
                    img, 
                    blur_kernel=(blur_kernel, blur_kernel)
                )
                
                # M4: Morfologi
                morph_results = apply_morphology(
                    preprocess_results['binary'],
                    open_kernel=(open_kernel, open_kernel),
                    close_kernel=(close_kernel, close_kernel)
                )
                
                # Simpan ke session state untuk digunakan nanti
                st.session_state['preprocess_results'] = preprocess_results
                st.session_state['morph_results'] = morph_results
                
                # Tampilkan hasil dalam tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "M1: Grayscale", 
                    "M3: Gaussian Blur", 
                    "M4: Opening", 
                    "M4: Closing"
                ])
                
                with tab1:
                    st.image(
                        preprocess_results['gray'], 
                        caption="Grayscale (8-bit) - Hasil Konversi RGB", 
                        use_column_width=True, 
                        clamp=True
                    )
                    st.caption("M1: Sampling & Quantization")
                    
                with tab2:
                    st.image(
                        preprocess_results['blur'], 
                        caption=f"Gaussian Blur (kernel={blur_kernel}x{blur_kernel})", 
                        use_column_width=True, 
                        clamp=True
                    )
                    st.caption("M3: Linear Filter - Noise Reduction")
                    
                with tab3:
                    st.image(
                        morph_results['opening'], 
                        caption=f"Opening (kernel={open_kernel}x{open_kernel})", 
                        use_column_width=True, 
                        clamp=True
                    )
                    st.caption("M4: Erosi → Dilasi - Menghilangkan noise kecil")
                    
                with tab4:
                    st.image(
                        morph_results['closing'], 
                        caption=f"Closing (kernel={close_kernel}x{close_kernel})", 
                        use_column_width=True, 
                        clamp=True
                    )
                    st.caption("M4: Dilasi → Erosi - Menyambung tepi putus")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat preprocessing: {str(e)}")
                st.info("Pastikan gambar valid dan modul preprocessing sudah benar.")
    else:
        st.info("👈 Gambar akan ditampilkan di sini setelah diupload/dipilih")

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("🎓 Tugas Computer Vision - Implementasi M1-M7 untuk Klasifikasi Biji Kopi")