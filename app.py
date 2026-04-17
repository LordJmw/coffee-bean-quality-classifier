import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image

# Import modul yang sudah dibuat
from utils.dataset_loader import get_dataset_samples, get_dataset_statistics, load_image_from_dataset, DATASET_INFO
from utils.preprocessing import preprocess_pipeline
from utils.morphology import apply_morphology

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
            # Pilih kelas
            selected_class = st.selectbox(
                "Pilih kelas cacat:",
                list(samples.keys())
            )
            
            if selected_class:
                class_info = samples[selected_class]
                
                # Tampilkan info kelas
                grade_emoji = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "⚫"}
                st.info(f"{grade_emoji.get(class_info['grade'], '')} Grade {class_info['grade']}: {class_info['description']}")
                st.caption(f"Total: {class_info['total_images']} gambar | Label: {class_info['label']}")
                
                # Pilih gambar dalam kelas
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
                # Baca gambar
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

        # Tampilkan info dimensi
        original_h, original_w = img.shape[:2]
        st.write(f"**Dimensi Asli:** {original_w} x {original_h} piksel")
        st.write(f"**Dimensi Preprocessing:** {target_size[0]} x {target_size[1]} piksel")

        if len(img.shape) == 3:
            st.write(f"**Channel:** {img.shape[2]} (RGB)")

        # Tampilkan info resize
        if original_w != target_size[0] or original_h != target_size[1]:
            if original_w > target_size[0]:
                st.caption(f"⬇️ Downscale: {original_w}x{original_h} → {target_size[0]}x{target_size[1]}")
            else:
                st.caption(f"⬆️ Upscale: {original_w}x{original_h} → {target_size[0]}x{target_size[1]}")

        # Tampilkan expected grade jika dari dataset
        if input_mode == "📦 Gunakan Dataset Sample" and 'expected_grade' in st.session_state:
            grade_emoji = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "⚫"}
            st.write(f"**Expected Grade:** {grade_emoji.get(st.session_state['expected_grade'], '')} {st.session_state['expected_grade']}")
            if 'expected_class' in st.session_state:
                st.caption(f"Kelas: {st.session_state['expected_class']}")

        st.divider()
        st.subheader("🔬 Hasil Preprocessing (M1-M4)")

        # Jalankan preprocessing
        with st.spinner("Memproses gambar..."):
            try:
                # M1 & M3: Preprocessing
                preprocess_results = preprocess_pipeline(
                    img, 
                    target_size=target_size,
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
                        use_container_width=True, 
                        clamp=True
                    )
                    st.caption("M1: Sampling & Quantization")
                    
                with tab2:
                    st.image(
                        preprocess_results['blur'], 
                        caption=f"Gaussian Blur (kernel={blur_kernel}x{blur_kernel})", 
                        use_container_width=True, 
                        clamp=True
                    )
                    st.caption("M3: Linear Filter - Noise Reduction")
                    
                with tab3:
                    st.image(
                        morph_results['opening'], 
                        caption=f"Opening (kernel={open_kernel}x{open_kernel})", 
                        use_container_width=True, 
                        clamp=True
                    )
                    st.caption("M4: Erosi → Dilasi - Menghilangkan noise kecil")
                    
                with tab4:
                    st.image(
                        morph_results['closing'], 
                        caption=f"Closing (kernel={close_kernel}x{close_kernel})", 
                        use_container_width=True, 
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