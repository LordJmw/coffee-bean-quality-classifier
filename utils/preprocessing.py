import cv2
import numpy as np

def load_and_convert(image_array):
    """M1: Konversi input ke format RGB yang sesuai"""
    if len(image_array.shape) == 2:
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = image_array.copy()
    return img_rgb

def downsample_image(image_array, target_size=(224, 224), interpolation_method="Area-based"):
    """M2: Resize gambar untuk konsistensi koordinat fitur"""
    interp_dict = {
        "Nearest Neighbor": cv2.INTER_NEAREST,
        "Bilinear interpolation": cv2.INTER_LINEAR,
        "Bicubic interpolation": cv2.INTER_CUBIC,
        "Area-based": cv2.INTER_AREA,
        "Lanczos": cv2.INTER_LANCZOS4,
    }
    cv2_interp = interp_dict.get(interpolation_method, cv2.INTER_AREA)
    img_resized = cv2.resize(image_array, target_size, interpolation=cv2_interp)
    return img_resized

def rgb_to_grayscale(image_array):
    """M3: Konversi RGB ke Grayscale"""
    if len(image_array.shape) == 3:
        img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image_array.copy()
    return img_gray

def apply_gaussian_blur(gray_image, kernel_size=(5, 5)):
    """M3: Reduksi Noise dengan Gaussian Blur"""
    if kernel_size[0] % 2 == 0:
        kernel_size = (kernel_size[0] + 1, kernel_size[1] + 1)
    img_blur = cv2.GaussianBlur(gray_image, kernel_size, 0)
    return img_blur

def apply_threshold(gray_image, method='otsu'):
    """M3: Thresholding untuk Segmentasi Biji"""
    if method == 'otsu':
        # THRESH_BINARY_INV digunakan jika background lebih terang dari biji
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary

def preprocess_pipeline(image_array, target_size=(224, 224), blur_kernel=(5, 5), 
                        threshold_method='otsu', interpolation_method="Area-based"):
    """
    Menjalankan preprocessing dengan satu jalur ukuran (Single-Scale Pipeline)
    agar koordinat kontur sinkron dengan koordinat warna.
    """
    # 1. Load Original
    img_original = load_and_convert(image_array)
    
    # 2. Downsample (PENTING: Semua proses fitur pakai ukuran ini)
    img_rgb = downsample_image(img_original, target_size, interpolation_method)
    
    # 3. Grayscale & Blur
    img_gray = rgb_to_grayscale(img_rgb)
    img_blur = apply_gaussian_blur(img_gray, blur_kernel)
    
    # 4. Binary
    img_binary = apply_threshold(img_blur, threshold_method)
    
    results = {
        'rgb': img_rgb,           # Digunakan untuk tampilan UI & K-Means
        'gray': img_gray,         # Digunakan untuk mean_intensity
        'blur': img_blur,         # Digunakan untuk Canny/Hough Lines
        'binary': img_binary,     # Digunakan untuk Kontur & Morphologi
        'original_rgb': img_original, # Hanya untuk preview kualitas tinggi
        'target_size': target_size
    }
    
    return results