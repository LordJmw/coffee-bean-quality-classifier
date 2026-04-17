# utils/preprocessing.py
import cv2
import numpy as np

def load_and_convert(image_array):
    """
    Konversi input ke format RGB yang sesuai
    """
    if len(image_array.shape) == 2:
        # Grayscale → RGB
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        # RGBA → RGB
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    else:
        # Sudah RGB (dari PIL) atau BGR (dari OpenCV)
        img_rgb = image_array.copy()
    
    return img_rgb

def downsample_image(image_array, target_size=(224, 224)):
    """
    M1: Downsampling/Resize ke ukuran seragam 224x224
    - Untuk 256x256 (Normal): downscale sedikit → tetap tajam
    - Untuk 500x500 (Cacat): downscale ~2.2x → efisien
    """
    h, w = image_array.shape[:2]
    
    # Resize ke target_size untuk konsistensi
    img_resized = cv2.resize(image_array, target_size, interpolation=cv2.INTER_AREA)
    
    return img_resized

def rgb_to_grayscale(image_array):
    """
    M1: Quantization RGB 24-bit → Grayscale 8-bit
    """
    if len(image_array.shape) == 3:
        img_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image_array.copy()
    
    return img_gray

def apply_gaussian_blur(gray_image, kernel_size=(5, 5)):
    """
    M3: Gaussian Blur untuk noise reduction
    """
    # Kernel size harus ganjil
    if kernel_size[0] % 2 == 0:
        kernel_size = (kernel_size[0] + 1, kernel_size[1] + 1)
    
    img_blur = cv2.GaussianBlur(gray_image, kernel_size, 0)
    return img_blur

def apply_threshold(gray_image, method='otsu'):
    """
    Konversi Grayscale → Binary (untuk morfologi M4)
    """
    if method == 'otsu':
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    return binary

def preprocess_pipeline(image_array, target_size=(224, 224), blur_kernel=(5, 5), threshold_method='otsu'):
    """
    Menjalankan seluruh preprocessing (M1, M3, persiapan M4)
    
    Parameters:
    - image_array: Input gambar (numpy array)
    - target_size: Ukuran output resize (default: 224x224)
    - blur_kernel: Kernel size untuk Gaussian Blur
    - threshold_method: Metode thresholding ('otsu' atau 'binary')
    
    Returns:
    - Dictionary berisi hasil setiap tahapan
    """
    # M1: Load & Convert
    img_rgb = load_and_convert(image_array)
    
    # Simpan dimensi asli untuk info
    original_size = img_rgb.shape[:2]
    
    # M1: Downsampling/Resize
    img_down = downsample_image(img_rgb, target_size)
    
    # M1: RGB → Grayscale
    img_gray = rgb_to_grayscale(img_down)
    
    # M3: Gaussian Blur
    img_blur = apply_gaussian_blur(img_gray, blur_kernel)
    
    # Persiapan M4: Threshold → Binary
    img_binary = apply_threshold(img_blur, threshold_method)
    
    results = {
        'rgb': img_down,
        'gray': img_gray,
        'blur': img_blur,
        'binary': img_binary,
        'original_size': original_size,
        'processed_size': target_size
    }
    
    return results