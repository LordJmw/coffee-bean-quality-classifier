 
# utils/morphology.py
import cv2
import numpy as np

def morphological_opening(binary_image, kernel_size=(3, 3), iterations=1):
    """
    M4: Opening (Erosi → Dilasi)
    Menghilangkan noise kecil (debu, rambut halus)
    """
    kernel = np.ones(kernel_size, np.uint8)
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opening

def morphological_closing(binary_image, kernel_size=(3, 3), iterations=2):
    """
    M4: Closing (Dilasi → Erosi)
    Menyambung tepi biji yang putus-putus
    """
    kernel = np.ones(kernel_size, np.uint8)
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closing

def apply_morphology(binary_image, open_kernel=(3, 3), close_kernel=(5, 5)):
    """
    Menjalankan Opening → Closing
    """
    # Opening: hilangkan noise
    img_opening = morphological_opening(binary_image, open_kernel, iterations=1)
    
    # Closing: sambung tepi
    img_closing = morphological_closing(img_opening, close_kernel, iterations=2)
    
    return {
        'opening': img_opening,
        'closing': img_closing
    }