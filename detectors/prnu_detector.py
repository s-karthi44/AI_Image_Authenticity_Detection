import numpy as np
import cv2

def extract_noise_residual(image):
    if isinstance(image, str): # if path is given
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    # High-pass filter: original - gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    residual = cv2.subtract(gray, blur)
    return residual

def calculate_prnu_score(image):
    """
    Returns float [0-1]. 
    0.0 = no sensor noise (AI), 1.0 = strong sensor noise (Real camera)
    """
    try:
        residual = extract_noise_residual(image)
        std_dev = np.std(residual)
        # Map std_dev: smooth AI usually < 2, noisy Real usually > 5
        score = np.clip((std_dev - 1.0) / 8.0, 0.0, 1.0)
        return float(score)
    except Exception:
        return None
