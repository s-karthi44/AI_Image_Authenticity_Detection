import numpy as np
import cv2
import os

def load_image(path):
    """
    Load image safely, preserving noise structure.
    Does NOT aggressively normalize.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")
        
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image from {path}. May be corrupted or unsupported format.")
        
    # Standardize to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def is_grayscale(image):
    """
    Check if an image is grayscale despite having 3 channels.
    """
    if len(image.shape) == 2:
        return True
    
    # If 3 channels, check if R=G=B
    b, g, r = cv2.split(image)
    if np.array_equal(b, g) and np.array_equal(g, r):
        return True
    return False

def validate_format(path):
    """
    Check if format is supported and image is readable.
    """
    valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}
    ext = os.path.splitext(path)[1].lower()
    if ext not in valid_exts:
        return False
        
    return os.path.exists(path)

def normalize_resolution(image, target=1024):
    """
    Resize while preserving frequency and noise structure as much as possible.
    Avoid extremely smooth interpolations that destroy PRNU.
    """
    h, w = image.shape[:2]
    
    # If already smaller, don't upscale (prevents creating upsampling artifacts)
    if max(h, w) <= target:
        return image
        
    # Calculate scale preserving aspect ratio
    scale = target / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # INTER_AREA is preferred for downsampling without creating artifacts
    # and it preserves some noise statistics compared to INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def convert_color_space(image, target='RGB'):
    """
    Gracefully handle grayscale images or other conversions.
    """
    if target == 'GRAY':
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
        
    if target == 'RGB':
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image
        
    return image
