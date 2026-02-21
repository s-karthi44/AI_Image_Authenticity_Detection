import numpy as np
import cv2
from preprocessing.face_detection import extract_landmarks

def detect_shadows(image, landmarks):
    # Abstract representation of lighting direction based on face brightness
    if landmarks is None:
        return []
        
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    h, w = image.shape
    
    # Left cheek approx indices: [116, 117, 118, 100]
    # Right cheek approx indices: [345, 346, 347, 329]
    left_cheek_pts = landmarks[[116, 117, 118, 100], :2] * [w, h]
    right_cheek_pts = landmarks[[345, 346, 347, 329], :2] * [w, h]
    
    # Get average brightness
    left_val = 0
    for pt in left_cheek_pts:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < h and 0 <= x < w:
            left_val += image[y, x]
            
    right_val = 0
    for pt in right_cheek_pts:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= y < h and 0 <= x < w:
            right_val += image[y, x]
            
    # Return gradient vector (simple 1D representation here)
    return [(left_val / 4.0), (right_val / 4.0)]

def calculate_shadow_vectors(shadows):
    if not shadows or sum(shadows) == 0:
        return np.array([0, 0])
    # Normalize
    L, R = shadows
    total = L + R + 1e-6
    return np.array([L/total, R/total])

def verify_consistency(vectors):
    # If the image is extremely flat lit (AI often produces very flat perfect lighting)
    # or extremely high contrast, return a score predicting authenticity.
    diff = abs(vectors[0] - vectors[1])
    return diff

def compute_shadow_score(image):
    """
    Returns float [0-1].
    AI faces often lack natural 3D shadowing and appear 'painted' with flat lighting.
    Score: 1.0 (natural), 0.0 (flat/artificial)
    """
    try:
        landmarks = extract_landmarks(image)
        if landmarks is None:
            return None
            
        shadows = detect_shadows(image, landmarks)
        vectors = calculate_shadow_vectors(shadows)
        diff = verify_consistency(vectors)
        
        # Real images usually have some directional light, difference > 0.05
        # AI images are sometimes perfectly lit, difference < 0.02
        if diff < 0.02:
            return 0.2  # Unnaturally perfect lighting
        elif diff > 0.3:
            return 0.9  # Strong directional light, likely real
        else:
            return 0.6 + (diff * 2) # Range 0.6 - 0.9 depending on gradient
            
    except Exception:
        return None
