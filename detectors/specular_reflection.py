import numpy as np
import cv2
from preprocessing.face_detection import extract_landmarks

def detect_eye_catchlights(image, landmarks):
    """
    Extracts the brightest points (catchlights) in both eyes.
    Returns relative x,y positions of the catchlight within each eye.
    """
    if landmarks is None:
        return None
        
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    elif image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    h, w = image.shape
    
    # Left eye outline: 33, 133, 160, 158, 153, 144
    left_eye_pts = landmarks[[33, 133, 160, 158, 153, 144], :2] * [w, h]
    # Right eye outline: 362, 263, 387, 385, 380, 373
    right_eye_pts = landmarks[[362, 263, 387, 385, 380, 373], :2] * [w, h]
    
    def get_catchlight(eye_pts):
        x_min, y_min = np.min(eye_pts, axis=0).astype(int)
        x_max, y_max = np.max(eye_pts, axis=0).astype(int)
        
        # Pad bounds safely
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return None
            
        eye_roi = image[y_min:y_max, x_min:x_max]
        _, max_val, _, max_loc = cv2.minMaxLoc(eye_roi)
        
        # Relative position (0 to 1) within the bounding box
        rx = max_loc[0] / (x_max - x_min + 1e-6)
        ry = max_loc[1] / (y_max - y_min + 1e-6)
        return np.array([rx, ry])

    left_catch = get_catchlight(left_eye_pts)
    right_catch = get_catchlight(right_eye_pts)
    
    return [left_catch, right_catch]

def compare_bilateral_reflections(left, right):
    if left is None or right is None:
        return 1.0 # High error/mismatch basically
    diff = np.linalg.norm(left - right)
    return diff

def compute_reflection_score(image):
    """
    Returns float [0-1].
    In real photos, eye catchlights reflect the same light sources, so their 
    relative positions in each eye match closely. AI often renders 
    mismatched catchlights.
    """
    try:
        landmarks = extract_landmarks(image)
        if landmarks is None:
            return None
            
        catchlights = detect_eye_catchlights(image, landmarks)
        if not catchlights:
            return None
            
        diff = compare_bilateral_reflections(catchlights[0], catchlights[1])
        
        # If diff is small (< 0.1), highly consistent -> Real
        # If diff is large (> 0.2), highly inconsistent -> AI
        if diff < 0.1:
            return 0.8 # Consistent
        elif diff > 0.25:
            return 0.2 # Inconsistent
        else:
            return None # Uncertain
            
    except Exception:
        return None
