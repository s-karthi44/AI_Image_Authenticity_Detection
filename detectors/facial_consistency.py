import numpy as np
from preprocessing.face_detection import extract_landmarks

def calculate_symmetry_metrics(landmarks):
    if landmarks is None or len(landmarks) < 468:
        return None
        
    # Using mediapipe indices for eyes (centers)
    left_eye = np.mean(landmarks[[33, 133, 160, 158, 153, 144]], axis=0)
    right_eye = np.mean(landmarks[[362, 263, 387, 385, 380, 373]], axis=0)
    nose_tip = landmarks[4]
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]
    
    # Calculate geometric symmetry
    # Distance from nose tip to left eye vs right eye
    d_left_eye = np.linalg.norm(left_eye[:2] - nose_tip[:2])
    d_right_eye = np.linalg.norm(right_eye[:2] - nose_tip[:2])
    eye_sym = abs(d_left_eye - d_right_eye) / (max(d_left_eye, d_right_eye) + 1e-6)
    
    # Distance from nose tip to mouth corners
    d_left_mouth = np.linalg.norm(mouth_left[:2] - nose_tip[:2])
    d_right_mouth = np.linalg.norm(mouth_right[:2] - nose_tip[:2])
    mouth_sym = abs(d_left_mouth - d_right_mouth) / (max(d_left_mouth, d_right_mouth) + 1e-6)
    
    return {
        "eye_symmetry_diff": eye_sym,
        "mouth_symmetry_diff": mouth_sym
    }

def check_anatomical_correctness(landmarks):
    metrics = calculate_symmetry_metrics(landmarks)
    if not metrics:
        return False
    # If asymmetry is too extreme, it's anatomically incorrect
    if metrics["eye_symmetry_diff"] > 0.25 or metrics["mouth_symmetry_diff"] > 0.3:
        return False
    return True

def compute_facial_score(image):
    """
    Returns float [0-1].
    Real faces have natural asymmetry (around 0.05-0.15 diff).
    AI faces might be perfectly symmetric (<0.02) or wildly asymmetric (>0.25).
    Score mapping:
    - Normal (Real): 0.6 - 1.0 (diff 0.05 to 0.15)
    - Too perfect (AI): 0.2 - 0.5 (diff 0.0 to 0.04)
    - Extreme anomaly (AI): 0.0 - 0.2 (diff > 0.2)
    """
    try:
        landmarks = extract_landmarks(image)
        if landmarks is None:
            return None # No face found
            
        metrics = calculate_symmetry_metrics(landmarks)
        avg_diff = (metrics["eye_symmetry_diff"] + metrics["mouth_symmetry_diff"]) / 2.0
        
        if avg_diff < 0.03:
            # Suspiciously perfect symmetry
            return 0.3 + (avg_diff / 0.03) * 0.2
        elif avg_diff > 0.2:
            # Anatomical failure
            return max(0.0, 0.4 - ((avg_diff - 0.2) * 2.0))
        else:
            # Natural variance
            return min(1.0, 0.6 + ((avg_diff - 0.03) / 0.17) * 0.4)
    except Exception:
        return None
