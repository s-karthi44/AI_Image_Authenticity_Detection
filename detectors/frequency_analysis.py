import numpy as np
import cv2

def compute_fft(image):
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude_spectrum

def analyze_spectrum(fft_result):
    # Returns dict of frequency stats
    center_y, center_x = fft_result.shape[0] // 2, fft_result.shape[1] // 2
    center_energy = np.mean(fft_result[center_y-20:center_y+20, center_x-20:center_x+20])
    edge_energy = np.mean(fft_result[0:20, 0:20])
    return {"center": center_energy, "edge": edge_energy}

def detect_periodic_artifacts(spectrum):
    # Basic check: high frequency energy relative to center means artifacts
    stats = analyze_spectrum(spectrum)
    ratio = stats["edge"] / (stats["center"] + 1e-6)
    return ratio > 0.5

def calculate_artifact_score(image):
    """
    Returns float [0-1].
    0.0 = clean (real images have smooth decay), 1.0 = artifacts (AI images often have high frequency grids/peaks)
    """
    try:
        mag = compute_fft(image)
        stats = analyze_spectrum(mag)
        ratio = stats["edge"] / (stats["center"] + 1e-6)
        
        # Scale ratio to 0-1
        # Normal ratio is low (e.g. 0.1-0.3), AI might be 0.4-0.8+
        score = np.clip((ratio - 0.2) * 2.5, 0.0, 1.0)
        return float(score)
    except Exception:
        return None
