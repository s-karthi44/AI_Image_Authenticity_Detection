import numpy as np
import cv2

def extract_noise(image):
    if isinstance(image, str):
        image = cv2.imread(image)
        
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Convert to float to avoid uint8 overflow/clamping which destroys half the noise data
    gray_f = gray.astype(np.float32)
    
    # Extract noise by taking absolute difference between image and median filtered version
    median_f = cv2.medianBlur(gray, 3).astype(np.float32)
    noise = cv2.absdiff(gray_f, median_f)
    return noise

def calculate_entropy(noise):
    # Calculate Shannon Entropy of the noise
    # Use max bin of 64 since noise is usually small
    hist, _ = np.histogram(noise.ravel(), bins=64, range=(0, 64))
    hist = hist / (hist.sum() + 1e-10)
    
    # Only keep non-zero probabilities
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)

def compute_naturalness_score(image):
    """
    Returns float [0-1].
    0.0 = artificial (smooth AI noise or heavily compressed), 1.0 = natural (real camera noise)
    """
    try:
        noise = extract_noise(image)
        entropy = calculate_entropy(noise)
        
        # Real camera noise usually has a chaotic entropy profile.
        # AI images are either too clean (low entropy) or have synthetic structured noise.
        # Natural sensor noise typically yields entropy > 2.0 depending on ISO
        score = np.clip((entropy - 1.0) / 2.5, 0.0, 1.0)
        return float(score)
    except Exception:
        return None
