import cv2
import numpy as np

def detect_image_filters(image_path):
    """
    Analyzes an image to detect if it's grayscale, highly saturated (filters), 
    or has a distinct color cast like sepia.
    Returns a dictionary of boolean flags.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"is_grayscale": False, "is_high_saturation": False, "is_sepia": False}
            
        # Check if grayscale
        b, g, r = cv2.split(image)
        
        # Calculate mean absolute difference between channels
        diff_rg = np.mean(np.abs(r.astype(int) - g.astype(int)))
        diff_gb = np.mean(np.abs(g.astype(int) - b.astype(int)))
        diff_rb = np.mean(np.abs(r.astype(int) - b.astype(int)))
        
        avg_diff = (diff_rg + diff_gb + diff_rb) / 3.0
        
        is_grayscale = avg_diff < 3.0 # Threshold for being effectively black & white
        
        # Check high saturation (often a sign of Instagram-like filters)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        is_high_saturation = mean_saturation > 180 # 0-255 scale in OpenCV HSV
        
        is_sepia = False
        if not is_grayscale:
            # Simple sepia check: R > G > B consistently by a wide margin
            r_mean = np.mean(r)
            g_mean = np.mean(g)
            b_mean = np.mean(b)
            if r_mean > g_mean + 15 and g_mean > b_mean + 15:
                is_sepia = True
                
        return {
            "is_grayscale": is_grayscale,
            "is_high_saturation": is_high_saturation,
            "is_sepia": is_sepia
        }
    except Exception:
        return {"is_grayscale": False, "is_high_saturation": False, "is_sepia": False}
