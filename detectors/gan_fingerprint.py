import numpy as np
import torch
import torch.nn as nn
from scipy import fftpack
import cv2

class GANFingerprintDetector:
    """
    Detects specific artifacts from GAN/diffusion models:
    - Checkerboard patterns (upsampling)
    - Spectral irregularities
    - Color bleeding artifacts
    - Boundary inconsistencies
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def detect_checkerboard_artifacts(self, image: np.ndarray) -> float:
        """
        GANs using deconvolution/upsampling create checkerboard patterns.
        Most visible in high frequencies.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
        else:
            gray = image.astype(float)
        
        # Apply 2D FFT
        fft = fftpack.fft2(gray)
        fft_shift = fftpack.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Look for checkerboard pattern in frequency domain
        # Checkerboard creates peaks at Nyquist frequency
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Sample high-frequency corners (where checkerboard appears)
        corner_regions = [
            magnitude[0:h//4, 0:w//4],           # Top-left
            magnitude[0:h//4, 3*w//4:],          # Top-right
            magnitude[3*h//4:, 0:w//4],          # Bottom-left
            magnitude[3*h//4:, 3*w//4:]          # Bottom-right
        ]
        
        # Calculate energy in corner regions
        corner_energy = sum(np.sum(region**2) for region in corner_regions)
        total_energy = np.sum(magnitude**2)
        
        # High corner energy = checkerboard artifact
        checkerboard_score = corner_energy / (total_energy + 1e-10)
        
        # Normalize to 0-1
        return min(1.0, float(checkerboard_score * 10))
    
    def detect_color_bleeding(self, image: np.ndarray) -> float:
        """
        AI models sometimes create color bleeding across boundaries.
        """
        if len(image.shape) != 3:
            return 0.0
        
        # Edge detection on each channel
        edges_r = cv2.Canny(image[:,:,0], 50, 150)
        edges_g = cv2.Canny(image[:,:,1], 50, 150)
        edges_b = cv2.Canny(image[:,:,2], 50, 150)
        
        # In real photos, edges align across channels
        # In AI images, channels may have misaligned edges (color bleeding)
        
        # Calculate edge agreement
        total_edges = edges_r | edges_g | edges_b
        aligned_edges = edges_r & edges_g & edges_b
        
        if np.sum(total_edges) == 0:
            return 0.0
        
        alignment = np.sum(aligned_edges) / np.sum(total_edges)
        
        # Low alignment = color bleeding artifact
        bleeding_score = 1.0 - alignment
        
        return float(bleeding_score)
    
    def detect_upsampling_artifacts(self, image: np.ndarray) -> float:
        """
        Many AI models upsample latent representations.
        This creates periodic patterns.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Compute autocorrelation
        # Real images: low autocorrelation (unique content)
        # AI upsampled: higher autocorrelation (repeated patterns)
        
        normalized = (gray - np.mean(gray)) / (np.std(gray) + 1e-10)
        
        # Compute autocorrelation using FFT (efficient)
        fft = fftpack.fft2(normalized)
        power_spectrum = np.abs(fft) ** 2
        autocorr = fftpack.ifft2(power_spectrum).real
        autocorr = fftpack.fftshift(autocorr)
        
        # Normalize
        autocorr = autocorr / autocorr.max()
        
        h, w = autocorr.shape
        center_h, center_w = h // 2, w // 2
        
        # Look for peaks away from center (periodic patterns)
        # Exclude center (DC component)
        mask = np.ones_like(autocorr)
        mask[center_h-10:center_h+10, center_w-10:center_w+10] = 0
        
        # Find peaks
        peaks = autocorr * mask
        peak_score = np.max(peaks)
        
        return float(peak_score)
    
    def analyze(self, image: np.ndarray) -> dict:
        """
        Complete GAN fingerprint analysis.
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        checkerboard = self.detect_checkerboard_artifacts(image)
        color_bleeding = self.detect_color_bleeding(image)
        upsampling = self.detect_upsampling_artifacts(image)
        
        # Combine scores
        gan_score = 0.4 * checkerboard + 0.3 * color_bleeding + 0.3 * upsampling
        
        return {
            'gan_fingerprint_score': float(gan_score),
            'checkerboard_score': float(checkerboard),
            'color_bleeding_score': float(color_bleeding),
            'upsampling_score': float(upsampling),
            'has_gan_artifacts': gan_score > 0.6
        }
