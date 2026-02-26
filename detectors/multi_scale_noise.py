import numpy as np
import pywt
from scipy import signal, stats
import cv2

class MultiScaleNoiseAnalyzer:
    """
    Advanced noise analysis across multiple scales.
    Works on grayscale, handles filters and effects.
    """
    
    def __init__(self):
        self.scales = [1, 2, 4, 8]  # Multi-scale pyramid
        self.wavelet = 'db4'  # Daubechies 4 wavelet
        
    def analyze(self, image: np.ndarray) -> dict:
        """
        Analyze noise characteristics at multiple scales.
        
        Returns:
            {
                'noise_consistency_score': float,  # 0-1
                'natural_noise_score': float,
                'scale_coherence': float,
                'is_natural': bool
            }
        """
        # Load string paths if needed
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        scores = []
        
        # Analyze at each scale
        for scale in self.scales:
            # Downsample to scale
            h, w = gray.shape
            scaled = cv2.resize(gray, (w // scale, h // scale))
            
            # Extract noise using wavelet decomposition
            coeffs = pywt.dwt2(scaled, self.wavelet)
            cA, (cH, cV, cD) = coeffs
            
            # Combine detail coefficients (high-frequency = noise)
            noise = np.sqrt(cH**2 + cV**2 + cD**2)
            
            # Calculate noise statistics
            entropy = self._calculate_entropy(noise)
            kurtosis = stats.kurtosis(noise.flatten())
            skewness = stats.skew(noise.flatten())
            
            # Natural noise characteristics:
            # - High entropy (random)
            # - Near-Gaussian kurtosis (~3.0)
            # - Low skewness (symmetric)
            
            natural_score = self._score_noise_naturalness(
                entropy, kurtosis, skewness
            )
            scores.append(natural_score)
        
        # Check consistency across scales
        # Real photos: noise consistent across scales
        # AI images: noise varies or missing at certain scales
        consistency = 1.0 - np.std(scores)
        
        return {
            'noise_consistency_score': float(consistency),
            'natural_noise_score': float(np.mean(scores)),
            'scale_coherence': float(1.0 - np.std(scores)),
            'is_natural': consistency > 0.7 and np.mean(scores) > 0.6
        }
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy."""
        hist, _ = np.histogram(data.flatten(), bins=256, density=True)
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))
    
    def _score_noise_naturalness(self, entropy: float, 
                                  kurtosis: float, 
                                  skewness: float) -> float:
        """
        Score how natural the noise appears.
        
        Natural camera noise:
        - Entropy: 6.0 - 8.0
        - Kurtosis: 2.5 - 3.5 (near-Gaussian)
        - Skewness: -0.5 - 0.5 (symmetric)
        """
        # Entropy score
        entropy_score = 1.0 if 6.0 <= entropy <= 8.0 else max(0, 1.0 - abs(entropy - 7.0) / 3.0)
        
        # Kurtosis score (3.0 is perfect Gaussian)
        kurtosis_score = max(0, 1.0 - abs(kurtosis - 3.0) / 2.0)
        
        # Skewness score (0 is perfect symmetry)
        skewness_score = max(0, 1.0 - abs(skewness) / 1.0)
        
        # Weighted combination
        return 0.4 * entropy_score + 0.3 * kurtosis_score + 0.3 * skewness_score
