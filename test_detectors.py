import numpy as np
import cv2

from detectors.multi_scale_noise import MultiScaleNoiseAnalyzer
from detectors.gan_fingerprint import GANFingerprintDetector

dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

print("Testing MultiScaleNoiseAnalyzer...")
analyzer = MultiScaleNoiseAnalyzer()
result = analyzer.analyze(dummy_image)
print(result)

print("Testing GANFingerprintDetector...")
analyzer = GANFingerprintDetector()
result = analyzer.analyze(dummy_image)
print(result)

print("All tests passed.")
