import cv2
import numpy as np

class CompressionHistoryAnalyzer:
    """
    Real photos: multiple compressions (camera → storage → upload)
    AI images: single or no compression history
    """
    
    def analyze_jpeg_artifacts(self, image_path: str) -> dict:
        """
        Analyze JPEG compression history.
        """
        try:
            # Load image as raw bytes
            with open(image_path, 'rb') as f:
                jpeg_data = f.read()
            
            # Extract quantization tables from JPEG
            quant_tables = self._extract_quantization_tables(jpeg_data)
            
            # Real photos: typically 2-3 compression cycles
            # AI images: 0-1 compression cycles
            compression_cycles = len(quant_tables)
            
            # Calculate compression inconsistency
            # Look for block boundaries (8x8 DCT blocks)
            img = cv2.imread(image_path)
            if img is not None:
                inconsistency = self._detect_block_inconsistencies(img)
            else:
                inconsistency = 0.0
            
            return {
                'compression_cycles': compression_cycles,
                'has_multiple_compressions': compression_cycles >= 2,
                'block_inconsistency': float(inconsistency),
                'compression_score': float(min(1.0, compression_cycles / 3.0))
            }
        except Exception as e:
            return {
                'compression_cycles': 0,
                'has_multiple_compressions': False,
                'block_inconsistency': 0.0,
                'compression_score': 0.0
            }

    def _extract_quantization_tables(self, jpeg_data: bytes) -> list:
        """
        Extract DQT segments from JPEG bytes.
        """
        tables = []
        i = 0
        while i < len(jpeg_data) - 1:
            # Look for 0xFF marker
            if jpeg_data[i] == 0xFF:
                marker = jpeg_data[i+1]
                # DQT marker is 0xDB
                if marker == 0xDB:
                    # found DQT
                    tables.append(True)
                    # Skip to next marker by reading length (not doing full parsing for simplicity, 
                    # just counting DQT segments as a proxy. Actually DQT segments might contain multiple tables.
                    # This is a heuristic)
                    length = (jpeg_data[i+2] << 8) + jpeg_data[i+3]
                    i += length + 2
                    continue
                elif marker == 0xDA: # SOS marker (Start of Scan) - no more markers after this typically except EOI
                    break
                else:
                    # just standard marker processing, skip 2 bytes for marker
                    # handles APPn, SOFn, etc. if length is present
                    # FF D8 (SOI), FF D9 (EOI), FF 00 (escaped FF) don't have lengths
                    if marker not in [0xD8, 0xD9, 0x00, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7]:
                        if i+3 < len(jpeg_data):
                            length = (jpeg_data[i+2] << 8) + jpeg_data[i+3]
                            i += length + 2
                            continue
            i += 1
        return tables

    def _detect_block_inconsistencies(self, image: np.ndarray) -> float:
        """
        JPEG compresses in 8x8 blocks.
        Look for visible block boundaries.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Calculate gradient
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Look for peaks at 8-pixel intervals (block boundaries)
        h, w = gray.shape
        if h < 16 or w < 16:
            return 0.0
            
        # Sample horizontal and vertical profiles
        h_profile = np.sum(np.abs(gradient_x), axis=0)
        v_profile = np.sum(np.abs(gradient_y), axis=1)
        
        # Check for periodicity at 8-pixel intervals
        h_fft = np.fft.fft(h_profile)
        v_fft = np.fft.fft(v_profile)
        
        # Look for peak at 8-pixel frequency
        block_freq_h = w // 8
        block_freq_v = h // 8
        
        h_peak = np.abs(h_fft[block_freq_h]) if block_freq_h < len(h_fft) else 0
        v_peak = np.abs(v_fft[block_freq_v]) if block_freq_v < len(v_fft) else 0
        
        total_energy = np.sum(np.abs(h_fft)) + np.sum(np.abs(v_fft)) + 1e-10
        block_score = (h_peak + v_peak) / total_energy
        
        return float(block_score * 100)
    
    def analyze(self, image: np.ndarray, image_path: str = None) -> dict:
        if image_path:
            return self.analyze_jpeg_artifacts(image_path)
            
        # Fallback if only image is provided
        inconsistency = self._detect_block_inconsistencies(image)
        return {
            'compression_cycles': 0,
            'has_multiple_compressions': False,
            'block_inconsistency': float(inconsistency),
            'compression_score': 0.0
        }
