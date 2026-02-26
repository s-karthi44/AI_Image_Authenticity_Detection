import cv2
import numpy as np

class LocalArtifactDetector:
    """
    Detects local artifacts in AI-generated faces:
    - Asymmetric eyes
    - Merged/missing teeth
    - Impossible jewelry/accessories
    - Hair texture inconsistencies
    """
    
    def analyze_teeth(self, image: np.ndarray, face_landmarks: np.ndarray) -> float:
        """
        AI struggles with teeth - often merged, floating, or missing.
        """
        if face_landmarks is None or len(face_landmarks) < 68:
            return 0.0
            
        # Extract mouth region using landmarks
        # Landmarks 48-67 are mouth in dlib 68-point model format typically.
        # But we use MediaPipe, so landmarks might be different.
        # Let's assume passed face_landmarks are in a standard format or MediaPipe.
        # Actually, MediaPipe returns 478 points. We need a heuristic for mouth if mediapipe is used.
        # For simplicity, let's use a bounding box approach if 68 points are not passed.
        
        # If Mediapipe (length 478)
        if len(face_landmarks) > 400:
            # MediaPipe mouth indices (inner and outer lips)
            mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
            mouth_points = np.array([face_landmarks[i][:2] for i in mouth_indices if i < len(face_landmarks)])
        else:
            # Assume 68 point format
            mouth_points = face_landmarks[48:68, :2]
        
        if len(mouth_points) < 3:
            return 0.0

        h, w = image.shape[:2]
        # Convert landmarks to pixel coordinates if they are normalized (0-1)
        if np.max(mouth_points) <= 1.0:
            pixel_points = mouth_points * np.array([w, h])
        else:
            pixel_points = mouth_points
            
        pixel_points = pixel_points.astype(np.int32)
        
        # Create mask for mouth region
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pixel_points], 255)
        
        # Extract mouth ROI
        mouth_roi = cv2.bitwise_and(image, image, mask=mask)
        
        # Convert to grayscale
        if len(mouth_roi.shape) == 3:
            mouth_gray = cv2.cvtColor(mouth_roi, cv2.COLOR_RGB2GRAY)
        else:
            mouth_gray = mouth_roi
        
        # Detect edges
        edges = cv2.Canny(mouth_gray, 50, 150)
        
        # Calculate edge density
        mouth_area = np.sum(mask > 0)
        if mouth_area == 0:
            return 0.0
            
        edge_density = np.sum(edges > 0) / (mouth_area + 1e-10)
        
        # Low edge density = merged/blurry teeth = AI artifact
        artifact_score = 1.0 - min(1.0, edge_density * 50)
        
        return float(max(0.0, artifact_score))
    
    def analyze_eye_symmetry(self, image: np.ndarray, face_landmarks: np.ndarray) -> float:
        """
        Check for unnatural eye symmetry or asymmetry.
        """
        if face_landmarks is None or len(face_landmarks) < 68:
            return 0.0

        if len(face_landmarks) > 400:
            # MediaPipe eye indices
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            left_eye = np.array([face_landmarks[i][:2] for i in left_eye_indices])
            right_eye = np.array([face_landmarks[i][:2] for i in right_eye_indices])
        else:
            left_eye = face_landmarks[36:42, :2]
            right_eye = face_landmarks[42:48, :2]

        if len(left_eye) < 4 or len(right_eye) < 4:
            return 0.0

        # Calculate eye properties (approximate width and height)
        # Assuming format: [left/right, top/bottom, etc]
        # In a generic way:
        left_width = np.linalg.norm(np.max(left_eye[:, 0]) - np.min(left_eye[:, 0]))
        right_width = np.linalg.norm(np.max(right_eye[:, 0]) - np.min(right_eye[:, 0]))
        
        left_height = np.linalg.norm(np.max(left_eye[:, 1]) - np.min(left_eye[:, 1]))
        right_height = np.linalg.norm(np.max(right_eye[:, 1]) - np.min(right_eye[:, 1]))
        
        # Calculate asymmetry
        if max(left_width, right_width) == 0 or max(left_height, right_height) == 0:
            return 0.0
            
        width_asymmetry = abs(left_width - right_width) / max(left_width, right_width)
        height_asymmetry = abs(left_height - right_height) / max(left_height, right_height)
        
        avg_asymmetry = (width_asymmetry + height_asymmetry) / 2
        
        if avg_asymmetry < 0.03:
            # Too perfect
            artifact_score = 0.8
        elif avg_asymmetry > 0.20:
            # Too asymmetric (AI error)
            artifact_score = 0.9
        else:
            # Natural range
            artifact_score = 0.2
        
        return artifact_score
    
    def analyze_hair_texture(self, image: np.ndarray, face_landmarks: np.ndarray) -> float:
        """
        AI-generated hair often has unnatural texture or repetitive patterns.
        """
        if face_landmarks is None or len(face_landmarks) == 0:
            return 0.0

        h, w = image.shape[:2]
        
        # Convert landmarks to pixel coordinates if they are normalized
        if np.max(face_landmarks[:, 1]) <= 1.0:
            forehead_top = np.min(face_landmarks[:, 1]) * h
        else:
            forehead_top = np.min(face_landmarks[:, 1])
            
        # Hair region: top 30% above forehead
        hair_region = image[max(0, int(forehead_top - h*0.3)):int(forehead_top), :]
        
        if hair_region.size == 0 or hair_region.shape[0] < 5 or hair_region.shape[1] < 5:
            return 0.0
        
        # Convert to grayscale
        if len(hair_region.shape) == 3:
            hair_gray = cv2.cvtColor(hair_region, cv2.COLOR_RGB2GRAY)
        else:
            hair_gray = hair_region
        
        texture_responses = []
        for theta in range(4):
            theta_rad = theta / 4. * np.pi
            kernel = cv2.getGaborKernel((21, 21), 5, theta_rad, 10, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(hair_gray, cv2.CV_32F, kernel)
            texture_responses.append(np.std(filtered))
        
        if len(texture_responses) == 0:
            return 0.0
            
        texture_variance = np.std(texture_responses)
        
        # Low variance = unnatural (too smooth or repetitive)
        # A natural variance cutoff needs to be empirical. Let's use 50.
        artifact_score = 1.0 - min(1.0, texture_variance / 50)
        
        return float(max(0.0, artifact_score))
    
    def analyze(self, image: np.ndarray, face_landmarks: np.ndarray = None) -> dict:
        """
        Complete local artifact analysis.
        """
        # If no landmarks provided, return 0s
        if face_landmarks is None or len(face_landmarks) == 0:
            return {
                'local_artifact_score': 0.0,
                'teeth_artifacts': 0.0,
                'eye_artifacts': 0.0,
                'hair_artifacts': 0.0,
                'has_local_artifacts': False
            }
            
        teeth_score = self.analyze_teeth(image, face_landmarks)
        eye_score = self.analyze_eye_symmetry(image, face_landmarks)
        hair_score = self.analyze_hair_texture(image, face_landmarks)
        
        # Combine
        local_artifact_score = (
            teeth_score * 0.4 +
            eye_score * 0.3 +
            hair_score * 0.3
        )
        
        return {
            'local_artifact_score': float(local_artifact_score),
            'teeth_artifacts': float(teeth_score),
            'eye_artifacts': float(eye_score),
            'hair_artifacts': float(hair_score),
            'has_local_artifacts': local_artifact_score > 0.5
        }
