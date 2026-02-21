import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

# Download the model if it doesn't exist
model_asset_path = 'face_landmarker.task'
if not os.path.exists(model_asset_path):
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    urllib.request.urlretrieve(url, model_asset_path)

# Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path=model_asset_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

def _load_image(image):
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def extract_landmarks(image):
    # Returns array of landmarks
    image = _load_image(image)
    
    # Convert OpenCV image to MediaPipe image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    
    # Detect face landmarks
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks:
        return None
    
    landmarks = detection_result.face_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

def detect_face(image):
    # Returns bbox [x_min, y_min, width, height]
    image = _load_image(image)
    landmarks_arr = extract_landmarks(image)
    
    if landmarks_arr is None:
        return None
        
    h, w, _ = image.shape
    x_coords = [lm[0] * w for lm in landmarks_arr]
    y_coords = [lm[1] * h for lm in landmarks_arr]
    
    return [min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]

def crop_to_face(image, padding=0.2):
    image = _load_image(image)
    bbox = detect_face(image)
    if not bbox:
        return image
        
    h, w, _ = image.shape
    x, y, bw, bh = bbox
    
    x1 = max(0, int(x - bw * padding))
    y1 = max(0, int(y - bh * padding))
    x2 = min(w, int(x + bw + bw * padding))
    y2 = min(h, int(y + bh + bh * padding))
    
    return image[y1:y2, x1:x2]

def validate_face_present(image):
    return extract_landmarks(image) is not None
