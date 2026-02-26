import cv2
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from fusion.decision_engine import DecisionEngine

def test_decision_engine():
    engine = DecisionEngine()
    
    # Test with mock scores
    scores_dict = {
        'prnu': 0.8,
        'frequency': 0.9,
        'pixel_noise': 0.8,
        'multi_scale_noise': 0.85,
        'gan_fingerprint': 0.9,
        'facial': 0.8,
        'shadow': None,
        'reflection': 0.9,
        'metadata': 0.7,
        'ai_model': 0.9,
        'deep_features': 0.85,
        'local_artifacts': 0.8,
        'compression': 0.8
    }
    
    result = engine.analyze(scores_dict)
    print("Decision Engine Result (Real/High):", result)
    
    # Test Fake
    scores_dict_fake = {
        'prnu': 0.2,
        'frequency': 0.1,
        'pixel_noise': 0.1,
        'multi_scale_noise': 0.2,
        'gan_fingerprint': 0.1,
        'facial': 0.2,
        'shadow': 0.1,
        'reflection': None,
        'metadata': 0.1,
        'ai_model': 0.1,
        'deep_features': 0.15,
        'local_artifacts': 0.2,
        'compression': 0.1
    }
    result_fake = engine.analyze(scores_dict_fake)
    print("Decision Engine Result (Fake/Low):", result_fake)

if __name__ == "__main__":
    test_decision_engine()
