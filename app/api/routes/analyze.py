from fastapi import APIRouter, File, UploadFile, HTTPException
import tempfile
import os
import torch
import shutil

from ai_model.predict import load_model, predict_single
from detectors.prnu_detector import calculate_prnu_score
from detectors.frequency_analysis import calculate_artifact_score
from detectors.pixel_noise import compute_naturalness_score
from detectors.multi_scale_noise import MultiScaleNoiseAnalyzer
from detectors.gan_fingerprint import GANFingerprintDetector
from fusion.decision_engine import DecisionEngine

router = APIRouter()

# Global model instance for efficiency
model_cache = None

from fastapi.responses import Response

# Phase 3 In-memory persistent state (simulated DB)
analysis_db = {}

def _process_image_file(image_file, filename):
    global model_cache
    if not filename.endswith('.jpg') and not filename.endswith('.jpeg') and not filename.endswith('.png') and not filename.endswith('.webp'):
        raise HTTPException(status_code=400, detail="INVALID_IMAGE_FORMAT")
        
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            shutil.copyfileobj(image_file, temp_file)
            temp_path = temp_file.name

        # Load AI Model context
        model_path = 'best_model.pth'
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail="AI model checkpoint 'best_model.pth' not found.")
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_cache is None:
            model_cache = load_model(model_path, device=device)
            
        ai_score = predict_single(model_cache, temp_path, device=device)
        ai_model_aligned = 1.0 - ai_score if ai_score is not None else None
        
        from detectors.exif_analysis import compute_metadata_score
        from detectors.facial_consistency import compute_facial_score
        from detectors.shadow_geometry import compute_shadow_score
        from detectors.specular_reflection import compute_reflection_score
        
        from detectors.deep_feature_inconsistency import DeepFeatureAnalyzer
        from detectors.local_artifact_detector import LocalArtifactDetector
        from detectors.compression_history import CompressionHistoryAnalyzer
        from preprocessing.face_detection import extract_landmarks

        # Detector Scores
        prnu = calculate_prnu_score(temp_path)
        freq_artifact = calculate_artifact_score(temp_path)
        freq_aligned = 1.0 - freq_artifact if freq_artifact is not None else None
        pixel = compute_naturalness_score(temp_path)
        
        # Load image once for array-based detectors
        import cv2
        import numpy as np
        img_cv = cv2.imread(temp_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB) if img_cv is not None else np.zeros((224,224,3), dtype=np.uint8)

        # Phase 1 Major Improvement Plan Detectors
        multi_scale_noise_analyzer = MultiScaleNoiseAnalyzer()
        ms_noise_result = multi_scale_noise_analyzer.analyze(temp_path)
        ms_noise_score = ms_noise_result['natural_noise_score']
        
        gan_fingerprint_detector = GANFingerprintDetector()
        gan_result = gan_fingerprint_detector.analyze(temp_path)
        gan_score = gan_result['gan_fingerprint_score']
        gan_aligned = 1.0 - gan_score if gan_score is not None else None
        
        # Phase 2 metrics
        metadata = compute_metadata_score(temp_path)
        facial = compute_facial_score(temp_path)
        shadow = compute_shadow_score(temp_path)
        reflection = compute_reflection_score(temp_path)

        # New Phase 3 metrics
        deep_analyzer = DeepFeatureAnalyzer()
        deep_result = deep_analyzer.analyze(img_rgb)
        deep_aligned = 1.0 - deep_result['inconsistency_score']

        local_detector = LocalArtifactDetector()
        try:
            landmarks = extract_landmarks(img_rgb)
            local_result = local_detector.analyze(img_rgb, landmarks)
            local_aligned = 1.0 - local_result['local_artifact_score']
        except Exception:
            local_aligned = None

        comp_analyzer = CompressionHistoryAnalyzer()
        comp_result = comp_analyzer.analyze_jpeg_artifacts(temp_path)
        comp_aligned = comp_result['compression_score'] # higher compression = real
        
        scores_dict = {
            'prnu': prnu,
            'frequency': freq_aligned,
            'pixel_noise': pixel,
            'multi_scale_noise': ms_noise_score,
            'gan_fingerprint': gan_aligned,
            'facial': facial,
            'shadow': shadow,
            'reflection': reflection,
            'metadata': metadata,
            'ai_model': ai_model_aligned,
            'deep_features': deep_aligned,
            'local_artifacts': local_aligned,
            'compression': comp_aligned
        }
        
        # Fuse and Classify
        engine = DecisionEngine()
        results = engine.analyze(scores_dict)
        
        # Clean up
        os.remove(temp_path)
        
        out = {
            "image_id": filename,
            "verdict": results['verdict'],
            "confidence": results['confidence'],
            "final_score": results['final_score'],
            "scores": results['scores'],
            "reasoning": results['reasoning']
        }
        
        analysis_db[filename] = out
        return out
        
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    return _process_image_file(image.file, image.filename)

@router.post("/batch")
async def analyze_batch(images: list[UploadFile] = File(...)):
    results = []
    for image in images:
        results.append(_process_image_file(image.file, image.filename))
        
    counts = {"AI_GENERATED": 0, "REAL": 0, "UNCERTAIN": 0}
    for r in results:
        counts[r["verdict"]] += 1
        
    return {
        "batch_id": f"batch_{os.urandom(4).hex()}",
        "total_images": len(images),
        "results": results,
        "summary": {
            "ai_generated": counts["AI_GENERATED"],
            "real": counts["REAL"],
            "uncertain": counts["UNCERTAIN"]
        }
    }

@router.get("/status/{image_id}")
async def get_analysis_status(image_id: str):
    if image_id in analysis_db:
        return {
            "image_id": image_id,
            "status": "COMPLETED",
            "progress": 100,
            "estimated_time_remaining": 0
        }
    else:
        # Mocking NOT_FOUND vs Pending state since jobs are synchronous right now
        raise HTTPException(status_code=404, detail="Analysis not found or pending.")

@router.get("/report/{image_id}")
async def get_pdf_report(image_id: str):
    if image_id not in analysis_db:
        raise HTTPException(status_code=404, detail="Analysis not found.")
        
    from app.api.services.report_gen import generate_pdf_report
    pdf_bytes = generate_pdf_report(analysis_db[image_id])
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report_{image_id}.pdf"}
    )
