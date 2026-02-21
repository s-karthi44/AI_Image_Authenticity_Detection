import streamlit as st
import sys
import os
# Fix python path for Streamlit Cloud to find our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

try:
    from ai_model.predict import load_model, predict_single
    from app.api.services.report_gen import generate_pdf_report
except ModuleNotFoundError:
    # Fallback if working directory is slightly different
    sys.path.append(os.getcwd())
    from ai_model.predict import load_model, predict_single
    from app.api.services.report_gen import generate_pdf_report

# Set page config
st.set_page_config(page_title="AI Image Authenticity Detection", page_icon="🔍", layout="wide")

def main():
    st.title("🔍 AI Image Authenticity Detection - Phase 3 Enterprise")
    st.markdown("""
    This enhanced application uses deep learning (ResNet50) and Phase 2 advanced forensic analysis 
    signals to detect if an image is AI-generated (Fake) or captured by a camera (Real).
    """)

    # Sidebar info
    st.sidebar.title("About")
    st.sidebar.info("Upload one or multiple images to analyze authenticity. The system checks for deep neural network artifacts and forensic inconsistencies across 7 detectors.")
    
    # Initialize session state for mock database / analytics
    if "db" not in st.session_state:
        st.session_state.db = []
        
    tab1, tab2 = st.tabs(["🔮 Analyzer", "📊 Admin Analytics"])

    with tab1:
        # File upload
        uploaded_files = st.file_uploader("Upload images for analysis...", type=["jpg", "png", "jpeg", "webp"], accept_multiple_files=True)
        
        if uploaded_files:
            model_path = 'best_model.pth'
            if not os.path.exists(model_path):
                st.error("Model file 'best_model.pth' not found! Please train the model first.")
                st.info("Run `python -m ai_model.train` to train the model on provided dataset.")
                return
                
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(model_path, device=device)
            
            from detectors.prnu_detector import calculate_prnu_score
            from detectors.frequency_analysis import calculate_artifact_score
            from detectors.pixel_noise import compute_naturalness_score
            from detectors.exif_analysis import compute_metadata_score
            from detectors.facial_consistency import compute_facial_score
            from detectors.shadow_geometry import compute_shadow_score
            from detectors.specular_reflection import compute_reflection_score
            from fusion.decision_engine import DecisionEngine

            for uploaded_file in uploaded_files:
                st.markdown("---")
                st.subheader(f"Analysis: {uploaded_file.name}")
                col1, col2 = st.columns([1, 1])
                
                image = Image.open(uploaded_file)
                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                
                with col2:
                    # Save temporary file for prediction
                    temp_path = f"temp_{uploaded_file.name}"
                    image.convert("RGB").save(temp_path)
                    
                    with st.spinner(f"Analyzing {uploaded_file.name}..."):
                        try:
                            ai_score = predict_single(model, temp_path, device=device)
                            ai_model_aligned = 1.0 - ai_score if ai_score is not None else None
                            
                            prnu = calculate_prnu_score(temp_path)
                            freq_artifact = calculate_artifact_score(temp_path)
                            freq_aligned = 1.0 - freq_artifact if freq_artifact is not None else None
                            pixel = compute_naturalness_score(temp_path)
                            metadata = compute_metadata_score(temp_path)
                            facial = compute_facial_score(temp_path)
                            shadow = compute_shadow_score(temp_path)
                            reflection = compute_reflection_score(temp_path)
                            
                            scores_dict = {
                                'prnu': prnu,
                                'frequency': freq_aligned,
                                'pixel_noise': pixel,
                                'facial': facial,
                                'shadow': shadow,
                                'reflection': reflection,
                                'metadata': metadata,
                                'ai_model': ai_model_aligned
                            }
                            
                            engine = DecisionEngine()
                            results = engine.analyze(scores_dict)
                            results['image_id'] = uploaded_file.name
                            
                            # Add to analytics DB
                            st.session_state.db.append(results)
                            
                            # Results visualization
                            st.metric(label="Verdict", value=results['verdict'])
                            st.metric(label="Overall Confidence", value=f"{results['confidence']}%")
                            
                            if results['verdict'] == "AI_GENERATED":
                                st.error(f"Verdict: {results['verdict']}")
                            elif results['verdict'] == "REAL":
                                st.success(f"Verdict: {results['verdict']}")
                            else:
                                st.warning(f"Verdict: {results['verdict']}")
                                
                            # Report DL config
                            pdf_bytes = generate_pdf_report(results)
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_bytes,
                                file_name=f"report_{uploaded_file.name}.pdf",
                                mime="application/pdf",
                                key=f"dl_{uploaded_file.name}"
                            )
                                
                            st.write("### Forensic Module Scores (0=AI, 1=REAL)")
                            colA, colB, colC = st.columns(3)
                            with colA:
                                st.metric(label="PRNU Sensor Noise", value=f"{prnu:.2f}" if prnu is not None else "N/A")
                                st.metric(label="Pixel Naturalness", value=f"{pixel:.2f}" if pixel is not None else "N/A")
                                st.metric(label="Metadata (EXIF)", value=f"{metadata:.2f}" if metadata is not None else "N/A")
                            with colB:
                                st.metric(label="Frequency Alignment", value=f"{freq_aligned:.2f}" if freq_aligned is not None else "N/A")
                                st.metric(label="AI Model Consistency", value=f"{ai_model_aligned:.2f}" if ai_model_aligned is not None else "N/A")
                                st.metric(label="Shadow Geometry", value=f"{shadow:.2f}" if shadow is not None else "N/A")
                            with colC:
                                st.metric(label="Facial Consistency", value=f"{facial:.2f}" if facial is not None else "N/A")
                                st.metric(label="Specular Reflection", value=f"{reflection:.2f}" if reflection is not None else "N/A")

                            st.write("### Detailed Reasoning")
                            for reason in results['reasoning']:
                                st.write(f"✓ {reason}")
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    with tab2:
        st.header("Admin Analytics Dashboard")
        st.write("Enterprise Phase 3 reporting and usage metrics.")
        
        db = st.session_state.db
        total = len(db)
        
        if total == 0:
            st.info("No images processed yet. Upload images in the Analyzer tab to populate statistics.")
        else:
            col1, col2, col3 = st.columns(3)
            real_count = sum(1 for item in db if item.get('verdict') == 'REAL')
            fake_count = sum(1 for item in db if item.get('verdict') == 'AI_GENERATED')
            unc_count = sum(1 for item in db if item.get('verdict') == 'UNCERTAIN')
            
            with col1:
                st.metric("Total Scans", total)
            with col2:
                st.metric("AI Detected", f"{int(fake_count/total*100)}%")
            with col3:
                avg_conf = sum(item.get('confidence', 0) for item in db) / total
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                
            st.divider()
            
            st.subheader("Verdict Distribution")
            
            # Use simple bar chart via Streamlit
            st.bar_chart({
                "Verdicts": {"REAL": real_count, "AI_GENERATED": fake_count, "UNCERTAIN": unc_count}
            })
            
            st.subheader("Recent Scan Logs")
            for item in reversed(db[-5:]):
                st.text(f"File: {item.get('image_id')} | Verdict: {item.get('verdict')} | Score: {item.get('final_score'):.2f}")

if __name__ == "__main__":
    main()
