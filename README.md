# AI Image Authenticity Detection

This project provides a comprehensive system for detecting AI-generated images using forensic analysis, deep learning, and multi-modal fusion techniques.

## Overview
The system combines traditional image forensics (PRNU, noise analysis, frequency analysis) with a state-of-the-art CNN model to classify images as real or AI-generated.

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download dataset:
   ```bash
   python scripts/download_dataset.py
   ```
3. Run the application:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Structure
- `preprocessing/`: Image loading and normalization
- `detectors/`: Hand-crafted forensic feature extractors
- `ai_model/`: Deep learning classifier
- `fusion/`: Decision logic combining all signals
- `app/`: User interface and API
