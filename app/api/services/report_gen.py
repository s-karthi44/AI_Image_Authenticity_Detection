from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import datetime

def generate_pdf_report(data):
    """
    Generate PDF report from a response dictionary.
    Returns bytes.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "AI Image Authenticity Analysis Report")
    
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 85, f"Image: {data.get('image_id', 'Unknown')}")
    
    # Verdict
    c.setFont("Helvetica-Bold", 14)
    verdict_text = f"Verdict: {data.get('verdict')} ({data.get('confidence', 0)}% Confidence)"
    c.drawString(50, height - 120, verdict_text)
    
    # Scores
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 150, "Forensic Scores (0=AI, 1=REAL)")
    c.setFont("Helvetica", 10)
    
    y = height - 170
    for key, value in data.get('scores', {}).items():
        if isinstance(value, dict):
            val_str = str(value)
        else:
            val_str = f"{value:.2f}" if value is not None else "N/A"
        c.drawString(70, y, f"{key}: {val_str}")
        y -= 15
        
    # Reasoning
    y -= 15
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Detailed Reasoning")
    c.setFont("Helvetica", 10)
    y -= 20
    
    for reason in data.get('reasoning', []):
        c.drawString(70, y, f"- {reason}")
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
            
    c.save()
    buffer.seek(0)
    return buffer.getvalue()
