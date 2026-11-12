"""
ReportLab PDF Document Rendering Engine.
GastricPath AI — Phase 9: Professional PDF Generation Engine.

Assembles Platypus flowables and renders publication-ready A4 PDF files.
"""

from pathlib import Path
from typing import Any
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate

from research.clinical_reporting.pdf.canvas_numbered import NumberedCanvas


def render_pdf_document(flowables: list[Any], output_path: Path, document_title: str = "GastricPath AI Medical Report") -> Path:
    """Renders a sequence of ReportLab flowables to an A4 PDF document with NumberedCanvas."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # A4: 595.27 x 841.89 points
    # Standard margins: 36 points (0.5 inch) left/right, 42 points top/bottom
    doc = SimpleDocTemplate(
        str(output_path.resolve()),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=48,
        title=document_title,
        author="GastricPath AI Computational Pathology System"
    )

    doc.build(flowables, canvasmaker=NumberedCanvas)
    return output_path

