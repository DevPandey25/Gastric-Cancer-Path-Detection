"""
Histopathology Image Atlas Report Generator (Mode B).
GastricPath AI — Phase 8: Research Manuscript and Image Atlas Composer.

Generates an image-centered PDF showcasing genuine H&E histopathology patches,
model predictions, probability distributions, and attribution heatmaps.
"""

from pathlib import Path
from typing import Any, Optional
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, Spacer, Table, Image as RLImage, PageBreak, KeepTogether

from research.clinical_reporting.pdf.styles import (
    get_reporting_styles, get_clinical_table_style, get_alert_box_style
)
from research.clinical_reporting.pdf.renderer import render_pdf_document
from research.clinical_reporting.figures.image_panels import generate_multi_panel_figure
from app.core.config import settings

BASE_DIR = Path(__file__).resolve().parents[3]


def generate_image_atlas_pdf(
    sample_items: list[dict[str, Any]],
    output_pdf_path: Path,
    title: str = "GastricPath AI — Histopathology Image Atlas"
) -> Path:
    """Generates Mode B: Professional Histopathology Image Atlas PDF."""
    styles = get_reporting_styles()
    flowables = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    flowables.append(Spacer(1, 40))
    flowables.append(Paragraph("GASTRICPATH AI COMPUTATIONAL PATHOLOGY", styles["CoverSubtitle"]))
    flowables.append(Paragraph("HISTOPATHOLOGY IMAGE ATLAS &amp; MODEL ANALYSIS DOSSIER", styles["CoverTitle"]))
    flowables.append(Paragraph("Multi-Class Tumor Microenvironment Tissue Decomposition in Gastric Adenocarcinoma", styles["CoverSubtitle"]))
    flowables.append(Spacer(1, 15))

    # Medical Use Notice Callout
    notice_text = (
        "<b>IMPORTANT NOTICE FOR RESEARCH USE ONLY:</b> This document presents microscopic image documentation "
        "and algorithmic research outputs. The tissue classifications and attribution heatmaps herein derive from "
        "patch-level research models (GastricCancerRNN, baseline checkpoint SHA-256: 65369ef6...) evaluated on "
        "224x224 pixel crops from the HMU-GC-HE-30K dataset. They do not constitute a clinical pathology diagnosis, "
        "whole-slide tumor burden quantification, or anatomical cancer staging."
    )
    callout_table = Table([[Paragraph(notice_text, styles["CalloutText"])]], colWidths=[520])
    callout_table.setStyle(get_alert_box_style())
    flowables.append(callout_table)
    flowables.append(Spacer(1, 25))

    # Metadata Block
    meta_data = [
        [Paragraph("<b>Document Type:</b>", styles["TableCell"]), Paragraph("Histopathology Research Image Atlas (Mode B)", styles["TableCell"])],
        [Paragraph("<b>Dataset Source:</b>", styles["TableCell"]), Paragraph("HMU-GC-HE-30K (Harbin Medical University Cancer Hospital)", styles["TableCell"])],
        [Paragraph("<b>Total Plates Included:</b>", styles["TableCell"]), Paragraph(f"{len(sample_items)} Multi-Panel Figure Plates", styles["TableCell"])],
        [Paragraph("<b>Classifier Baseline:</b>", styles["TableCell"]), Paragraph("GastricCancerRNN (Conv2D + BiLSTM + Additive Attention)", styles["TableCell"])],
        [Paragraph("<b>Checkpoint SHA-256:</b>", styles["TableCell"]), Paragraph("65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8", styles["MonoMeta"])],
        [Paragraph("<b>Inference Latency:</b>", styles["TableCell"]), Paragraph("38.56 ms CPU warm inference (lightweight sequence baseline)", styles["TableCell"])],
        [Paragraph("<b>Compilation Date:</b>", styles["TableCell"]), Paragraph("November 12, 2026", styles["TableCell"])],
    ]
    t_meta = Table(meta_data, colWidths=[140, 380])
    t_meta.setStyle(get_clinical_table_style())
    flowables.append(t_meta)
    flowables.append(PageBreak())

    # =========================================================================
    # EXECUTIVE METHODOLOGICAL SUMMARY
    # =========================================================================
    flowables.append(Paragraph("1. Technical Acquisition & Figure Generation Methodology", styles["SectionHeading"]))
    flowables.append(Paragraph(
        "This atlas compiles high-resolution histological microscopic image patches representing 8 distinct "
        "tissue categories within the gastric adenocarcinoma microenvironment: Adipose (ADI), Debris (DEB), "
        "Lymphocytes (LYM), Mucus (MUC), Smooth Muscle (MUS), Normal Mucosa (NOR), Stroma (STR), and Tumor "
        "Epithelium (TUM). Each image plate is presented as a standardized multi-panel figure:",
        styles["BodyText"]
    ))
    
    panel_desc = [
        [Paragraph("<b>Panel A: Original H&E Patch</b>", styles["TableCell"]), Paragraph("Genuine microscopic 224x224 RGB tissue crop. Optical scale is uncalibrated unless indicated by scanner headers.", styles["TableCell"])],
        [Paragraph("<b>Panel B: Prediction Bar Chart</b>", styles["TableCell"]), Paragraph("Complete 8-class softmax probability distribution generated by the certified PyTorch model checkpoint.", styles["TableCell"])],
        [Paragraph("<b>Panel C: Attribution Map</b>", styles["TableCell"]), Paragraph("Grad-CAM feature saliency map reflecting convolutional feature activations across spatial sub-regions.", styles["TableCell"])],
    ]
    t_panel = Table(panel_desc, colWidths=[170, 350])
    t_panel.setStyle(get_clinical_table_style())
    flowables.append(t_panel)
    flowables.append(Spacer(1, 15))

    # =========================================================================
    # IMAGE PLATES (One Plate Per Page)
    # =========================================================================
    flowables.append(Paragraph("2. Histopathological Image Gallery & Model Analyses", styles["SectionHeading"]))
    flowables.append(Spacer(1, 10))

    for idx, item in enumerate(sample_items, 1):
        image_path = item["image_path"]
        sample_id = item.get("sample_id", f"SMP-{idx:02d}")
        known_class = item.get("known_class", "TUM")
        attr_path = item.get("attribution_path")
        
        # Generate multi-panel figure image
        fig_dict = generate_multi_panel_figure(
            image_path=image_path,
            sample_id=sample_id,
            known_class=known_class,
            attribution_image_path=attr_path,
            output_filename=f"atlas_plate_{sample_id}.png"
        )
        
        plate_flowables = []
        plate_flowables.append(Paragraph(f"Plate {idx}: Tissue Sample {sample_id} ({known_class})", styles["SubSectionHeading"]))
        plate_flowables.append(Spacer(1, 4))
        
        # Embed Figure Image (width ~500pt, aspect ratio preserved)
        plate_flowables.append(RLImage(fig_dict["figure_image_path"], width=510, height=180))
        plate_flowables.append(Spacer(1, 6))
        
        # Caption & Scientific Interpretation Boundaries
        plate_flowables.append(Paragraph(f"<b>Caption:</b> {fig_dict['caption']}", styles["Caption"]))
        plate_flowables.append(Spacer(1, 6))

        # Metrics Breakdown Table
        probs = fig_dict["probabilities"]
        top_prob = probs[0]
        runner_up = probs[1] if len(probs) > 1 else {"class_code": "None", "probability": 0.0}
        
        row_data = [
            [Paragraph("<b>Sample ID</b>", styles["TableHeader"]), Paragraph("<b>Known Class</b>", styles["TableHeader"]), Paragraph("<b>Model Prediction</b>", styles["TableHeader"]), Paragraph("<b>Confidence</b>", styles["TableHeader"]), Paragraph("<b>Runner-up Class</b>", styles["TableHeader"])],
            [
                Paragraph(sample_id, styles["TableCell"]),
                Paragraph(known_class, styles["TableCell"]),
                Paragraph(fig_dict["predicted_class"], styles["TableCell"]),
                Paragraph(f"{fig_dict['confidence_score']:.1f}%", styles["TableCell"]),
                Paragraph(f"{runner_up['class_code']} ({runner_up['probability']:.1f}%)", styles["TableCell"])
            ]
        ]
        t_row = Table(row_data, colWidths=[90, 85, 115, 95, 135])
        t_row.setStyle(get_clinical_table_style())
        plate_flowables.append(t_row)
        plate_flowables.append(Spacer(1, 15))

        flowables.append(KeepTogether(plate_flowables))
        if idx < len(sample_items):
            flowables.append(PageBreak())

    # Build and render document
    render_pdf_document(flowables, output_pdf_path, document_title="GastricPath AI — Histopathology Image Atlas")
    return output_pdf_path

