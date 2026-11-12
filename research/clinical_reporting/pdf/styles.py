"""
Professional Medical Document Typography and Table Styles.
GastricPath AI — Phase 9: Professional PDF Generation Engine.

Defines styling hierarchy for clinical dossiers, image atlases, and manuscripts.
"""

from typing import Any
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import TableStyle


def get_reporting_styles() -> dict[str, Any]:
    """Generates an authoritative, styled paragraph stylesheet."""
    base = getSampleStyleSheet()
    styles = {}

    # Cover Title
    styles["CoverTitle"] = ParagraphStyle(
        "CoverTitle",
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        alignment=0,
        spaceAfter=12
    )

    # Cover Subtitle
    styles["CoverSubtitle"] = ParagraphStyle(
        "CoverSubtitle",
        fontName="Helvetica",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0d9488"),
        alignment=0,
        spaceAfter=20
    )

    # Document Section Heading (H1)
    styles["SectionHeading"] = ParagraphStyle(
        "SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    # Subsection Heading (H2)
    styles["SubSectionHeading"] = ParagraphStyle(
        "SubSectionHeading",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    # Standard Body Text
    styles["BodyText"] = ParagraphStyle(
        "ReportingBody",
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )

    # Small Body / Captions
    styles["Caption"] = ParagraphStyle(
        "Caption",
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8
    )

    # Callout / Alert Box Text
    styles["CalloutText"] = ParagraphStyle(
        "CalloutText",
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#991b1b")
    )

    # Table Cell Text
    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )

    # Table Header Cell Text
    styles["TableHeader"] = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    # Code / Metadata Span
    styles["MonoMeta"] = ParagraphStyle(
        "MonoMeta",
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f766e")
    )

    return styles


def get_clinical_table_style() -> TableStyle:
    """Standard clinical table formatting with clean alternating shading."""
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
    ])


def get_alert_box_style() -> TableStyle:
    """Styling for warning and clinical disclaimer callout boxes."""
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#f87171")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])
