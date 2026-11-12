# GASTRICPATH AI — PROFESSIONAL PDF GENERATION ENGINE GUIDE

**Document Identifier:** `docs/PDF_GENERATION_GUIDE.md`  
**Module:** ReportLab Document Compilation Engine  
**Status:** Authoritative Operational Guide  
**Date:** November 2026  

---

## 1. Document Architecture & Standards

GastricPath AI produces publication-grade medical and scientific documents using ReportLab Platypus architecture combined with a two-pass `NumberedCanvas`.

### Key Specifications:
- **Geometry:** Standard International A4 ($595.27 \times 841.89$ points / $210 \times 297$ mm).
- **Margins:** 36 pt (0.5 in) left and right; 42 pt top; 48 pt bottom.
- **Typography:** Helvetica, Helvetica-Bold, Helvetica-Oblique, and Courier.
- **Two-Pass Numbering:** Computes the total page count across all pages and renders running footers in the format:  
  `Page X of Y`
- **Running Headers & Footers:** Features thin rule separators, document confidentiality tags, date stamps, and medical disclaimers on every page following the cover page.
- **Table Formatting:** Explicit column widths, autowrapping paragraph cells, alternating row fills (`#ffffff` and `#f8fafc`), and dark navy headers (`#0f172a`).

---

## 2. Supported Report Modes

| Mode | Function | Target Audience | Primary Artifacts |
|:---|:---|:---|:---|
| **Mode A** | Gastric Cancer Clinical Case Review Dossier | Multidisciplinary Tumor Boards, Oncologists | Full 20-section structured case, CAP synoptic data, alignment matrix, MDT checklist |
| **Mode B** | Histopathology Image Atlas | Pathologists, Deep Learning Researchers | Plate-by-plate multi-panel figures, Grad-CAM heatmaps, probability charts |
| **Mode C** | Gastric Cancer Research Manuscript | Scientific Peer Reviewers, Journal Submission | CLAIM 2024 / TRIPOD+AI compliant manuscript, statistical tables, Vancouver references |

---

## 3. Python API Usage

```python
from pathlib import Path
from research.clinical_reporting.schemas.case_dossier import CaseDossier
from research.clinical_reporting.report_engine.case_review_generator import generate_case_review_dossier_pdf
from research.clinical_reporting.report_engine.image_atlas_generator import generate_image_atlas_pdf
from research.clinical_reporting.report_engine.manuscript_generator import generate_research_manuscript_pdf

# 1. Generate Mode A: Clinical Case Dossier
dossier = CaseDossier(...)
pdf_a = generate_case_review_dossier_pdf(dossier, Path("output/Case_Review.pdf"))

# 2. Generate Mode B: Image Atlas
sample_items = [
    {"image_path": Path("data/raw/sample.png"), "sample_id": "SMP-01", "known_class": "TUM"}
]
pdf_b = generate_image_atlas_pdf(sample_items, Path("output/Image_Atlas.pdf"))

# 3. Generate Mode C: Research Manuscript
pdf_c = generate_research_manuscript_pdf(Path("output/Manuscript.pdf"))
```

---

## 4. Visual Quality Guidelines

1. **Table Integrity:** Tables automatically wrap text within ReportLab `Paragraph` cells; no text clipping or horizontal truncation.
2. **High-DPI Embeddings:** Figures and image plates render at 200–300 DPI, ensuring clean readability when printed or zoomed on high-resolution displays.
3. **No Empty Pages:** Document sections use `KeepTogether` and calculated `Spacer` heights to prevent orphan headings and unintentional blank pages.
