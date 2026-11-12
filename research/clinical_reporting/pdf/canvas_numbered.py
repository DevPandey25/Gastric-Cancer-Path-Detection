"""
Custom Numbered Canvas with Two-Pass Page Numbering and Running Headers/Footers.
GastricPath AI — Phase 9: Professional PDF Generation Engine.

Renders 'Page X of Y', running headers with confidentiality tags,
running medical disclaimers in footers, and PDF document outline bookmarks.
"""

from reportlab.pdfgen import canvas
from reportlab.lib import colors


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas calculating total page count for running footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Skip headers and footers on Cover Page (Page 1)
        if self._pageNumber > 1:
            # --- Running Header ---
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(40, 802, 555, 802)
            
            self.drawString(40, 808, "GASTRICPATH AI — Clinical Research & Reporting Platform")
            self.drawRightString(555, 808, "CONFIDENTIAL / RESEARCH USE ONLY")

            # --- Running Footer ---
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(40, 45, 555, 45)
            
            disclaimer = "DRAFT — For multidisciplinary team review. Not an autonomous diagnostic device."
            self.drawString(40, 32, disclaimer)
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(555, 32, page_text)
            
        self.restoreState()

