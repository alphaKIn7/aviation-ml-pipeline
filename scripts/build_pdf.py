#!/usr/bin/env python3
"""
Build a PDF from all Markdown notes in docs/notes/.

Usage:
    python scripts/build_pdf.py

To add a new topic:
    1. Create a new .md file inside docs/notes/  (e.g. 05_my_new_topic.md)
    2. Re-run this script.
    Files are included in alphabetical order by filename, so prefix with a number.

Output: docs/aviation_ml_notes.pdf
"""

from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "docs" / "notes"
OUTPUT = ROOT / "docs" / "aviation_ml_notes.pdf"

MARGIN = 18
PAGE_W = 210          # A4 width in mm
CONTENT_W = PAGE_W - 2 * MARGIN

BODY_SIZE = 10
LINE_H = 5.2          # line height in mm for body text
CODE_SIZE = 8.5
CODE_LINE_H = 4.5

# Colour palette
C = {
    "h1_bg":   (25, 70, 145),
    "h1_fg":   (255, 255, 255),
    "h2_fg":   (25, 70, 145),
    "h3_fg":   (60, 60, 60),
    "code_bg": (245, 246, 248),
    "code_fg": (40, 40, 80),
    "border":  (200, 200, 210),
    "th_bg":   (215, 228, 248),
    "th_fg":   (25, 70, 145),
    "body":    (35, 35, 35),
    "rule":    (175, 175, 185),
    "cover":   (20, 55, 120),
}


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

def safe(text: str) -> str:
    """Replace characters outside Latin-1 with readable ASCII equivalents."""
    replacements = {
        "→": "->",   "←": "<-",   "↔": "<->",
        "—": "--",   "–": "-",
        "•": "-",    "●": "-",
        "‘": "'",    "’": "'",
        "“": '"',    "”": '"',
        "≤": "<=",   "≥": ">=",   "≠": "!=",
        "≈": "~=",   "…": "...",
        "×": "x",    "÷": "/",
        "°": "deg",  "∞": "inf",
        "α": "alpha","β": "beta",
        "μ": "mu",   "σ": "sigma",
        "∑": "sum",  "∏": "prod",
    }
    for char, rep in replacements.items():
        text = text.replace(char, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ---------------------------------------------------------------------------
# PDF class (footer only)
# ---------------------------------------------------------------------------

class NotesPDF(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(155, 155, 165)
        self.cell(0, 8,
                  f"Aviation ML Pipeline -- Learning Notes  |  Page {self.page_no()}",
                  align="C")
        self.set_text_color(*C["body"])


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class Renderer:
    def __init__(self):
        self.pdf = NotesPDF(orientation="P", unit="mm", format="A4")
        self.pdf.set_margins(MARGIN, MARGIN, MARGIN)
        self.pdf.set_auto_page_break(auto=True, margin=20)
        self.md = MarkdownIt()
        self._list_depth = 0
        self._pending_bullet = False

    # ------------------------------------------------------------------
    # Cover page
    # ------------------------------------------------------------------

    def cover(self, title: str, subtitle: str, date: str):
        pdf = self.pdf
        pdf.add_page()

        pdf.set_fill_color(*C["cover"])
        pdf.rect(0, 0, 210, 297, "F")

        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_xy(MARGIN, 85)
        pdf.multi_cell(CONTENT_W, 12, safe(title), align="C")

        y = pdf.get_y() + 7
        pdf.set_draw_color(200, 210, 235)
        pdf.set_line_width(0.6)
        pdf.line(MARGIN + 15, y, MARGIN + CONTENT_W - 15, y)

        pdf.set_font("Helvetica", "", 12)
        pdf.set_xy(MARGIN, y + 9)
        pdf.multi_cell(CONTENT_W, 7, safe(subtitle), align="C")

        pdf.set_font("Helvetica", "I", 10)
        pdf.set_xy(MARGIN, 252)
        pdf.cell(CONTENT_W, 7, safe(date), align="C")

        pdf.set_text_color(*C["body"])
        pdf.add_page()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_file(self, path: Path):
        """Parse a single Markdown file and add it to the PDF."""
        content = path.read_text(encoding="utf-8")
        tokens = self.md.parse(content)
        self._list_depth = 0
        self._pending_bullet = False
        self._render_tokens(tokens)
        self.pdf.ln(6)

    def save(self, output: Path):
        self.pdf.output(str(output))
        print(f"Saved: {output}")

    # ------------------------------------------------------------------
    # Token dispatch loop
    # ------------------------------------------------------------------

    def _render_tokens(self, tokens):
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            t = tok.type

            if t == "heading_open":
                text = self._plain(tokens[i + 1].children)
                self._heading(tok.tag, text)
                i += 3
                continue

            elif t == "paragraph_open":
                inline = tokens[i + 1]
                if self._pending_bullet:
                    self._list_item(inline.children)
                    self._pending_bullet = False
                else:
                    self._paragraph(inline.children)
                i += 3
                continue

            elif t == "inline":
                # tight list items have inline directly (no paragraph wrapper)
                if self._pending_bullet:
                    self._list_item(tok.children)
                    self._pending_bullet = False

            elif t == "fence":
                self._code_block(tok.content)

            elif t == "bullet_list_open":
                self._list_depth += 1

            elif t == "bullet_list_close":
                self._list_depth -= 1
                if self._list_depth == 0:
                    self.pdf.ln(2)

            elif t == "list_item_open":
                self._pending_bullet = True

            elif t == "hr":
                self._hr()

            elif t == "table_open":
                j = i + 1
                table_toks = []
                while j < len(tokens) and tokens[j].type != "table_close":
                    table_toks.append(tokens[j])
                    j += 1
                self._table(table_toks)
                i = j + 1
                continue

            i += 1

    # ------------------------------------------------------------------
    # Element renderers
    # ------------------------------------------------------------------

    def _plain(self, children) -> str:
        if not children:
            return ""
        out = []
        for c in children:
            if c.type in ("text", "code_inline"):
                out.append(c.content)
            elif c.type == "softbreak":
                out.append(" ")
        return "".join(out)

    def _heading(self, tag: str, text: str):
        pdf = self.pdf
        text = safe(text)

        if tag == "h1":
            pdf.ln(3)
            pdf.set_fill_color(*C["h1_bg"])
            pdf.set_text_color(*C["h1_fg"])
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(CONTENT_W, 9, "  " + text, fill=True, align="L")
            pdf.ln(3)

        elif tag == "h2":
            pdf.ln(5)
            pdf.set_text_color(*C["h2_fg"])
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            y = pdf.get_y()
            pdf.set_draw_color(*C["h2_fg"])
            pdf.set_line_width(0.4)
            pdf.line(MARGIN, y, MARGIN + CONTENT_W, y)
            pdf.ln(3)

        elif tag == "h3":
            pdf.ln(3)
            pdf.set_text_color(*C["h3_fg"])
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)

        pdf.set_text_color(*C["body"])
        pdf.set_font("Helvetica", "", BODY_SIZE)

    def _paragraph(self, children):
        if not children:
            return
        pdf = self.pdf
        pdf.set_font("Helvetica", "", BODY_SIZE)
        pdf.set_text_color(*C["body"])
        bold = False
        italic = False

        for child in children:
            ct = child.type
            if ct == "strong_open":
                bold = True
                pdf.set_font("Helvetica", "BI" if italic else "B", BODY_SIZE)
            elif ct == "strong_close":
                bold = False
                pdf.set_font("Helvetica", "I" if italic else "", BODY_SIZE)
            elif ct == "em_open":
                italic = True
                pdf.set_font("Helvetica", "BI" if bold else "I", BODY_SIZE)
            elif ct == "em_close":
                italic = False
                pdf.set_font("Helvetica", "B" if bold else "", BODY_SIZE)
            elif ct == "code_inline":
                cur_style = "BI" if (bold and italic) else "B" if bold else "I" if italic else ""
                pdf.set_font("Courier", "", CODE_SIZE)
                pdf.set_text_color(*C["code_fg"])
                pdf.write(LINE_H, safe(f" {child.content} "))
                pdf.set_text_color(*C["body"])
                pdf.set_font("Helvetica", cur_style, BODY_SIZE)
            elif ct == "text":
                pdf.write(LINE_H, safe(child.content))
            elif ct == "softbreak":
                pdf.write(LINE_H, " ")
            elif ct == "hardbreak":
                pdf.ln(LINE_H)

        pdf.ln(LINE_H + 1)
        pdf.set_font("Helvetica", "", BODY_SIZE)
        pdf.set_text_color(*C["body"])

    def _list_item(self, children):
        pdf = self.pdf
        indent = MARGIN + (self._list_depth - 1) * 5
        pdf.set_x(indent)
        pdf.set_font("Helvetica", "", BODY_SIZE)
        pdf.set_text_color(*C["body"])
        pdf.write(LINE_H, "-  ")
        bold = False

        for child in children:
            ct = child.type
            if ct == "strong_open":
                bold = True
                pdf.set_font("Helvetica", "B", BODY_SIZE)
            elif ct == "strong_close":
                bold = False
                pdf.set_font("Helvetica", "", BODY_SIZE)
            elif ct == "code_inline":
                pdf.set_font("Courier", "", CODE_SIZE)
                pdf.set_text_color(*C["code_fg"])
                pdf.write(LINE_H, safe(f"`{child.content}`"))
                pdf.set_text_color(*C["body"])
                pdf.set_font("Helvetica", "B" if bold else "", BODY_SIZE)
            elif ct == "text":
                pdf.write(LINE_H, safe(child.content))
            elif ct == "softbreak":
                pdf.write(LINE_H, " ")

        pdf.ln(LINE_H + 0.5)
        pdf.set_font("Helvetica", "", BODY_SIZE)
        pdf.set_text_color(*C["body"])

    def _code_block(self, content: str):
        pdf = self.pdf
        lines = content.rstrip("\n").split("\n")
        padding = 3.0
        total_h = len(lines) * CODE_LINE_H + 2 * padding

        pdf.ln(2)
        y0 = pdf.get_y()

        if y0 + total_h > pdf.h - pdf.b_margin:
            pdf.add_page()
            y0 = pdf.get_y()

        pdf.set_fill_color(*C["code_bg"])
        pdf.set_draw_color(*C["border"])
        pdf.set_line_width(0.3)
        pdf.rect(MARGIN - 2, y0, CONTENT_W + 4, total_h, "DF")

        pdf.set_font("Courier", "", CODE_SIZE)
        pdf.set_text_color(*C["code_fg"])
        pdf.set_y(y0 + padding)

        for line in lines:
            pdf.set_x(MARGIN + 2)
            pdf.cell(CONTENT_W - 4, CODE_LINE_H, safe(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_text_color(*C["body"])
        pdf.set_font("Helvetica", "", BODY_SIZE)
        pdf.ln(4)

    def _hr(self):
        pdf = self.pdf
        pdf.ln(3)
        y = pdf.get_y()
        pdf.set_draw_color(*C["rule"])
        pdf.set_line_width(0.4)
        pdf.line(MARGIN, y, MARGIN + CONTENT_W, y)
        pdf.ln(5)

    def _table(self, tokens):
        pdf = self.pdf
        rows = []
        in_header = False
        current_row = None
        in_cell = False

        for tok in tokens:
            if tok.type == "thead_open":
                in_header = True
            elif tok.type == "thead_close":
                in_header = False
            elif tok.type == "tr_open":
                current_row = {"header": in_header, "cells": []}
            elif tok.type == "tr_close":
                if current_row is not None:
                    rows.append(current_row)
                current_row = None
            elif tok.type in ("th_open", "td_open"):
                in_cell = True
            elif tok.type in ("th_close", "td_close"):
                in_cell = False
            elif tok.type == "inline" and in_cell and current_row is not None:
                current_row["cells"].append(safe(self._plain(tok.children)))

        if not rows:
            return

        num_cols = max(len(r["cells"]) for r in rows)
        if num_cols == 0:
            return

        col_w = CONTENT_W / num_cols
        cell_h = 6.5

        pdf.ln(3)
        pdf.set_draw_color(*C["border"])
        pdf.set_line_width(0.3)

        for row in rows:
            is_header = row["header"]
            cells = row["cells"]

            if is_header:
                pdf.set_fill_color(*C["th_bg"])
                pdf.set_text_color(*C["th_fg"])
                pdf.set_font("Helvetica", "B", 9)
            else:
                pdf.set_fill_color(250, 251, 255)
                pdf.set_text_color(*C["body"])
                pdf.set_font("Helvetica", "", 9)

            for cell in cells:
                pdf.cell(col_w, cell_h, cell, border=1, fill=True)
            for _ in range(num_cols - len(cells)):
                pdf.cell(col_w, cell_h, "", border=1, fill=True)
            pdf.ln()

        pdf.set_font("Helvetica", "", BODY_SIZE)
        pdf.set_text_color(*C["body"])
        pdf.ln(4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not NOTES_DIR.exists():
        print(f"ERROR: Notes directory not found: {NOTES_DIR}")
        return

    files = sorted(NOTES_DIR.glob("*.md"))
    if not files:
        print("No .md files found in docs/notes/")
        return

    print(f"Found {len(files)} note file(s):")
    for f in files:
        print(f"  {f.name}")

    r = Renderer()
    r.cover(
        title="Aviation ML Pipeline\nLearning Notes",
        subtitle="Feature Engineering & Feature Selection\nExplanations, Concepts, and Code Walkthroughs",
        date="May 2026",
    )

    for f in files:
        print(f"Rendering: {f.name} ...")
        r.render_file(f)

    r.save(OUTPUT)
    print(f"\nDone. Open the PDF at:\n  {OUTPUT}")


if __name__ == "__main__":
    main()
