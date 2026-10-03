"""Render the fork's engineering note. Development dependency: reportlab."""

from html import escape
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "docs/paper/Strata-Unleashed.md"
    target = source.with_suffix(".pdf")
    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#183b56")
    styles["Heading2"].textColor = colors.HexColor("#007f82")
    styles["Heading2"].keepWithNext = True
    styles.add(
        ParagraphStyle(
            name="NoteBody", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=9
        )
    )
    story = []
    for block in source.read_text().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        style = styles["NoteBody"]
        if block.startswith("### "):
            style, block = styles["Heading2"], block[4:]
        elif block.startswith("## "):
            style, block = styles["Heading2"], block[3:]
        elif block.startswith("# "):
            style, block = styles["Title"], block[2:]
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", escape(block)).replace(
            "\n", "<br/>"
        )
        story.append(Paragraph(text, style))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#007f82"))
        canvas.line(48, 40, A4[0] - 48, 40)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            48,
            27,
            "Strata Unleashed | Engineering note | Upstream Strata credited separately",
        )
        canvas.drawRightString(A4[0] - 48, 27, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        rightMargin=48,
        leftMargin=48,
        topMargin=42,
        bottomMargin=55,
        title="Strata Unleashed: model selection and per-machine inference tuning",
        author="Strata Unleashed project",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(target)


if __name__ == "__main__":
    main()
