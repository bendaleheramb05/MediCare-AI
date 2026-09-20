"""
MediCare AI - PDF health report generator (ReportLab)
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (HRFlowable, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from predictor import DISCLAIMER

NAVY = colors.HexColor("#123b63")
TEAL = colors.HexColor("#0d9488")
LIGHT = colors.HexColor("#eef4fa")
RISK_COLORS = {"Low Risk": colors.HexColor("#1a7f4b"),
               "Moderate Risk": colors.HexColor("#b45309"),
               "High Risk": colors.HexColor("#b91c1c")}


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("Head", parent=s["Title"], fontSize=20,
                         textColor=NAVY, spaceAfter=2))
    s.add(ParagraphStyle("Sub", parent=s["Normal"], fontSize=9.5,
                         textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10))
    s.add(ParagraphStyle("H2", parent=s["Heading2"], fontSize=12.5,
                         textColor=NAVY, spaceBefore=12, spaceAfter=5))
    s.add(ParagraphStyle("Body", parent=s["Normal"], fontSize=9.5, leading=13.5))
    s.add(ParagraphStyle("Big", parent=s["Normal"], fontSize=10, leading=26))
    s.add(ParagraphStyle("Just", parent=s["Normal"], fontSize=8.2, leading=11.5,
                         alignment=TA_JUSTIFY, textColor=colors.HexColor("#7f1d1d")))
    return s


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d8e2ec"))
    canvas.line(18 * mm, 15 * mm, A4[0] - 18 * mm, 15 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawString(18 * mm, 10.5 * mm,
                      "MediCare AI - Preliminary AI health-risk assessment. Not a medical diagnosis.")
    canvas.drawRightString(A4[0] - 18 * mm, 10.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _table(rows, widths):
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d6e3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _bullets(items, st):
    return ListFlowable([ListItem(Paragraph(i, st["Body"]), leftIndent=12)
                         for i in items], bulletType="bullet", start="•",
                        leftIndent=14, bulletFontSize=8)


def build_pdf(patient, form, result, created_at=None, metrics=None):
    """Returns the PDF as a BytesIO buffer."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="MediCare AI Health Report",
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=20 * mm)
    st = _styles()
    story = []
    when = created_at or datetime.now().strftime("%d %b %Y, %I:%M %p")

    # ---------------- header -------------------------------------------
    story += [Paragraph("MediCare AI", st["Head"]),
              Paragraph("Preliminary AI-Based Health Risk Assessment Report", st["Sub"]),
              HRFlowable(width="100%", color=TEAL, thickness=1.5),
              Spacer(1, 10)]

    # ---------------- patient ------------------------------------------
    story.append(Paragraph("1. Patient Information", st["H2"]))
    story.append(_table([
        ["Field", "Detail", "Field", "Detail"],
        ["Name", patient["full_name"], "Report Date", when],
        ["Age", f"{form['age']} years", "Gender", form["gender"]],
        ["Email", patient["email"], "Report ID", f"MCA-{patient['id']}-{form.get('assessment_id', '000')}"],
    ], [26 * mm, 52 * mm, 26 * mm, 52 * mm]))

    # ---------------- parameters ---------------------------------------
    story.append(Paragraph("2. Health Parameters", st["H2"]))
    story.append(_table([
        ["Parameter", "Value", "Reference Range", "Status"],
        ["Height", f"{form['height']} cm", "-", "-"],
        ["Weight", f"{form['weight']} kg", "-", "-"],
        ["BMI", f"{result['bmi']} kg/m2", "18.5 - 24.9", result["bmi_category"]],
        ["Systolic BP", "Not provided" if form.get("systolic_bp") is None else f"{form['systolic_bp']} mmHg",
         "-" if form.get("systolic_bp") is None else "< 120", "-" if form.get("systolic_bp") is None else result["bp_category"]],
        ["Diastolic BP", "Not provided" if form.get("diastolic_bp") is None else f"{form['diastolic_bp']} mmHg",
         "-" if form.get("diastolic_bp") is None else "< 80", "-" if form.get("diastolic_bp") is None else result["bp_category"]],
        ["Blood Sugar Category", result.get("glucose_label", result["sugar_category"]), "Dataset category",
         result["sugar_category"]],
        ["Total Cholesterol Category", result.get("cholesterol_label", result["cholesterol_category"]), "Dataset category",
         result["cholesterol_category"]],
        ["Heart Rate", "Unavailable in dataset", "-", result["pulse_category"]],
        ["Smoking", "Yes" if form["smoking"] else "No", "No", "-"],
        ["Alcohol", "Yes" if form["alcohol"] else "No", "No / minimal", "-"],
        ["Physically Active", "Yes" if form["physical_activity"] else "No",
         "150 min/week", "-"],
        ["Family History", "Yes" if form["family_history"] else "No", "-", "-"],
    ], [38 * mm, 40 * mm, 40 * mm, 38 * mm]))

    # ---------------- risk ----------------------------------------------
    story.append(Paragraph("3. AI Risk Assessment", st["H2"]))
    rc = RISK_COLORS.get(result["risk_level"], NAVY)
    risk_tbl = Table([[
        Paragraph(f"<font size=22 color='{rc.hexval()}'><b>{result['risk_score']}%</b></font>"
                  f"<br/><font size=8 color='#555555'>Overall Health Screening Score</font>", st["Big"]),
        Paragraph(f"<font size=13 color='{rc.hexval()}'><b>{result['risk_level']}</b></font>"
                  f"<br/><font size=8 color='#555555'>Based on the health information provided in this assessment.</font>", st["Big"]),
    ]], colWidths=[52 * mm, 104 * mm])
    risk_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.8, rc),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story += [risk_tbl, Spacer(1, 8)]

    if result["conditions"]:
        story.append(Paragraph("<b>Possible Risk Factors to Discuss with a Doctor</b>", st["Body"]))
        story.append(Spacer(1, 3))
        story.append(_bullets([f"<b>{c}</b> - {d}" for c, d in result["conditions"]], st))

    if result["symptoms"]:
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Symptoms You Reported</b>", st["Body"]))
        story.append(Spacer(1, 3))
        story.append(_bullets(result["symptoms"], st))
        story.append(Paragraph("Persistent, severe, or concerning symptoms should be discussed with a healthcare professional.", st["Body"]))

    # ---------------- factors -------------------------------------------
    if result["factors"]:
        story.append(Paragraph("4. Key Factors Behind This Result", st["H2"]))
        story.append(_bullets([f"<b>{name}</b> - {why}" for name, why, _ in result["factors"]], st))

    # ---------------- advice --------------------------------------------
    story.append(Paragraph("5. General Health &amp; Lifestyle Advice", st["H2"]))
    rec = result["recommendations"]
    story.append(Paragraph("<b>Personalized Educational Guidance</b>", st["Body"]))
    story.append(Paragraph("Recommendations are based on the health information provided in this assessment.", st["Body"]))
    story.append(Paragraph("<b>Why this guidance?</b>", st["Body"]))
    story.append(_bullets(rec["why_guidance"], st))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Recommended food options</b>", st["Body"]))
    story.append(_bullets(rec["diet_guidance"], st))
    story.append(Spacer(1, 3))
    story.append(Paragraph("<b>Recommended Fruits</b>", st["Body"]))
    story.append(_bullets(rec["recommended_fruits"], st))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Recommended Vegetables</b>", st["Body"]))
    story.append(_bullets(rec["recommended_vegetables"], st))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Healthy Protein / Other Foods</b>", st["Body"]))
    story.append(_bullets(rec["healthy_protein_other"], st))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Foods to Limit</b>", st["Body"]))
    story.append(_bullets(rec["foods_to_limit"], st))
    story.append(Spacer(1, 5))
    story.append(Paragraph("<b>Lifestyle guidance</b>", st["Body"]))
    story.append(_bullets(rec["lifestyle"], st))
    story.append(Spacer(1, 5))
    story.append(Paragraph("<b>When to Seek Professional Advice</b>", st["Body"]))
    story.append(_bullets(rec["monitoring"], st))

    story.append(Paragraph("6. Safe General Exercise Suggestions", st["H2"]))
    story.append(_table([["Activity", "Suggestion"]] +
                        [[Paragraph(a, st["Body"]), Paragraph(d, st["Body"])]
                         for a, d in result["exercises"]],
                        [42 * mm, 114 * mm]))

    # ---------------- disclaimer ----------------------------------------
    story.append(Spacer(1, 12))
    d = Table([[Paragraph(DISCLAIMER, st["Just"])]], colWidths=[156 * mm])
    d.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#dc2626")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(d)

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buf.seek(0)
    return buf
