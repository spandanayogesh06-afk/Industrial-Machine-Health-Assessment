from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .assessment import maintenance_recommendations


def generate_assessment_pdf(
    assessment: dict,
    machine_values: dict,
    model_filename: str,
    image_filename: str | None = None,
) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Industrial Machine Health Assessment",
        author="Industrial Machine Health Assessment",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=9, leading=13))
    story = [
        Paragraph("Industrial Machine Health Assessment", styles["Title"]),
        Paragraph("Machine Failure Prediction &amp; Maintenance Assistant", styles["Normal"]),
        Spacer(1, 10),
        Paragraph(f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}", styles["SmallBody"]),
        Paragraph(f"Model artifact: {escape(model_filename)}", styles["SmallBody"]),
        Paragraph(f"Machine image: {escape(image_filename or 'Not provided')}", styles["SmallBody"]),
        Spacer(1, 14),
        Paragraph("Assessment", styles["Heading2"]),
    ]

    status = "Failure class predicted" if assessment["predicted_failure"] else "Failure class not predicted"
    summary = Table(
        [
            ["Result", status],
            ["Estimated failure probability", f"{assessment['failure_probability']:.1%}"],
            ["Risk band", assessment["risk_level"]],
            ["Machine type", str(machine_values["Machine Type"])],
        ],
        colWidths=[2.1 * inch, 4.6 * inch],
    )
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eef0")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#18272d")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d3d6")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(summary)
    story.extend([Spacer(1, 14), Paragraph("Entered operating parameters", styles["Heading2"])])

    feature_labels = [
        ("Air temperature [K]", "Air temperature [K]"),
        ("Process temperature [K]", "Process temperature [K]"),
        ("Rotational speed [rpm]", "Rotational speed [rpm]"),
        ("Torque [Nm]", "Torque [Nm]"),
        ("Tool wear [min]", "Tool wear [min]"),
    ]
    parameter_rows = [["Parameter", "Entered value"]]
    for label, key in feature_labels:
        parameter_rows.append([label, f"{float(machine_values[key]):g}"])
    parameter_table = Table(parameter_rows, colWidths=[3.2 * inch, 3.5 * inch], repeatRows=1)
    parameter_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176b68")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d3d6")),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ]
        )
    )
    story.append(parameter_table)

    story.extend([Spacer(1, 14), Paragraph("Model explanation", styles["Heading2"])])
    story.append(Paragraph(escape(assessment["explanation_source"]), styles["SmallBody"]))
    if assessment["factors"]:
        for factor in assessment["factors"]:
            if "rule" in factor:
                line = f"{factor['rule']} (entered value: {factor['value']:g})"
            else:
                line = f"{factor['feature']}: overall importance {factor['importance']:.3f}"
            story.append(Paragraph(f"&bull; {escape(line)}", styles["SmallBody"]))
    else:
        story.append(Paragraph("No feature-level explanation is exposed by this model artifact.", styles["SmallBody"]))

    story.extend([Spacer(1, 14), Paragraph("Maintenance recommendations", styles["Heading2"])])
    for recommendation in maintenance_recommendations(assessment):
        story.append(Paragraph(f"&bull; {escape(recommendation)}", styles["SmallBody"]))

    story.extend(
        [
            Spacer(1, 14),
            Paragraph(
                "Limitations: the uploaded image is visual reference only. Failure prediction uses entered operating "
                "parameters and the loaded model. The probability is the model output, not a guarantee, calibrated "
                "failure rate, or substitute for qualified inspection and site safety procedures.",
                styles["SmallBody"],
            ),
        ]
    )
    document.build(story)
    return buffer.getvalue()
