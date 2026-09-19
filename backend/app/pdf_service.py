from io import BytesIO
from typing import Any, Dict, List
import base64

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


def generate_licence_pdf(
    payload: Dict[str, Any],
) -> bytes:

    output = BytesIO()

    pdf = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=4 * mm,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        leading=13,
        spaceAfter=8 * mm,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    )

    normal_style = ParagraphStyle(
        "ReportNormal",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
    )

    small_style = ParagraphStyle(
        "ReportSmall",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
    )

    story: List[Any] = []

    # =====================================================
    # REPORT HEADER
    # =====================================================

    story.append(
        Paragraph(
            "DRIVING LICENCE",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "AI Extraction Report",
            subtitle_style,
        )
    )

    filename = payload.get(
        "filename",
        "Uploaded document",
    )

    story.append(
        Paragraph(
            f"<b>Source Document:</b> "
            f"{filename}",
            normal_style,
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    # =====================================================
    # DOCUMENTS
    # =====================================================

    documents = payload.get(
        "documents",
        [],
    )

    for licence_index, licence in enumerate(
        documents,
        start=1,
    ):

        extraction = licence.get(
            "extraction",
            {},
        )

        if not isinstance(
            extraction,
            dict,
        ):
            extraction = {}

        # -------------------------------------------------
        # Licence heading
        # -------------------------------------------------

        story.append(
            Paragraph(
                f"Licence {licence_index}",
                heading_style,
            )
        )

        # -------------------------------------------------
        # Extracted fields
        # -------------------------------------------------

        rows = [
            [
                Paragraph(
                    "<b>Field</b>",
                    normal_style,
                ),
                Paragraph(
                    "<b>Extracted Value</b>",
                    normal_style,
                ),
            ]
        ]

        fields = [
            (
                "full_name",
                "Full Name",
            ),
            (
                "licence_number",
                "Licence Number",
            ),
            (
                "date_of_birth",
                "Date of Birth",
            ),
            (
                "issue_date",
                "Issue Date",
            ),
            (
                "expiry_date",
                "Expiry Date",
            ),
            (
                "address",
                "Address",
            ),
            (
                "vehicle_classes",
                "Vehicle Classes",
            ),
            (
                "issuing_authority",
                "Issuing Authority",
            ),
            (
                "email",
                "Email Address",
            ),
            (
                "other_information",
                "Other Information",
            ),
        ]

        for field_name, label in fields:

            value = extraction.get(
                field_name
            )

            if value is None:
                value = ""

            if isinstance(
                value,
                list,
            ):
                value = ", ".join(
                    str(item)
                    for item in value
                )

            value = str(
                value
            ).strip()

            if not value:
                value = "Not available"

            rows.append(
                [
                    Paragraph(
                        label,
                        normal_style,
                    ),
                    Paragraph(
                        value,
                        normal_style,
                    ),
                ]
            )

        table = Table(
            rows,
            colWidths=[
                50 * mm,
                125 * mm,
            ],
            repeatRows=1,
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        story.append(table)

        story.append(
            Spacer(
                1,
                5 * mm,
            )
        )

        # -------------------------------------------------
        # Source information
        # -------------------------------------------------

        story.append(
            Paragraph(
                "Source Information",
                heading_style,
            )
        )

        source_rows = [
            [
                Paragraph(
                    "<b>Page</b>",
                    normal_style,
                ),
                Paragraph(
                    "<b>Region</b>",
                    normal_style,
                ),
            ],
            [
                str(
                    licence.get(
                        "page",
                        "",
                    )
                ),
                str(
                    licence.get(
                        "region",
                        "",
                    )
                ),
            ],
        ]

        source_table = Table(
            source_rows,
            colWidths=[
                50 * mm,
                50 * mm,
            ],
        )

        source_table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),
                ]
            )
        )

        story.append(
            source_table
        )

        # -------------------------------------------------
        # Document preview
        # -------------------------------------------------

        preview_data_url = licence.get(
            "preview_data_url"
        )

        if preview_data_url:

            try:

                if "," in preview_data_url:

                    _, encoded = (
                        preview_data_url.split(
                            ",",
                            1,
                        )
                    )

                    image_bytes = (
                        base64.b64decode(
                            encoded
                        )
                    )

                    image_buffer = BytesIO(
                        image_bytes
                    )

                    story.append(
                        Spacer(
                            1,
                            5 * mm,
                        )
                    )

                    story.append(
                        Paragraph(
                            "Document Preview",
                            heading_style,
                        )
                    )

                    img = Image(
                        image_buffer,
                        width=170 * mm,
                        height=100 * mm,
                    )

                    img.hAlign = "CENTER"

                    story.append(
                        img
                    )

            except Exception as exc:

                print(
                    "[PDF] Unable to add "
                    f"document preview: {exc}"
                )

        # -------------------------------------------------
        # Separate licences
        # -------------------------------------------------

        if licence_index < len(
            documents
        ):

            story.append(
                PageBreak()
            )

    # =====================================================
    # FOOTER NOTE
    # =====================================================

    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )

    story.append(
        Paragraph(
            "Generated using OCR and AI-based "
            "structured document extraction. "
            "Extracted information should be reviewed "
            "before official use.",
            small_style,
        )
    )

    # =====================================================
    # BUILD PDF
    # =====================================================

    pdf.build(
        story
    )

    return output.getvalue()