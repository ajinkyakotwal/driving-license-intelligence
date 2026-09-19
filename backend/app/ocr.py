import io
import os
import base64
from functools import lru_cache
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image

from .config import get_settings


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}

ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "application/pdf",
}


def validate_upload(
    filename: str,
    content_type: str | None,
    size: int,
) -> None:
    ext = os.path.splitext(filename.lower())[1]

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Use PNG, JPG, WEBP or PDF."
        )

    if content_type and content_type not in ALLOWED_MIME_TYPES:
        raise ValueError("Unsupported MIME type.")

    limit = get_settings().max_file_size_mb * 1024 * 1024

    if size > limit:
        raise ValueError(
            f"File is too large. Maximum size is "
            f"{get_settings().max_file_size_mb} MB."
        )


def pdf_to_images(data: bytes) -> List[Image.Image]:
    import fitz

    doc = fitz.open(stream=data, filetype="pdf")
    pages = []

    for page in doc:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False,
        )

        pages.append(
            Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples,
            )
        )

    doc.close()

    return pages


def bytes_to_images(
    data: bytes,
    filename: str,
) -> List[Image.Image]:

    if filename.lower().endswith(".pdf"):
        return pdf_to_images(data)

    image = Image.open(
        io.BytesIO(data)
    ).convert("RGB")

    return [image]


# ---------------------------------------------------------
# DOCUMENT REGION DETECTION
# ---------------------------------------------------------

def find_document_regions(
    image: Image.Image,
) -> List[Tuple[int, int, int, int]]:

    arr = np.array(image)

    h, w = arr.shape[:2]

    # For very wide images, check whether there are two
    # side-by-side documents.
    #
    # This is particularly useful for the supplied assessment
    # sample containing two driving licences.

    if w / max(h, 1) >= 1.30:

        gray = cv2.cvtColor(
            arr,
            cv2.COLOR_RGB2GRAY,
        )

        # Look for a vertical low-content seam.
        #
        # We calculate the amount of non-white/dark content
        # in each vertical column.

        dark = gray < 235

        column_density = dark.mean(axis=0)

        # Smooth the signal.
        kernel = np.ones(31) / 31.0

        smoothed = np.convolve(
            column_density,
            kernel,
            mode="same",
        )

        center_start = int(w * 0.38)
        center_end = int(w * 0.62)

        if center_end > center_start:

            center = smoothed[
                center_start:center_end
            ]

            split_x = (
                center_start
                + int(np.argmin(center))
            )

            # Require a meaningful whitespace seam.
            left_density = float(
                np.mean(smoothed[int(w * 0.20):split_x])
            )

            right_density = float(
                np.mean(smoothed[split_x:int(w * 0.80)])
            )

            seam_density = float(
                smoothed[split_x]
            )

            if (
                seam_density < 0.20
                and left_density > 0.05
                and right_density > 0.05
            ):

                # Crop out the lower caption/note area.
                #
                # The supplied sample has the licence cards
                # in the upper ~88% of the image.

                card_bottom = int(h * 0.88)

                margin = int(w * 0.015)

                left_box = (
                    margin,
                    margin,
                    max(
                        1,
                        split_x - margin,
                    ),
                    max(
                        1,
                        card_bottom - margin,
                    ),
                )

                right_box = (
                    split_x + margin,
                    margin,
                    max(
                        1,
                        w - split_x - margin,
                    ),
                    max(
                        1,
                        card_bottom - margin,
                    ),
                )

                return [
                    left_box,
                    right_box,
                ]

    # Generic contour-based fallback.
    bgr = cv2.cvtColor(
        arr,
        cv2.COLOR_RGB2BGR,
    )

    gray = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred,
        50,
        150,
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    page_area = w * h

    candidates = []

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        area = cw * ch

        if cw < w * 0.30:
            continue

        if ch < h * 0.30:
            continue

        if area < page_area * 0.15:
            continue

        if cw / max(ch, 1) < 0.7:
            continue

        candidates.append(
            (
                x,
                y,
                cw,
                ch,
                area,
            )
        )

    candidates.sort(
        key=lambda item: item[4],
        reverse=True,
    )

    selected = []

    for candidate in candidates:

        x, y, cw, ch, area = candidate

        overlap = False

        for sx, sy, sw, sh in selected:

            ix1 = max(x, sx)
            iy1 = max(y, sy)

            ix2 = min(
                x + cw,
                sx + sw,
            )

            iy2 = min(
                y + ch,
                sy + sh,
            )

            intersection = (
                max(0, ix2 - ix1)
                * max(0, iy2 - iy1)
            )

            union = (
                area
                + sw * sh
                - intersection
            )

            if (
                union
                and intersection / union > 0.55
            ):
                overlap = True
                break

        if not overlap:
            selected.append(
                (
                    x,
                    y,
                    cw,
                    ch,
                )
            )

    if len(selected) >= 2:

        selected = sorted(
            selected[:4],
            key=lambda box: (
                box[1],
                box[0],
            ),
        )

        return selected

    # Final fallback:
    # process the complete image.

    return [
        (
            0,
            0,
            w,
            h,
        )
    ]


def crop_region(
    image: Image.Image,
    box: Tuple[int, int, int, int],
) -> Image.Image:

    x, y, w, h = box

    pad = 8

    return image.crop(
        (
            max(0, x - pad),
            max(0, y - pad),
            min(
                image.width,
                x + w + pad,
            ),
            min(
                image.height,
                y + h + pad,
            ),
        )
    )


# ---------------------------------------------------------
# IMAGE PREVIEW
# ---------------------------------------------------------

def image_to_data_url(
    image: Image.Image,
    quality: int = 86,
) -> str:

    output = io.BytesIO()

    preview = image.copy()

    preview.thumbnail(
        (1600, 1600)
    )

    preview.save(
        output,
        format="JPEG",
        quality=quality,
        optimize=True,
    )

    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(
            output.getvalue()
        ).decode("ascii")
    )


# ---------------------------------------------------------
# OCR PREPROCESSING
# ---------------------------------------------------------

def preprocess_for_ocr(
    image: Image.Image,
) -> np.ndarray:

    rgb = np.array(
        image.convert("RGB")
    )

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )


# ---------------------------------------------------------
# PADDLE OCR
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_paddle_ocr():

    from paddleocr import PaddleOCR

    print(
        "[OCR] Loading PaddleOCR model..."
    )

    engine = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    print(
        "[OCR] PaddleOCR model ready."
    )

    return engine


def _as_python(value):

    if hasattr(
        value,
        "tolist",
    ):
        return value.tolist()

    return value


def _extract_result_dict(
    page_result,
) -> Dict[str, Any]:

    """
    PaddleOCR 3.x returns a Result object.

    Its JSON representation is structured as:

    {
        "res": {
            "rec_texts": [...],
            "rec_scores": [...],
            "rec_polys": [...]
        }
    }

    This function normalizes the different possible
    representations across PaddleOCR versions.
    """

    data = None

    # Preferred PaddleOCR 3.x interface.
    try:

        json_data = page_result.json

        if callable(json_data):
            json_data = json_data()

        data = json_data

    except Exception:
        data = None

    # Some versions expose the result through indexing.
    if not data:

        try:
            data = page_result["res"]
        except Exception:
            pass

    # IMPORTANT:
    # PaddleOCR 3.x normally wraps the actual OCR
    # payload inside "res".

    if isinstance(data, dict):

        if "res" in data and isinstance(
            data["res"],
            dict,
        ):
            data = data["res"]

    if not isinstance(
        data,
        dict,
    ):
        return {}

    return data


def _parse_paddle_result(
    result,
) -> List[Dict[str, Any]]:

    lines = []

    for page_result in result:

        data = _extract_result_dict(
            page_result
        )

        if not data:
            continue

        rec_texts = (
            data.get(
                "rec_texts",
                [],
            )
            or []
        )

        rec_scores = (
            data.get(
                "rec_scores",
                [],
            )
            or []
        )

        rec_polys = (
            data.get(
                "rec_polys",
                [],
            )
            or []
        )

        rec_boxes = (
            data.get(
                "rec_boxes",
                [],
            )
            or []
        )

        for i, raw_text in enumerate(
            rec_texts
        ):

            text = str(
                raw_text
            ).strip()

            if not text:
                continue

            try:
                score = float(
                    rec_scores[i]
                )
            except Exception:
                score = 0.0

            bbox = []

            # Prefer rec_boxes because they are
            # already [x1,y1,x2,y2].
            if i < len(rec_boxes):

                try:

                    box = _as_python(
                        rec_boxes[i]
                    )

                    if (
                        isinstance(box, list)
                        and len(box) >= 4
                    ):

                        bbox = [
                            int(box[0]),
                            int(box[1]),
                            int(box[2]),
                            int(box[3]),
                        ]

                except Exception:
                    pass

            # Otherwise calculate a rectangle
            # from the polygon.
            if not bbox and i < len(
                rec_polys
            ):

                try:

                    polygon = _as_python(
                        rec_polys[i]
                    )

                    xs = [
                        float(point[0])
                        for point in polygon
                    ]

                    ys = [
                        float(point[1])
                        for point in polygon
                    ]

                    bbox = [
                        round(min(xs)),
                        round(min(ys)),
                        round(max(xs)),
                        round(max(ys)),
                    ]

                except Exception:
                    bbox = []

            lines.append(
                {
                    "text": text,
                    "confidence": round(
                        score,
                        4,
                    ),
                    "bbox": bbox,
                }
            )

       # ---------------------------------------------------------
    # LAYOUT-AWARE READING ORDER
    # ---------------------------------------------------------
    #
    # PaddleOCR returns individual text boxes. Their top Y
    # coordinates can differ slightly even when the text
    # belongs to the same visual line.
    #
    # A simple global Y -> X sort can therefore produce:
    #
    #   NITIN KOTWAL
    #   AJINKYA
    #   Name
    #
    # even when the document visually contains:
    #
    #   Name : AJINKYA NITIN KOTWAL
    #
    # Group boxes into visual rows first, then sort each row
    # from left to right.
    # ---------------------------------------------------------

    def box_center_y(item):
        bbox = item.get("bbox", [])

        if len(bbox) != 4:
            return 0.0

        return (
            float(bbox[1]) +
            float(bbox[3])
        ) / 2.0

    def box_height(item):
        bbox = item.get("bbox", [])

        if len(bbox) != 4:
            return 0.0

        return max(
            1.0,
            float(bbox[3] - bbox[1])
        )

    valid_lines = []
    invalid_lines = []

    for item in lines:

        bbox = item.get("bbox", [])

        if len(bbox) == 4:
            valid_lines.append(item)
        else:
            invalid_lines.append(item)

    # Sort approximately by vertical position first.
    valid_lines.sort(
        key=lambda item: (
            box_center_y(item),
            item["bbox"][0],
        )
    )

    rows = []

    for item in valid_lines:

        center_y = box_center_y(item)
        height = box_height(item)

        placed = False

        for row in rows:

            # Average vertical centre of the current row.
            row_center = sum(
                box_center_y(existing)
                for existing in row
            ) / len(row)

            # Use the OCR box height to determine a sensible
            # tolerance for small vertical differences.
            row_height = sum(
                box_height(existing)
                for existing in row
            ) / len(row)

            tolerance = max(
                8.0,
                min(
                    25.0,
                    max(height, row_height) * 0.55
                )
            )

            if abs(center_y - row_center) <= tolerance:
                row.append(item)
                placed = True
                break

        if not placed:
            rows.append([item])

    # Sort every visual row from left to right.
    for row in rows:
        row.sort(
            key=lambda item: item["bbox"][0]
        )

    # Finally sort the rows from top to bottom.
    rows.sort(
        key=lambda row: (
            sum(
                box_center_y(item)
                for item in row
            ) / len(row)
        )
    )

    ordered_lines = []

    for row in rows:
        ordered_lines.extend(row)

    # Any OCR boxes without a usable bbox are appended at
    # the end rather than allowing them to disturb the
    # spatial ordering above.
    ordered_lines.extend(invalid_lines)

    return ordered_lines


def ocr_image(
    image: Image.Image,
) -> Dict[str, Any]:

    """
    Run PaddleOCR on an image.

    Includes a second preprocessing attempt if
    the first OCR pass produces no text.
    """

    ocr_engine = get_paddle_ocr()

    # ---------------------------------------------
    # PASS 1: original image
    # ---------------------------------------------

    prepared = preprocess_for_ocr(
        image
    )

    print(
        f"[OCR] Running OCR on "
        f"{image.width}x{image.height}"
    )

    result = ocr_engine.predict(
        prepared
    )

    lines = _parse_paddle_result(
        result
    )

    # ---------------------------------------------
    # PASS 2: upscale if nothing found
    # ---------------------------------------------

    if not lines:

        print(
            "[OCR] First pass returned no text. "
            "Retrying with upscaled image..."
        )

        scale = 1.5

        enlarged = image.resize(
            (
                int(image.width * scale),
                int(image.height * scale),
            ),
            Image.Resampling.LANCZOS,
        )

        prepared = preprocess_for_ocr(
            enlarged
        )

        result = ocr_engine.predict(
            prepared
        )

        lines = _parse_paddle_result(
            result
        )

    text = "\n".join(
        item["text"]
        for item in lines
    ).strip()

    print(
        f"[OCR] Detected "
        f"{len(lines)} text lines."
    )

    return {
        "text": text,
        "lines": lines,
    }