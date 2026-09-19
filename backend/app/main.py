from pathlib import Path
from typing import Any, Dict

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .ocr import validate_upload
from .schemas import (
    AnalyzeResponse,
    SaveRequest,
    ChatRequest,
    ChatResponse,
)
from .services import (
    analyze_file,
    update_saved_documents,
    grounded_chat,
)
from .store import store
from .pdf_service import generate_licence_pdf


settings = get_settings()


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "AI-powered driving licence "
        "document intelligence demo."
    ),
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin,
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "model": settings.openrouter_model,
    }


# =========================================================
# ANALYZE DOCUMENT
# =========================================================

@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
)
async def analyze(
    file: UploadFile = File(...),
):

    try:

        data = await file.read()

        validate_upload(
            filename=file.filename or "upload",
            content_type=file.content_type,
            size=len(data),
        )

        result = analyze_file(
            data,
            file.filename or "upload",
        )

        store.put(
            result["document_id"],
            result,
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Document processing failed: "
                f"{str(exc)[:300]}"
            ),
        )


# =========================================================
# SAVE EDITED EXTRACTION
# =========================================================

@app.post(
    "/api/documents/{document_id}/save"
)
def save(
    document_id: str,
    request: SaveRequest,
):

    existing = store.get(
        document_id
    )

    if not existing:

        raise HTTPException(
            status_code=404,
            detail=(
                "Document session not found."
            ),
        )

    updated = update_saved_documents(
        existing,
        [
            item.model_dump()
            for item in request.documents
        ],
    )

    store.update(
        document_id,
        updated,
    )

    return {
        "status": "saved",
        "document_id": document_id,
    }


# =========================================================
# GROUNDED CHAT
# =========================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    existing = store.get(
        request.document_id
    )

    if not existing:

        raise HTTPException(
            status_code=404,
            detail=(
                "Document session not found."
            ),
        )

    try:

        return grounded_chat(
            existing,
            question=request.question,
            selected_region=(
                request.selected_region
            ),
        )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Question processing failed: "
                f"{str(exc)[:300]}"
            ),
        )


# =========================================================
# EXPORT PDF
# =========================================================

@app.post(
    "/api/export-pdf"
)
def export_pdf(
    payload: Dict[str, Any],
):

    try:

        print(
            "[PDF] Generating "
            "driving licence report..."
        )

        pdf_bytes = generate_licence_pdf(
            payload
        )

        print(
            f"[PDF] Generated "
            f"{len(pdf_bytes)} bytes"
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    'attachment; '
                    'filename='
                    '"driving-licence-report.pdf"'
                )
            },
        )

    except Exception as exc:

        print(
            f"[PDF] Generation failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF generation failed: "
                f"{str(exc)[:300]}"
            ),
        )


# =========================================================
# SERVE REACT FRONTEND
# =========================================================

frontend_dist = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "dist"
)


if frontend_dist.exists():

    app.mount(
        "/assets",
        StaticFiles(
            directory=str(
                frontend_dist / "assets"
            )
        ),
        name="assets",
    )

    @app.get("/{path:path}")
    def serve_frontend(
        path: str,
    ):

        requested = (
            frontend_dist / path
        )

        if (
            path
            and requested.exists()
            and requested.is_file()
        ):

            return FileResponse(
                requested
            )

        return FileResponse(
            frontend_dist / "index.html"
        )
    