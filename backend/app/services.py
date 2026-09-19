import uuid
from typing import Any, Dict, List

from .ocr import (
    bytes_to_images,
    crop_region,
    find_document_regions,
    image_to_data_url,
    ocr_image,
)

from .llm import extract_licence, answer_question
from .rag import retrieve


# =========================================================
# DOCUMENT ANALYSIS
# =========================================================

def analyze_file(
    data: bytes,
    filename: str,
) -> Dict[str, Any]:

    pages = bytes_to_images(
        data,
        filename,
    )

    if not pages:
        raise ValueError(
            "No readable pages found in the uploaded file."
        )

    all_documents: List[Dict[str, Any]] = []

    first_page_preview = image_to_data_url(
        pages[0]
    )

    for page_number, page_image in enumerate(
        pages,
        start=1,
    ):

        print(
            f"[ANALYZE] Processing page "
            f"{page_number}/{len(pages)}"
        )

        regions = find_document_regions(
            page_image
        )

        print(
            f"[ANALYZE] Detected "
            f"{len(regions)} document region(s)"
        )

        page_documents = []

        # =================================================
        # OCR EACH REGION
        # =================================================

        for region_number, box in enumerate(
            regions,
            start=1,
        ):

            print(
                f"[ANALYZE] OCR page={page_number} "
                f"region={region_number}"
            )

            crop = crop_region(
                page_image,
                box,
            )

            ocr_result = ocr_image(
                crop
            )

            ocr_text = ocr_result.get(
                "text",
                "",
            )

            if ocr_text.strip():

                page_documents.append(
                    {
                        "region": region_number,
                        "box": box,
                        "crop": crop,
                        "ocr_result": ocr_result,
                    }
                )

        # =================================================
        # FALLBACK: FULL PAGE OCR
        # =================================================

        if not page_documents:

            print(
                f"[ANALYZE] Region OCR failed on "
                f"page {page_number}. "
                f"Trying full-page OCR..."
            )

            full_page_result = ocr_image(
                page_image
            )

            full_page_text = full_page_result.get(
                "text",
                "",
            )

            if full_page_text.strip():

                page_documents.append(
                    {
                        "region": 1,
                        "box": (
                            0,
                            0,
                            page_image.width,
                            page_image.height,
                        ),
                        "crop": page_image,
                        "ocr_result": full_page_result,
                    }
                )

            else:

                raise ValueError(
                    f"No readable text was detected "
                    f"on page {page_number}."
                )

        # =================================================
        # LLM STRUCTURED EXTRACTION
        #
        # IMPORTANT:
        # RAG IS NOT USED HERE.
        #
        # OCR -> Ling extraction only.
        # =================================================

        for item in page_documents:

            region_number = item["region"]
            crop = item["crop"]
            ocr_result = item["ocr_result"]

            ocr_text = ocr_result.get(
                "text",
                "",
            )

            print(
                f"[LLM] Extracting structured "
                f"licence data from page={page_number}, "
                f"region={region_number}"
            )

            extraction = extract_licence(
                ocr_text=ocr_text,
                page=page_number,
                region=region_number,
            )

            # -------------------------------------------------
            # DEBUGGING
            #
            # This is useful while we are fixing the pipeline.
            # -------------------------------------------------

            print(
                f"[LLM] Extraction result "
                f"page={page_number}, "
                f"region={region_number}:"
            )

            print(extraction)

            # -------------------------------------------------
            # Store EVERYTHING required by RAG.
            # -------------------------------------------------

            doc = {
                "id": str(
                    uuid.uuid4()
                ),

                "page": page_number,

                "region": region_number,

                "title": (
                    f"Licence "
                    f"{len(all_documents) + 1}"
                ),

                "preview_data_url": (
                    image_to_data_url(
                        crop
                    )
                ),

                # Original OCR text
                "ocr_text": ocr_text,

                # OCR line-level information
                "ocr_lines": ocr_result.get(
                    "lines",
                    [],
                ),

                # Ling structured extraction
                "extraction": extraction,
            }

            all_documents.append(
                doc
            )

    if not all_documents:

        raise ValueError(
            "No document regions could be processed."
        )

    print(
        f"[ANALYZE] Completed. "
        f"Extracted {len(all_documents)} licence(s)."
    )

    # =====================================================
    # DEBUG: SHOW WHAT RAG WILL RECEIVE
    # =====================================================

    for document in all_documents:

        print(
            "\n================ DOCUMENT ================"
        )

        print(
            f"Page   : {document['page']}"
        )

        print(
            f"Region : {document['region']}"
        )

        print(
            f"OCR chars: {len(document['ocr_text'])}"
        )

        print(
            "Extraction:"
        )

        print(
            document["extraction"]
        )

        print(
            "==========================================\n"
        )

    return {
        "document_id": str(
            uuid.uuid4()
        ),

        "filename": filename,

        "pages": len(pages),

        "preview_data_url": first_page_preview,

        "documents": all_documents,
    }


# =========================================================
# UPDATE EDITED DOCUMENTS
# =========================================================

def update_saved_documents(
    payload: Dict[str, Any],
    edited_documents: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:

    by_id = {
        item["id"]: item
        for item in edited_documents
    }

    for original in payload["documents"]:

        if original["id"] in by_id:

            original["extraction"] = by_id[
                original["id"]
            ]["extraction"]

    return payload


# =========================================================
# GROUNDED CHAT
#
# IMPORTANT ARCHITECTURE:
#
# User question
#       ↓
# RAG retrieval
#       ↓
# Relevant document evidence
#       ↓
# Ling LLM reasoning
#       ↓
# Answer
#
# RAG NEVER generates the final answer.
# =========================================================

def grounded_chat(
    payload: Dict[str, Any],
    question: str,
    selected_region: int | None = None,
) -> Dict[str, Any]:

    question = (
        question or ""
    ).strip()

    if not question:

        return {
            "answer": "Please enter a question.",
            "found": False,
            "sources": [],
        }

    print(
        "\n================================================"
    )

    print(
        f"[CHAT] Question: {question}"
    )

    print(
        f"[CHAT] Selected region: "
        f"{selected_region}"
    )

    # =====================================================
    # STEP 1
    # RAG RETRIEVAL
    # =====================================================

    candidates = retrieve(
        payload["documents"],
        query=question,
        top_k=5,
        selected_region=selected_region,
    )

    print(
        f"[RAG] Retrieved "
        f"{len(candidates)} candidate(s)"
    )

    # -----------------------------------------------------
    # DEBUG: SHOW EXACT EVIDENCE SENT TO LLM
    # -----------------------------------------------------

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"\n[RAG] Candidate {index}"
        )

        print(
            f"  Chunk   : {candidate['chunk_id']}"
        )

        print(
            f"  Page    : {candidate['page']}"
        )

        print(
            f"  Region  : {candidate['region']}"
        )

        print(
            f"  Type    : "
            f"{candidate.get('source_type', 'unknown')}"
        )

        print(
            f"  Score   : "
            f"{candidate.get('score', 0)}"
        )

        print(
            "  Evidence:"
        )

        print(
            candidate["text"]
        )

    # =====================================================
    # NO EVIDENCE
    # =====================================================

    if not candidates:

        print(
            "[RAG] No evidence found."
        )

        print(
            "================================================\n"
        )

        return {
            "answer": (
                "I couldn't find that information "
                "in the uploaded document."
            ),
            "found": False,
            "sources": [],
        }

    # =====================================================
    # STEP 2
    # BUILD CONTEXT FOR LING
    # =====================================================

    context_parts = []

    for candidate in candidates:

        context_parts.append(
            f"[{candidate['chunk_id']}] "
            f"Page {candidate['page']}, "
            f"Region {candidate['region']}\n"
            f"{candidate['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    print(
        "\n[LLM] Sending retrieved evidence to Ling..."
    )

    print(
        "[LLM] Context:"
    )

    print(
        context
    )

    # =====================================================
    # STEP 3
    # LING REASONS OVER THE EVIDENCE
    # =====================================================

    model_result = answer_question(
        question=question,
        context=context,
    )

    print(
        "\n[LLM] Answer:"
    )

    print(
        model_result.get(
            "answer",
            "",
        )
    )

    print(
        "[LLM] Found:"
    )

    print(
        model_result.get(
            "found",
            False,
        )
    )

    print(
        "[LLM] Source IDs:"
    )

    print(
        model_result.get(
            "source_ids",
            [],
        )
    )

    # =====================================================
    # STEP 4
    # VALIDATE SOURCES
    #
    # Ling is allowed to reason, but it may only cite
    # evidence that was actually retrieved by RAG.
    # =====================================================

    valid_ids = {
        candidate["chunk_id"]
        for candidate in candidates
    }

    model_source_ids = model_result.get(
        "source_ids",
        [],
    )

    if not isinstance(
        model_source_ids,
        list,
    ):
        model_source_ids = []

    valid_source_ids = [
        source_id
        for source_id in model_source_ids
        if source_id in valid_ids
    ]

    # -----------------------------------------------------
    # If Ling didn't return source IDs, but it did return
    # an answer, use the retrieved evidence rather than
    # throwing away the answer.
    #
    # The answer is still grounded because the LLM was only
    # given retrieved document evidence.
    # -----------------------------------------------------

    if not valid_source_ids:

        if model_result.get(
            "found",
            False,
        ):

            valid_source_ids = [
                candidate["chunk_id"]
                for candidate in candidates
            ]

    # =====================================================
    # FINAL FOUND DECISION
    # =====================================================

    found = bool(
        model_result.get(
            "found",
            False,
        )
        and valid_source_ids
    )

    if not found:

        print(
            "[CHAT] LLM determined that the "
            "information is not supported."
        )

        print(
            "================================================\n"
        )

        return {
            "answer": (
                "I couldn't find that information "
                "in the uploaded document."
            ),
            "found": False,
            "sources": [],
        }

    # =====================================================
    # STEP 5
    # BUILD SOURCES
    # =====================================================

    sources = [
        {
            "page": candidate["page"],
            "region": candidate["region"],
            "chunk_id": candidate["chunk_id"],
            "text": candidate["text"],
        }
        for candidate in candidates
        if candidate["chunk_id"]
        in valid_source_ids
    ]

    print(
        f"[CHAT] Returning answer with "
        f"{len(sources)} source(s)."
    )

    print(
        "================================================\n"
    )

    return {
        "answer": model_result.get(
            "answer",
            "",
        ),

        "found": True,

        "sources": sources,
    }