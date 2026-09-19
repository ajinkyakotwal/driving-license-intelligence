from typing import Dict, List, Any, Optional, Tuple
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# FIELD ALIASES
# =========================================================

FIELD_ALIASES = {
    "dl_number": [
        "dl number",
        "dl no",
        "dl no.",
        "driving licence number",
        "driving license number",
        "driving licence no",
        "driving license no",
        "licence number",
        "license number",
        "licence no",
        "license no",
        "licence id",
        "license id",
    ],

    "full_name": [
        "full name",
        "name of holder",
        "holder name",
        "licence holder",
        "license holder",
        "name",
    ],

    "date_of_birth": [
        "date of birth",
        "dob",
        "birth date",
    ],

    "issue_date": [
        "issue date",
        "issued date",
        "date of issue",
        "valid from",
    ],

    "expiry_date": [
        "expiry date",
        "expiration date",
        "expiry",
        "expiration",
        "valid till",
        "valid until",
        "valid upto",
        "valid up to",
    ],

    "address": [
        "address",
        "residential address",
        "permanent address",
    ],

    "vehicle_classes": [
        "vehicle class",
        "vehicle classes",
        "class of vehicle",
        "authorized vehicles",
        "driving class",
    ],

    "issuing_authority": [
        "issuing authority",
        "issued by",
        "issuing office",
        "rto",
        "transport authority",
        "authority",
    ],

    "email": [
        "email",
        "email address",
        "e mail",
        "e-mail",
        "mail id",
        "email id",
    ],
}


# =========================================================
# REGION DETECTION
# =========================================================

def infer_region_from_query(
    query: str,
) -> Optional[int]:

    q = query.lower().strip()

    second_patterns = [
        r"\bsecond\s+(?:driving\s+)?licen[cs]e\b",
        r"\blicen[cs]e\s+(?:number\s+)?2\b",
        r"\b2(?:nd)?\s+(?:driving\s+)?licen[cs]e\b",
        r"\bsecond\s+(?:driving\s+)?document\b",
        r"\bsecond\s+card\b",
    ]

    for pattern in second_patterns:
        if re.search(pattern, q):
            return 2

    first_patterns = [
        r"\bfirst\s+(?:driving\s+)?licen[cs]e\b",
        r"\blicen[cs]e\s+(?:number\s+)?1\b",
        r"\b1(?:st)?\s+(?:driving\s+)?licen[cs]e\b",
        r"\bfirst\s+(?:driving\s+)?document\b",
        r"\bfirst\s+card\b",
    ]

    for pattern in first_patterns:
        if re.search(pattern, q):
            return 1

    return None


# =========================================================
# FIELD DETECTION
# =========================================================

def infer_field_from_query(
    query: str,
) -> Optional[str]:

    q = query.lower().strip()

    q = re.sub(
        r"[^a-z0-9\s]",
        " ",
        q,
    )

    q = re.sub(
        r"\s+",
        " ",
        q,
    ).strip()

    matches = []

    for field, aliases in FIELD_ALIASES.items():

        for alias in aliases:

            alias_normalized = alias.lower().strip()

            if alias_normalized in q:

                matches.append(
                    (
                        len(alias_normalized),
                        field,
                    )
                )

    if not matches:
        return None

    # Longest matching phrase wins.
    matches.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return matches[0][1]


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(
    text: str,
) -> str:

    text = text or ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(
    text: str,
) -> List[str]:

    return re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )


# =========================================================
# LEXICAL SCORE
# =========================================================

def exact_match_score(
    query: str,
    text: str,
) -> float:

    q_norm = normalize_text(
        query
    )

    t_norm = normalize_text(
        text
    )

    if not q_norm or not t_norm:
        return 0.0

    score = 0.0

    # Exact phrase.
    if q_norm in t_norm:
        score += 2.0

    query_tokens = tokenize(
        query
    )

    text_tokens = tokenize(
        text
    )

    if not query_tokens or not text_tokens:
        return score

    text_token_set = set(
        text_tokens
    )

    matched = 0

    for token in query_tokens:

        if token in text_token_set:

            matched += 1

            # Technical abbreviations and short tokens.
            if len(token) <= 5:
                score += 0.35
            else:
                score += 0.10

    coverage = (
        matched / len(query_tokens)
    )

    score += coverage * 0.75

    return score


# =========================================================
# BUILD CHUNKS
# =========================================================

def build_chunks(
    documents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    chunks: List[
        Dict[str, Any]
    ] = []

    for document in documents:

        page = int(
            document.get(
                "page",
                1,
            )
        )

        region = int(
            document.get(
                "region",
                1,
            )
        )

        ocr_text = (
            document.get(
                "ocr_text",
                "",
            )
            or ""
        )

        lines = [
            line.strip()
            for line in ocr_text.splitlines()
            if line.strip()
        ]

        # -------------------------------------------------
        # OCR chunks
        # -------------------------------------------------

        chunk_size = 4

        for start in range(
            0,
            len(lines),
            chunk_size,
        ):

            block = lines[
                start:start + chunk_size
            ]

            if not block:
                continue

            chunk_no = (
                start // chunk_size
            ) + 1

            chunk_id = (
                f"p{page}-r{region}-c{chunk_no}"
            )

            chunks.append({
                "chunk_id": chunk_id,
                "page": page,
                "region": region,
                "text": "\n".join(block),
                "source_type": "ocr",
            })

        # -------------------------------------------------
        # STRUCTURED EXTRACTION
        # -------------------------------------------------

        extraction = document.get(
            "extraction"
        )

        if isinstance(
            extraction,
            dict,
        ):

            for field_name, value in extraction.items():

                if value is None:
                    continue

                if isinstance(
                    value,
                    list,
                ):

                    value = ", ".join(
                        str(item)
                        for item in value
                        if item is not None
                    )

                if not isinstance(
                    value,
                    str,
                ):

                    value = str(
                        value
                    )

                value = value.strip()

                if not value:
                    continue

                readable_name = (
                    field_name
                    .replace(
                        "_",
                        " ",
                    )
                    .title()
                )

                field_text = (
                    f"{readable_name}: "
                    f"{value}"
                )

                field_chunk_id = (
                    f"p{page}-r{region}"
                    f"-field-{field_name}"
                )

                chunks.append({
                    "chunk_id": field_chunk_id,
                    "page": page,
                    "region": region,
                    "text": field_text,
                    "source_type": "extraction",
                    "field_name": field_name,
                })

    return chunks


# =========================================================
# RETRIEVE
# =========================================================

def retrieve(
    documents: List[Dict[str, Any]],
    query: str,
    top_k: int = 5,
    selected_region: Optional[int] = None,
) -> List[Dict[str, Any]]:

    if not query or not query.strip():
        return []

    all_chunks = build_chunks(
        documents
    )

    if not all_chunks:
        return []

    # -----------------------------------------------------
    # REGION
    # -----------------------------------------------------

    inferred_region = (
        infer_region_from_query(
            query
        )
    )

    if inferred_region is not None:
        effective_region = inferred_region
    else:
        effective_region = selected_region

    chunks = all_chunks

    if effective_region is not None:

        chunks = [
            chunk
            for chunk in chunks
            if int(
                chunk["region"]
            ) == int(
                effective_region
            )
        ]

    if not chunks:
        return []

    # -----------------------------------------------------
    # FIELD INTENT
    # -----------------------------------------------------

    requested_field = (
        infer_field_from_query(
            query
        )
    )

    print(
        f"[RAG] Detected field intent: "
        f"{requested_field}"
    )

    print(
        f"[RAG] Effective region: "
        f"{effective_region}"
    )

    # =====================================================
    # EXACT STRUCTURED FIELD RETRIEVAL
    # =====================================================

    if requested_field:

        field_matches = [
            chunk
            for chunk in chunks
            if (
                chunk.get(
                    "source_type"
                ) == "extraction"
                and chunk.get(
                    "field_name"
                ) == requested_field
            )
        ]

        if field_matches:

            print(
                f"[RAG] Exact structured field "
                f"match found: "
                f"{requested_field}"
            )

            results = []

            for chunk in field_matches:

                results.append({
                    **chunk,
                    "score": 1.0,
                    "match_type": (
                        "structured_field"
                    ),
                })

            # -------------------------------------------------
            # IMPORTANT:
            # Return the exact field plus nearby OCR evidence
            # where available.
            # -------------------------------------------------

            exact_result = results[0]

            nearby_ocr = []

            for chunk in chunks:

                if (
                    chunk.get(
                        "source_type"
                    ) == "ocr"
                ):

                    lexical = exact_match_score(
                        query,
                        chunk["text"],
                    )

                    if lexical > 0:
                        nearby_ocr.append(
                            (
                                chunk,
                                lexical,
                            )
                        )

            nearby_ocr.sort(
                key=lambda item: item[1],
                reverse=True,
            )

            final_results = [
                exact_result
            ]

            for chunk, score in nearby_ocr:

                if (
                    chunk["chunk_id"]
                    == exact_result["chunk_id"]
                ):
                    continue

                final_results.append({
                    **chunk,
                    "score": round(
                        float(score),
                        4,
                    ),
                    "match_type": "ocr_exact",
                })

                if len(
                    final_results
                ) >= top_k:
                    break

            return final_results

    # =====================================================
    # TF-IDF FALLBACK
    # =====================================================

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words=None,
            token_pattern=r"(?u)\b\w+\b",
        )

        matrix = vectorizer.fit_transform(
            texts
        )

        query_vector = vectorizer.transform(
            [query]
        )

        tfidf_scores = cosine_similarity(
            query_vector,
            matrix,
        )[0]

    except ValueError:

        tfidf_scores = [
            0.0
            for _ in chunks
        ]

    # =====================================================
    # HYBRID RANKING
    # =====================================================

    ranked: List[
        Tuple[
            Dict[str, Any],
            float
        ]
    ] = []

    for chunk, tfidf_score in zip(
        chunks,
        tfidf_scores,
    ):

        tfidf_score = float(
            tfidf_score
        )

        lexical_score = (
            exact_match_score(
                query,
                chunk["text"],
            )
        )

        lexical_score = min(
            lexical_score,
            2.0,
        ) / 2.0

        final_score = (
            tfidf_score * 0.45
            + lexical_score * 0.55
        )

        if chunk.get(
            "source_type"
        ) == "extraction":

            final_score += 0.10

        ranked.append(
            (
                chunk,
                final_score,
            )
        )

    ranked.sort(
        key=lambda pair: pair[1],
        reverse=True,
    )

    selected = []

    for chunk, score in ranked:

        if score >= 0.02:

            selected.append(
                (
                    chunk,
                    score,
                )
            )

        if len(
            selected
        ) >= top_k:
            break

    if not selected:

        selected = ranked[
            :top_k
        ]

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    results = []

    seen = set()

    for chunk, score in selected:

        key = (
            chunk["page"],
            chunk["region"],
            normalize_text(
                chunk["text"]
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        results.append({
            **chunk,
            "score": round(
                float(score),
                4,
            ),
            "match_type": "hybrid",
        })

        if len(
            results
        ) >= top_k:
            break

    return results