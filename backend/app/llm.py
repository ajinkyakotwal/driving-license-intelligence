import json
import re
import time
from typing import Any, Dict

import requests

from .config import get_settings


EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_driving_licence",
        "description": (
            "Extract only information explicitly present "
            "in a driving licence OCR transcript."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {"type": ["string", "null"]},
                "licence_number": {"type": ["string", "null"]},
                "date_of_birth": {"type": ["string", "null"]},
                "date_of_issue": {"type": ["string", "null"]},
                "date_of_expiry": {"type": ["string", "null"]},
                "address": {"type": ["string", "null"]},
                "vehicle_classes": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "issuing_authority": {"type": ["string", "null"]},
                "blood_group": {"type": ["string", "null"]},
                "father_spouse_name": {"type": ["string", "null"]},
                "restrictions": {"type": ["string", "null"]},
                "other_information": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "field_evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "quote": {"type": "string"},
                        },
                        "required": ["field", "quote"],
                    },
                },
            },
            "required": [
                "full_name",
                "licence_number",
                "date_of_birth",
                "date_of_issue",
                "date_of_expiry",
                "address",
                "vehicle_classes",
                "issuing_authority",
                "blood_group",
                "father_spouse_name",
                "restrictions",
                "other_information",
                "field_evidence",
            ],
        },
    },
}


def _headers() -> Dict[str, str]:
    settings = get_settings()

    return {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": settings.app_name,
    }


def _post(
    messages,
    tools=None,
    tool_choice=None,
    temperature=0.0,
    max_tokens=1800,
):
    settings = get_settings()

    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    payload: Dict[str, Any] = {
        "model": settings.openrouter_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if tools:
        payload["tools"] = tools

    if tool_choice:
        payload["tool_choice"] = tool_choice

    url = (
        f"{settings.openrouter_base_url.rstrip('/')}"
        "/chat/completions"
    )

    last_error = None

    for attempt in range(3):
        try:
            print(
                f"[LLM] Calling OpenRouter "
                f"(attempt {attempt + 1}/3)..."
            )

            response = requests.post(
                url,
                headers=_headers(),
                json=payload,
                timeout=120,
            )

            if response.status_code < 400:
                return response.json()

            error_text = response.text[:2000]

            print(
                f"[LLM] OpenRouter returned "
                f"{response.status_code}: "
                f"{error_text}"
            )

            last_error = (
                f"OpenRouter error "
                f"{response.status_code}: "
                f"{error_text}"
            )

            if response.status_code in (
                500,
                502,
                503,
                504,
            ) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue

            raise RuntimeError(last_error)

        except requests.RequestException as exc:
            last_error = (
                f"OpenRouter request failed: {exc}"
            )

            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue

            raise RuntimeError(last_error) from exc

    raise RuntimeError(
        last_error or "OpenRouter request failed."
    )


def _parse_json_from_text(
    text: str,
) -> Dict[str, Any]:

    if not text:
        raise ValueError(
            "Model returned an empty response."
        )

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        try:
            result = json.loads(
                text[start:end + 1]
            )

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Model did not return valid JSON."
    )


def extract_licence(
    ocr_text: str,
    page: int,
    region: int,
) -> Dict[str, Any]:

    prompt = f"""
Extract structured information from this ONE
driving licence OCR transcript.

IMPORTANT:
- Use ONLY the OCR transcript.
- Never guess.
- Never infer missing information.
- Never use outside knowledge.
- If a value is unreadable or absent, use null.
- Preserve licence numbers exactly.
- Preserve dates exactly as shown.
- Extract every vehicle class explicitly shown.
- Preserve the address accurately.
- Evidence quotes must be VERBATIM text from the OCR.
- Do not provide an explanation.
- Immediately call the extraction function.

Page: {page}
Region: {region}

OCR:
----------------
{ocr_text}
----------------
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a high-precision document "
                "extraction engine. "
                "Do not explain your reasoning. "
                "Immediately use the provided extraction "
                "function. Never invent information."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    settings = get_settings()

    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured."
        )

    payload = {
        "model": settings.openrouter_model,
        "messages": messages,
        "tools": [EXTRACTION_TOOL],
        "tool_choice": {
            "type": "function",
            "function": {
                "name": "extract_driving_licence"
            },
        },
        "parallel_tool_calls": False,
        "temperature": 0,
        "max_tokens": 8000,
        "reasoning": {
            "effort": "low"
        },
    }

    url = (
        f"{settings.openrouter_base_url.rstrip('/')}"
        "/chat/completions"
    )

    print(
        "[LLM] Sending licence extraction request..."
    )

    response = requests.post(
        url,
        headers=_headers(),
        json=payload,
        timeout=120,
    )

    if response.status_code >= 400:
        print(
            "[LLM] OpenRouter extraction error:"
        )
        print(
            response.text[:3000]
        )

        raise RuntimeError(
            f"OpenRouter error "
            f"{response.status_code}: "
            f"{response.text[:1000]}"
        )

    data = response.json()

    message = data["choices"][0]["message"]

    print(
        "[LLM] OpenRouter extraction response received."
    )

    tool_calls = message.get("tool_calls")

    if not tool_calls:
        print(
            "[LLM] No tool call returned."
        )
        print(
            json.dumps(
                message,
                indent=2,
                ensure_ascii=False,
            )[:5000]
        )

        raise RuntimeError(
            "Ling did not return the required "
            "structured extraction."
        )

    raw_args = (
        tool_calls[0]
        .get("function", {})
        .get("arguments")
    )

    if not raw_args:
        raise RuntimeError(
            "Ling returned an empty extraction."
        )

    try:
        result = (
            json.loads(raw_args)
            if isinstance(raw_args, str)
            else raw_args
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Ling returned invalid extraction JSON."
        ) from exc

    # -------------------------------------------------
    # EVIDENCE
    # -------------------------------------------------

    evidence = []

    for item in (
        result.get("field_evidence", [])
        or []
    ):
        if not isinstance(item, dict):
            continue

        field = str(
            item.get("field", "")
        ).strip()

        quote = str(
            item.get("quote", "")
        ).strip()

        if not field or not quote:
            continue

        evidence.append(
            {
                "field": field,
                "quote": quote,
                "page": page,
                "region": region,
            }
        )

    result["field_evidence"] = evidence

    return normalize_extraction(result)


def normalize_extraction(
    value: Dict[str, Any],
) -> Dict[str, Any]:

    fields = [
        "full_name",
        "licence_number",
        "date_of_birth",
        "date_of_issue",
        "date_of_expiry",
        "address",
        "issuing_authority",
        "blood_group",
        "father_spouse_name",
        "restrictions",
    ]

    result = {}

    for field in fields:
        item = value.get(field)

        if item is None:
            result[field] = None
        else:
            result[field] = (
                str(item).strip() or None
            )

    result["vehicle_classes"] = [
        str(x).strip()
        for x in (
            value.get(
                "vehicle_classes",
                [],
            )
            or []
        )
        if str(x).strip()
    ]

    result["other_information"] = [
        str(x).strip()
        for x in (
            value.get(
                "other_information",
                [],
            )
            or []
        )
        if str(x).strip()
    ]

    result["field_evidence"] = (
        value.get("field_evidence")
        or []
    )

    return result


def answer_question(
    question: str,
    context: str,
) -> Dict[str, Any]:

    prompt = f"""
Answer the user's question using ONLY the
retrieved OCR evidence below.

STRICT RULES:

- Do not use outside knowledge.
- Do not guess.
- Do not infer unsupported information.
- Every factual statement must be supported
  by the supplied evidence.
- If the evidence does not answer the question,
  found must be false.
- If not found, use exactly:
  "I couldn't find that information in the uploaded document."
- Keep the answer concise.
- source_ids must reference only IDs present
  in the supplied evidence.
- Return ONLY valid JSON.

QUESTION:
{question}

RETRIEVED EVIDENCE:
{context}

Return:

{{
  "answer": "string",
  "found": true,
  "source_ids": ["p1-r1-c1"]
}}
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a grounded document QA "
                "assistant. The supplied evidence is "
                "the only source of truth."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    data = _post(
        messages,
        temperature=0,
        max_tokens=700,
    )

    message = data["choices"][0]["message"]

    content = (
        message.get("content")
        or ""
    )

    if not content:
        raise RuntimeError(
            "Ling returned an empty chatbot response."
        )

    result = _parse_json_from_text(
        content
    )

    source_ids = result.get(
        "source_ids",
        [],
    )

    if not isinstance(
        source_ids,
        list,
    ):
        source_ids = []

    return {
        "answer": str(
            result.get(
                "answer",
                "I couldn't find that information "
                "in the uploaded document.",
            )
        ),
        "found": bool(
            result.get("found")
        ),
        "source_ids": [
            str(x)
            for x in source_ids
        ],
    }