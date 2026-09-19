# Architecture Notes

## Components

### 1. PaddleOCR — visual perception

PaddleOCR is a pretrained AI OCR pipeline. It handles:

- text detection
- text recognition
- confidence scoring
- text bounding boxes

The application does not train an OCR model.

### 2. Ling 3.0 Flash Fin — semantic intelligence

The requested OpenRouter model is used for:

- mapping different licence layouts into one schema
- resolving field labels such as `DL No`, `Driving Licence No.`, `Valid Till`, etc.
- producing a human-friendly document assistant response
- refusing unsupported answers

### 3. Retrieval

OCR text is divided into small chunks. TF-IDF retrieves relevant chunks for
each question. The LLM sees only the retrieved evidence.

### 4. Grounding

The Q&A prompt requires:

- no external knowledge
- no guessing
- `found=false` when evidence is missing
- valid source IDs for every positive answer

The backend independently verifies that the source IDs returned by the LLM
actually belong to the retrieved chunks.

## Why not use the LLM for OCR?

The Ling Fin model is text-oriented. Keeping visual perception in PaddleOCR
makes the pipeline modular:

```text
Image → PaddleOCR → OCR + coordinates → Ling → JSON / Chat
```

If OCR quality needs improvement later, PaddleOCR can be replaced without
changing the extraction or chatbot layers.
