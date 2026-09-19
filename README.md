# Driving Licence Intelligence

A full-stack AI document-intelligence application for the assessment assignment.

## What it demonstrates

- Driving licence upload: PNG/JPG/WEBP/PDF
- OCR/document processing
- Automatic structured extraction
- Editable/reviewable form
- Document + extracted data side-by-side
- Grounded document Q&A using retrieval-augmented generation (RAG)
- Source/page/region references
- Missing-information handling to reduce hallucination
- Basic file validation and API-key protection
- Docker deployment
- Clean separation between OCR, extraction, retrieval and UI

## Important model decision

The requested model is:

`inclusionai/ling-3.0-flash-fin:free`

At the time this project was prepared, OpenRouter lists this model as a text model, not an image-input model. Therefore the application uses **Tesseract OCR** for the image/PDF-to-text step and uses the requested Ling model for:

1. structured information extraction
2. document question answering

This is intentional. If the interviewer specifically wants an LLM to directly inspect pixels, OpenRouter also has a free visual model such as `inclusionai/ling-3.0-flash-vl:free`; that can be substituted in the OCR stage.

## Architecture

```text
                    ┌──────────────────────────────┐
                    │          React UI             │
                    │ Upload / Preview / Form / QA │
                    └──────────────┬───────────────┘
                                   │ REST
                                   ▼
                    ┌──────────────────────────────┐
                    │        FastAPI Backend        │
                    └──────────────┬───────────────┘
                                   │
                  ┌────────────────┼─────────────────┐
                  ▼                ▼                 ▼
             File Parser        OCR              RAG/Q&A
           PDF/Image → pages   Tesseract        TF-IDF retrieval
                  │                │                 │
                  └────────────┬───┘                 │
                               ▼                     │
                     Ling 3.0 Flash Fin             │
                       structured extraction         │
                               │                     │
                               └──────────────┬──────┘
                                              ▼
                                      grounded answer
```

## Project structure

```text
driving-license-intelligence/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── store.py
│   │   ├── ocr.py
│   │   ├── llm.py
│   │   ├── rag.py
│   │   └── services.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── sample-data/
│   └── Fictional_Sample_Driving_License.png
├── .env.example
├── .gitignore
├── docker-compose.yml
└── README.md
```

## Local setup

### 1. Prerequisites

- Python 3.11+
- Node.js 20+
- Tesseract OCR
- OpenRouter API key

For Windows, install Tesseract and make sure `tesseract.exe` is available on PATH.

For Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

### 2. Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` in the project root and add:

```env
OPENROUTER_API_KEY=your_key_here
```

Start the backend:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

`http://localhost:5173`

## Docker setup

The Docker image installs Tesseract automatically.

```bash
docker compose up --build
```

Open:

`http://localhost:8000`

Set the OpenRouter key in `.env` before starting Docker.

## API endpoints

### `POST /api/analyze`

Multipart upload:

```text
file=<image/pdf>
```

Returns OCR, extracted fields, source metadata and preview data.

### `POST /api/documents/{document_id}/save`

Persists edited fields in the in-memory document session.

### `POST /api/chat`

Request:

```json
{
  "document_id": "uuid",
  "question": "What is the licence expiry date?"
}
```

Response:

```json
{
  "answer": "The licence expires on 15-06-2034.",
  "found": true,
  "sources": [
    {
      "page": 1,
      "region": 1,
      "chunk_id": "p1-r1-c2"
    }
  ]
}
```

## RAG design

The OCR output is converted into chunks containing:

- page
- region
- chunk id
- text

At query time:

1. TF-IDF retrieves the most relevant OCR chunks.
2. A minimum relevance threshold prevents obviously unrelated context from being sent.
3. The retrieved chunks are passed to Ling.
4. Ling is instructed to answer only from those chunks.
5. It must return `NOT_FOUND` when evidence is absent.
6. Source IDs are validated against the chunks actually retrieved.

This is deliberately lightweight for a 48-hour take-home assignment. A production version could replace TF-IDF with embeddings/vector search while retaining the same retrieval interface.

## Security considerations

- API key is server-side only.
- File size is limited.
- Only expected image/PDF MIME types are accepted.
- Uploaded files are never executed.
- The demo uses ephemeral in-memory storage.
- User-entered questions are bounded in size.
- The LLM is explicitly instructed not to invent missing information.

For production, add authentication, malware scanning, encrypted object storage, audit logging, PII retention/deletion policies and persistent vector storage.

## AI development tools used

Suggested disclosure:

- ChatGPT — architecture, implementation assistance, debugging and documentation
- OpenRouter — model gateway
- inclusionAI Ling 3.0 Flash Fin — extraction and grounded Q&A
- Tesseract OCR — document text extraction

Only list tools you actually used in the final submission.

## Demo script

1. Open the application.
2. Upload the supplied sample licence image.
3. Explain that the image contains two licence cards and the backend detects separate document regions.
4. Show OCR/extraction progress.
5. Show the two extracted licences.
6. Edit one field.
7. Click Save.
8. Ask:
   - "What is the licence number?"
   - "What is the expiry date?"
   - "What vehicles is this person authorised to drive?"
9. Ask a question whose answer is not present and show the `Information not found` behavior.
10. Point out source page/region references.
11. Briefly explain OCR → extraction → RAG → answer architecture.
