import { useMemo, useState } from "react";

const API = "/api";

const FIELD_CONFIG = [
  ["full_name", "Full Name"],
  ["licence_number", "Driving Licence Number"],
  ["date_of_birth", "Date of Birth"],
  ["date_of_issue", "Date of Issue"],
  ["date_of_expiry", "Date of Expiry"],
  ["address", "Address"],
  ["vehicle_classes", "Vehicle / Class of Licence"],
  ["issuing_authority", "Issuing Authority"],
  ["blood_group", "Blood Group"],
  ["father_spouse_name", "Father / Spouse Name"],
  ["restrictions", "Restrictions"]
];

function App() {
  const [result, setResult] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [exporting, setExporting] = useState(false);

  // Tracks whether the current edited data has already been saved.
  const [reviewSaved, setReviewSaved] = useState(false);

  const [error, setError] = useState("");

  const [question, setQuestion] = useState("");
  const [chat, setChat] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);

  const selected =
    result?.documents?.[selectedIndex] ?? null;

  const previewLabel = useMemo(() => {
    if (!result) return "No document uploaded";

    return result.documents.length > 1
      ? `${result.documents.length} licence regions detected`
      : "1 licence region detected";
  }, [result]);

  // ============================================================
  // UPLOAD DOCUMENT
  // ============================================================

  async function uploadFile(file) {
    if (!file) return;

    setLoading(true);
    setError("");
    setChat([]);
    setResult(null);
    setReviewSaved(false);

    try {
      const body = new FormData();
      body.append("file", file);

      const response = await fetch(`${API}/analyze`, {
        method: "POST",
        body
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Upload failed."
        );
      }

      setResult(data);
      setSelectedIndex(0);
      setReviewSaved(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  // ============================================================
  // EDIT EXTRACTED FIELD
  // ============================================================

  function updateField(field, value) {
    // Any edit means the current review is no longer saved.
    setReviewSaved(false);

    setResult((current) => {
      if (!current) return current;

      const documents = current.documents.map(
        (doc, index) => {
          if (index !== selectedIndex) {
            return doc;
          }

          const extraction = {
            ...doc.extraction
          };

          if (field === "vehicle_classes") {
            extraction[field] = value
              .split(",")
              .map((item) => item.trim())
              .filter(Boolean);
          } else {
            extraction[field] = value;
          }

          return {
            ...doc,
            extraction
          };
        }
      );

      return {
        ...current,
        documents
      };
    });
  }

  // ============================================================
  // SAVE REVIEW + AUTOMATICALLY DOWNLOAD PDF
  // ============================================================

  async function saveChanges() {
    if (!result || saving || exporting) {
      return;
    }

    setSaving(true);
    setError("");

    try {
      // --------------------------------------------------------
      // STEP 1: Save reviewed data
      // --------------------------------------------------------

      const saveResponse = await fetch(
        `${API}/documents/${result.document_id}/save`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            documents: result.documents
          })
        }
      );

      const saveData = await saveResponse.json();

      if (!saveResponse.ok) {
        throw new Error(
          saveData.detail || "Save failed."
        );
      }

      // --------------------------------------------------------
      // STEP 2: Generate PDF from reviewed data
      // --------------------------------------------------------

      setExporting(true);

      const pdfResponse = await fetch(
        `${API}/export-pdf`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(result)
        }
      );

      if (!pdfResponse.ok) {
        let message = "PDF generation failed.";

        try {
          const data = await pdfResponse.json();
          message = data.detail || message;
        } catch {
          // Keep default message.
        }

        throw new Error(message);
      }

      // --------------------------------------------------------
      // STEP 3: Download PDF
      // --------------------------------------------------------

      const blob = await pdfResponse.blob();

      const url =
        window.URL.createObjectURL(blob);

      const link =
        document.createElement("a");

      link.href = url;
      link.download =
        "driving-licence-reviewed-report.pdf";

      document.body.appendChild(link);
      link.click();
      link.remove();

      window.URL.revokeObjectURL(url);

      // --------------------------------------------------------
      // STEP 4: Mark review as saved
      // --------------------------------------------------------

      setReviewSaved(true);

      setChat((items) => [
        ...items,
        {
          role: "system",
          text:
            "Review saved and PDF downloaded successfully."
        }
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
      setExporting(false);
    }
  }

  // ============================================================
  // DOWNLOAD PDF AGAIN AFTER REVIEW
  // ============================================================

  async function downloadReviewedPdf() {
    if (!result || exporting) {
      return;
    }

    setExporting(true);
    setError("");

    try {
      const response = await fetch(
        `${API}/export-pdf`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(result)
        }
      );

      if (!response.ok) {
        let message = "PDF generation failed.";

        try {
          const data = await response.json();
          message = data.detail || message;
        } catch {
          // Keep default message.
        }

        throw new Error(message);
      }

      const blob = await response.blob();

      const url =
        window.URL.createObjectURL(blob);

      const link =
        document.createElement("a");

      link.href = url;
      link.download =
        "driving-licence-reviewed-report.pdf";

      document.body.appendChild(link);
      link.click();
      link.remove();

      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setExporting(false);
    }
  }

  // ============================================================
  // CHAT / DOCUMENT Q&A
  // ============================================================

  async function askQuestion(
    event,
    preset = null
  ) {
    event?.preventDefault();

    const q =
      (preset ?? question).trim();

    if (!q || !result || chatLoading) {
      return;
    }

    setChat((items) => [
      ...items,
      {
        role: "user",
        text: q
      }
    ]);

    setQuestion("");
    setChatLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API}/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            document_id:
              result.document_id,
            question: q,
            selected_region:
              selected?.region ?? null
          })
        }
      );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Question failed."
        );
      }

      setChat((items) => [
        ...items,
        {
          role: "assistant",
          text: data.answer,
          sources: data.sources
        }
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setChatLoading(false);
    }
  }

  // ============================================================
  // UI
  // ============================================================

  return (
    <div className="app-shell">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="topbar">
        <div>
          <div className="eyebrow">
            AI DOCUMENT INTELLIGENCE
          </div>

          <h1>
            Driving Licence Intelligence
          </h1>

          <p>
            OCR → structured extraction →
            human review → grounded document Q&A
          </p>
        </div>

        <div className="model-badge">
          Ling 3.0 Flash Fin · OpenRouter
        </div>
      </header>

      <main className="container">

        {/* ====================================================
            UPLOAD
        ==================================================== */}

        <section className="hero-card">
          <div>
            <h2>
              Upload a driving licence
            </h2>

            <p>
              Upload PNG, JPG, WEBP or PDF.
              The application detects document
              regions, extracts fields and makes
              them editable.
            </p>
          </div>

          <label className="upload-button">
            {loading
              ? "Processing…"
              : "Choose document"}

            <input
              type="file"
              accept=".png,.jpg,.jpeg,.webp,.pdf"
              onChange={(e) =>
                uploadFile(
                  e.target.files?.[0]
                )
              }
              disabled={loading}
            />
          </label>
        </section>

        {/* ====================================================
            ERROR
        ==================================================== */}

        {error && (
          <div className="error-banner">
            {error}
          </div>
        )}

        {/* ====================================================
            EMPTY STATE
        ==================================================== */}

        {!result && !loading && (
          <section className="empty-state">
            <div className="empty-icon">
              DL
            </div>

            <h2>
              Ready for document analysis
            </h2>

            <p>
              Try the supplied fictional
              sample image. It contains two
              licence cards and demonstrates
              multi-region extraction.
            </p>
          </section>
        )}

        {/* ====================================================
            PROCESSING
        ==================================================== */}

        {loading && (
          <section className="processing-card">
            <div className="spinner" />

            <div>
              <strong>
                Analysing document
              </strong>

              <p>
                Detecting regions, running OCR
                and extracting structured fields…
              </p>
            </div>
          </section>
        )}

        {/* ====================================================
            MAIN WORKSPACE
        ==================================================== */}

        {result && selected && (
          <>
            <section className="workspace-grid">

              {/* ==================================================
                  DOCUMENT PANEL
              ================================================== */}

              <div className="panel document-panel">

                <div className="panel-header">
                  <div>
                    <span className="section-kicker">
                      DOCUMENT
                    </span>

                    <h2>
                      {result.filename}
                    </h2>
                  </div>

                  <span className="status-pill">
                    {previewLabel}
                  </span>
                </div>

                {/* DOCUMENT TABS */}

                <div className="document-tabs">
                  {result.documents.map(
                    (doc, index) => (
                      <button
                        key={doc.id}
                        className={
                          index === selectedIndex
                            ? "tab active"
                            : "tab"
                        }
                        onClick={() =>
                          setSelectedIndex(index)
                        }
                      >
                        {doc.title}

                        <span>
                          Page {doc.page}
                        </span>
                      </button>
                    )
                  )}
                </div>

                {/* DOCUMENT PREVIEW */}

                <div className="preview-wrap">
                  <img
                    src={
                      selected.preview_data_url
                    }
                    alt={`Licence region ${selected.region}`}
                  />
                </div>

                {/* OCR */}

                <details className="ocr-details">
                  <summary>
                    View AI OCR transcript &
                    confidence
                  </summary>

                  <pre>
                    {selected.ocr_text}
                  </pre>

                  <div className="ocr-lines">
                    {(selected.ocr_lines || [])
                      .map((line, i) => (
                        <div
                          className="ocr-line"
                          key={i}
                        >
                          <span>
                            {line.text}
                          </span>

                          <small>
                            {Math.round(
                              (line.confidence ||
                                0) * 100
                            )}
                            % OCR · bbox [
                            {Array.isArray(
                              line.bbox
                            )
                              ? line.bbox.join(
                                  ", "
                                )
                              : "n/a"}
                            ]
                          </small>
                        </div>
                      ))}
                  </div>
                </details>
              </div>

              {/* ==================================================
                  EXTRACTED DATA PANEL
              ================================================== */}

              <div className="panel form-panel">

                <div className="panel-header">
                  <div>
                    <span className="section-kicker">
                      EXTRACTED DATA
                    </span>

                    <h2>
                      Review & edit
                    </h2>
                  </div>

                  <span className="verified-badge">
                    {reviewSaved
                      ? "Review saved"
                      : "AI extracted"}
                  </span>
                </div>

                {/* FIELDS */}

                <div className="field-grid">

                  {FIELD_CONFIG.map(
                    ([key, label]) => {
                      const value =
                        selected.extraction[
                          key
                        ];

                      const isLong =
                        key === "address" ||
                        key === "restrictions";

                      return (
                        <label
                          className={
                            isLong
                              ? "field field-wide"
                              : "field"
                          }
                          key={key}
                        >
                          <span>
                            {label}
                          </span>

                          {isLong ? (
                            <textarea
                              value={
                                value ?? ""
                              }
                              placeholder="Not available"
                              onChange={(e) =>
                                updateField(
                                  key,
                                  e.target.value
                                )
                              }
                              rows={
                                key ===
                                "address"
                                  ? 3
                                  : 2
                              }
                            />
                          ) : (
                            <input
                              value={
                                Array.isArray(
                                  value
                                )
                                  ? value.join(
                                      ", "
                                    )
                                  : value ?? ""
                              }
                              placeholder="Not available"
                              onChange={(e) =>
                                updateField(
                                  key,
                                  e.target.value
                                )
                              }
                            />
                          )}
                        </label>
                      );
                    }
                  )}

                </div>

                {/* OTHER INFORMATION */}

                {selected.extraction
                  .other_information
                  ?.length > 0 && (
                  <div className="other-info">

                    <h3>
                      Other information
                    </h3>

                    <ul>
                      {selected.extraction
                        .other_information
                        .map(
                          (item, i) => (
                            <li key={i}>
                              {item}
                            </li>
                          )
                        )}
                    </ul>

                  </div>
                )}

                {/* ==================================================
                    REVIEW / DOWNLOAD BUTTON
                ================================================== */}

                <div className="action-buttons">

                  <button
                    className="primary-button"
                    onClick={
                      reviewSaved
                        ? downloadReviewedPdf
                        : saveChanges
                    }
                    disabled={
                      saving ||
                      exporting
                    }
                  >
                    {saving
                      ? "Saving review…"
                      : exporting
                      ? "Generating PDF…"
                      : reviewSaved
                      ? "Download after review"
                      : "Save reviewed data"}
                  </button>

                </div>

                {/* ==================================================
                    EXTRACTION EVIDENCE
                ================================================== */}

                <div className="evidence-box">

                  <h3>
                    Extraction evidence
                  </h3>

                  {selected.extraction
                    .field_evidence
                    ?.length ? (
                    selected.extraction
                      .field_evidence
                      .map(
                        (evidence, i) => (
                          <div
                            className="evidence-row"
                            key={i}
                          >
                            <strong>
                              {
                                evidence.field
                              }
                            </strong>

                            <span>
                              “
                              {
                                evidence.quote
                              }
                              ”
                            </span>

                            <small>
                              Page{" "}
                              {
                                evidence.page
                              }{" "}
                              · Region{" "}
                              {
                                evidence.region
                              }
                            </small>
                          </div>
                        )
                      )
                  ) : (
                    <p>
                      No evidence snippets
                      returned.
                    </p>
                  )}

                </div>

              </div>
            </section>

            {/* ==================================================
                CHAT / RAG
            ================================================== */}

            <section className="panel chat-panel">

              <div className="panel-header">

                <div>
                  <span className="section-kicker">
                    RAG + LLM
                  </span>

                  <h2>
                    Ask the document
                  </h2>
                </div>

                <span className="grounding-badge">
                  Grounded answers only
                </span>

              </div>

              {/* SUGGESTIONS */}

              <div className="suggestions">

                <button
                  onClick={(e) =>
                    askQuestion(
                      e,
                      "What is the licence number?"
                    )
                  }
                >
                  Licence number
                </button>

                <button
                  onClick={(e) =>
                    askQuestion(
                      e,
                      "What is the licence expiry date?"
                    )
                  }
                >
                  Expiry date
                </button>

                <button
                  onClick={(e) =>
                    askQuestion(
                      e,
                      "What vehicles is this person authorised to drive?"
                    )
                  }
                >
                  Authorised vehicles
                </button>

                <button
                  onClick={(e) =>
                    askQuestion(
                      e,
                      "What is the email address?"
                    )
                  }
                >
                  Test missing information
                </button>

              </div>

              {/* CHAT WINDOW */}

              <div className="chat-window">

                {chat.length === 0 && (
                  <div className="chat-empty">
                    Ask a question about the
                    selected licence. Answers are
                    generated only from retrieved
                    OCR evidence.
                  </div>
                )}

                {chat.map(
                  (message, index) => (
                    <div
                      className={`chat-message ${message.role}`}
                      key={index}
                    >
                      <div className="message-label">
                        {message.role ===
                        "user"
                          ? "You"
                          : message.role ===
                            "system"
                          ? "System"
                          : "AI"}
                      </div>

                      <div className="message-body">

                        {message.text}

                        {message.sources
                          ?.length > 0 && (
                          <div className="sources">

                            {message.sources.map(
                              (source) => (
                                <div
                                  className="source"
                                  key={
                                    source.chunk_id
                                  }
                                >
                                  <strong>
                                    {
                                      source.chunk_id
                                    }
                                  </strong>

                                  <span>
                                    Page{" "}
                                    {
                                      source.page
                                    }{" "}
                                    · Region{" "}
                                    {
                                      source.region
                                    }
                                  </span>
                                </div>
                              )
                            )}

                          </div>
                        )}

                      </div>
                    </div>
                  )
                )}

                {chatLoading && (
                  <div className="chat-message assistant">

                    <div className="message-label">
                      AI
                    </div>

                    <div className="message-body typing">
                      Retrieving evidence…
                    </div>

                  </div>
                )}

              </div>

              {/* CHAT INPUT */}

              <form
                className="chat-input-row"
                onSubmit={askQuestion}
              >

                <input
                  value={question}
                  onChange={(e) =>
                    setQuestion(
                      e.target.value
                    )
                  }
                  placeholder="Ask something about this licence…"
                  disabled={chatLoading}
                />

                <button
                  type="submit"
                  className="primary-button"
                  disabled={
                    chatLoading ||
                    !question.trim()
                  }
                >
                  Ask
                </button>

              </form>

            </section>
          </>
        )}

      </main>

      <footer>
        Assessment demo · Fictional/sample licence
        data · No production PII storage
      </footer>

    </div>
  );
}

export default App;