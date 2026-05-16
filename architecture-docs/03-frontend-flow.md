# Frontend Flow

## UI Rendering Pipeline

![Sequence](./diagrams/out/03-sequence.png)

The frontend is a single-page Streamlit application with 4 tabs and a sidebar for document upload.

### Tab Structure

```python
# frontend/app.py:131-136
tabs = ["📄 Extraction", "✏️ Review & Edit", "📝 Draft", "📊 Evaluation"]
active_tab = st.radio(
    "Section", tabs, index=tabs.index(st.session_state.tab),
    horizontal=True, label_visibility="collapsed",
)
```

Radio buttons are used instead of `st.tabs()` to prevent tab-jump bugs on rerun.

### State Management

Session state keys:

```python
# frontend/app.py:21-23
for key in ["ocr_result", "extracted", "draft", "edited_fields"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "edited_fields" else {}
```

| Key | Type | Set By | Used By |
|-----|------|--------|---------|
| `ocr_result` | dict | SSE stream result | Extraction tab, Draft tab |
| `extracted` | dict | SSE stream result | Extraction tab, Review tab, Draft tab, Evaluation tab |
| `draft` | str | Draft generation | Draft tab |
| `edited_fields` | dict | Review tab edits | Review tab |
| `tab` | str | Radio button | Tab selection persistence |

### SSE Stream Processing

The extraction endpoint uses Server-Sent Events for progress reporting. The frontend parses the stream line by line:

```python
# frontend/app.py:90-119
resp = requests.post(
    f"{API_BASE_URL}/extract/stream",
    files={"file": (filename, file_bytes, "application/pdf")},
    stream=True, timeout=300,
)
for line in resp.iter_lines():
    if not line:
        continue
    decoded = line.decode()
    if decoded.startswith("event: "):
        event_type = decoded[7:]
    elif decoded.startswith("data: ") and event_type:
        data = json.loads(decoded[6:])
        if event_type == "result":
            st.session_state.ocr_result = data.get("ocr", {})
            st.session_state.extracted = data.get("extracted", {})
```

---

## Tab Details

### Tab 1: Extraction

Displays OCR metadata (file name, doc type, page count, confidence) and extracted fields with humanized labels. Each field type renders differently:

| Field Type | Rendering |
|------------|-----------|
| `str` | Plain text |
| `list` | Comma-separated values |
| `dict` | Nested expander section |

### Tab 2: Review & Edit

Renders each extracted field as an editable input. Field types map to Streamlit widgets:

| Python Type | Streamlit Widget |
|-------------|-----------------|
| `str` | `st.text_input` |
| `int` / `float` | `st.number_input` |
| `list` | `st.text_area` (JSON edit) |
| `dict` | Nested `st.text_input` per key |

A "Submit Corrections" button sends the diff to `POST /feedback`.

### Tab 3: Draft

Calls `POST /draft` with the extracted data and OCR text. Renders the generated memo as Markdown. Provides a download button for `.md` export.

### Tab 4: Evaluation

5 expandable sections:

| Section | API Endpoint | Key Metrics Displayed |
|---------|-------------|----------------------|
| LLM-as-Judge | `POST /evaluate` | context_relevance, answer_faithfulness, hallucination_rate |
| Classifier Benchmark | `POST /benchmark?mode=classifier` | overall_accuracy, per-doc-type, per-clause, keyword_coverage |
| OCR Benchmark | `POST /benchmark?mode=ocr` | avg CER/WER, median/p95/p99 CER, per-document table |
| Extraction Benchmark | `POST /benchmark?mode=extraction` | grounding_rate, hallucination_rate, per-field breakdown |
| Run All | `POST /benchmark?mode=all` | Summary of all 3 above |

---

## API Helper

```python
# frontend/app.py:32-56
def call_api(endpoint, file_bytes=None, filename=None, json_data=None):
    try:
        if file_bytes:
            resp = requests.post(f"{API_BASE_URL}{endpoint}", ...)
        else:
            resp = requests.post(f"{API_BASE_URL}{endpoint}", json=json_data or {}, ...)
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API error ({resp.status_code}): {...}")
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend at {API_BASE_URL}")
```

---

## Known Issues

| # | Concern | Location |
|---|---------|----------|
| 1 | Benchmark progress bar uses indeterminate `st.progress(0)` — no step count | `frontend/app.py:366` |
| 2 | LLM-as-Judge uses extracted data as ground truth (self-evaluation) | `frontend/app.py:269-272` |
| 3 | Large JSON results expand to full height — no max-height scroll | `frontend/app.py:347,400,474,509` |
