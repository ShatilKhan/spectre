"""Spectre frontend — Streamlit UI for legal document processing."""

import os
import json
import streamlit as st
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Spectre — Legal Document Processor",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Spectre — Legal Document Processor")


# ─── Session state ──────────────────────────────────────

for key in ["ocr_result", "extracted", "draft", "edited_fields"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "edited_fields" else {}

if "tab" not in st.session_state:
    st.session_state.tab = "📄 Extraction"


# ─── API helper ─────────────────────────────────────────

def call_api(endpoint: str, file_bytes=None, filename=None, json_data=None) -> dict | None:
    try:
        if file_bytes:
            resp = requests.post(
                f"{API_BASE_URL}{endpoint}",
                files={"file": (filename, file_bytes, "application/pdf")},
                timeout=120,
            )
        else:
            resp = requests.post(
                f"{API_BASE_URL}{endpoint}",
                json=json_data or {},
                timeout=600,
            )
        if resp.status_code == 200:
            return resp.json()
        detail = resp.json().get("detail", str(resp.text))
        st.error(f"API error ({resp.status_code}): {detail}")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The LLM may still be processing.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend at {API_BASE_URL}")
        return None


# ─── Sidebar ────────────────────────────────────────────

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"],
        help="Upload a legal document (fee proposal, engagement letter, contract, etc.)",
    )

    col1, col2 = st.columns(2)
    extract_btn = col1.button("🚀 Extract", use_container_width=True)
    clear_btn = col2.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        for key in ["ocr_result", "extracted", "draft", "edited_fields"]:
            st.session_state[key] = None if key != "edited_fields" else {}

    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        st.success("✅ Backend connected" if r.status_code == 200 else "❌ Backend error")
    except Exception:
        st.error("❌ Backend unreachable")


# ─── Process upload (streaming) ──────────────────────

if extract_btn and uploaded_file is not None:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name
    status = st.status("📄 Processing document...", expanded=True)

    try:
        resp = requests.post(
            f"{API_BASE_URL}/extract/stream",
            files={"file": (filename, file_bytes, "application/pdf")},
            stream=True,
            timeout=300,
        )
        if resp.status_code != 200:
            status.update(label=f"❌ Stream error: {resp.status_code}", state="error")
        else:
            event_type = None
            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode()
                if decoded.startswith("event: "):
                    event_type = decoded[7:]
                elif decoded.startswith("data: ") and event_type:
                    data = json.loads(decoded[6:])
                    if "message" in data:
                        status.update(label=f"⏳ {data['message']}", state="running")
                    if event_type == "ocr_result":
                        st.session_state.ocr_result = data
                    if event_type == "result":
                        st.session_state.ocr_result = data.get("ocr", st.session_state.ocr_result)
                        st.session_state.extracted = data.get("extracted", {})
                        st.session_state.draft = None
                        status.update(label="✅ Extraction complete", state="complete")
                    if event_type == "error":
                        status.update(label=f"❌ {data.get('message', 'Error')}", state="error")
                        st.stop()
            st.session_state.tab = "📄 Extraction"
    except Exception as e:
        status.update(label=f"❌ Error: {e}", state="error")

elif extract_btn and uploaded_file is None:
    st.sidebar.error("Please upload a PDF file first.")


# ─── Tab navigation (radio stays selected across reruns) ──

st.divider()
tabs = ["📄 Extraction", "✏️ Review & Edit", "📝 Draft", "📊 Evaluation"]
active_tab = st.radio(
    "Section", tabs, index=tabs.index(st.session_state.tab),
    horizontal=True, label_visibility="collapsed",
)
st.session_state.tab = active_tab


# ─── Tab 1: Extraction ──────────────────────────────────

if active_tab == "📄 Extraction":
    ocr = st.session_state.ocr_result
    if ocr:
        st.subheader(f"📄 {ocr.get('file_name')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Document Type", ocr.get("doc_type", "?"))
        c2.metric("Pages", ocr.get("page_count", "?"))
        c3.metric("OCR Confidence", f"{ocr.get('confidence', 0):.1%}")

        if ocr.get("pages"):
            with st.expander("📝 OCR Text by Page", expanded=False):
                for page in ocr["pages"]:
                    st.markdown(f"**Page {page['page']}** (conf: {page['confidence']:.1%})")
                    st.text(page["text"] or "(no text detected)")
                    st.divider()

        ext = st.session_state.extracted
        if ext:
            st.subheader("🧠 Extracted Fields")
            for key, value in ext.items():
                if not value:
                    continue
                label = key.replace("_", " ").title()
                if isinstance(value, dict):
                    # Nestable fields section
                    with st.expander(f"**{label}**", expanded=True):
                        for k2, v2 in value.items():
                            if v2:
                                st.markdown(f"**{k2.replace('_', ' ').title()}:** {v2}")
                elif isinstance(value, list):
                    st.markdown(f"**{label}:** {', '.join(str(v) for v in value if v) or 'N/A'}")
                else:
                    st.markdown(f"**{label}:** {value}")
    else:
        st.info("Upload a PDF using the sidebar and click **🚀 Extract** to begin.")


# ─── Tab 2: Review & Edit ───────────────────────────────

elif active_tab == "✏️ Review & Edit":
    ext = st.session_state.extracted
    if ext:
        st.subheader("✏️ Edit Extracted Fields")
        edited = {}
        for key, value in ext.items():
            if not value:
                continue
            label = key.replace("_", " ").title()
            current_val = st.session_state.edited_fields.get(key, value)

            if isinstance(value, dict):
                st.markdown(f"**{label}**")
                edited[key] = {}
                for k2, v2 in value.items():
                    label2 = k2.replace("_", " ").title()
                    cv = current_val.get(k2, v2) if isinstance(current_val, dict) else v2
                    edited[key][k2] = st.text_input(label2, value=str(cv), key=f"edit_{key}_{k2}")
            elif isinstance(value, list):
                new_val = st.text_area(
                    label, value=json.dumps(current_val, indent=2) if isinstance(current_val, list) else str(current_val),
                    key=f"edit_{key}", height=80,
                )
                try:
                    edited[key] = json.loads(new_val) if new_val.startswith("[") else new_val
                except json.JSONDecodeError:
                    edited[key] = new_val
            elif isinstance(value, (int, float)):
                edited[key] = st.number_input(label, value=current_val, key=f"edit_{key}")
            else:
                edited[key] = st.text_input(label, value=str(current_val), key=f"edit_{key}")

        st.session_state.edited_fields = edited

        if st.button("💾 Submit Corrections"):
            payload = {
                "original": ext,
                "corrected": edited,
                "changed_fields": [k for k in edited if edited.get(k) != ext.get(k)],
                "document_type": st.session_state.ocr_result.get("doc_type", "unknown") if st.session_state.ocr_result else "",
            }
            try:
                resp = requests.post(f"{API_BASE_URL}/feedback", json=payload, timeout=30)
                if resp.status_code == 200:
                    st.success("✅ Corrections submitted")
                else:
                    st.error(f"Submission failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Extracted data will appear here for review after extraction.")


# ─── Tab 3: Draft ────────────────────────────────────────

elif active_tab == "📝 Draft":
    if st.session_state.extracted and st.button("📝 Generate Draft"):
        ocr = st.session_state.ocr_result
        full_text = ocr.get("full_text", "") if ocr else ""
        st.info("⏳ Generating draft (30-120 sec)...")
        result = call_api("/draft", json_data={
            "extracted_data": st.session_state.extracted,
            "ocr_text": full_text,
            "pages": ocr.get("pages", []) if ocr else [],
            "doc_id": ocr.get("file_name", "doc") if ocr else "doc",
        })
        if result and "draft" in result:
            st.session_state.draft = result["draft"]
            st.success(f"✅ Draft generated ({len(result['draft'])} chars from {result.get('evidence_count', 0)} passages)")

    if st.session_state.draft:
        st.subheader("📝 Generated Draft Memo")
        st.markdown(st.session_state.draft)
        st.download_button("💾 Download Draft", st.session_state.draft, file_name="draft_memo.md", mime="text/markdown")
    else:
        st.info("Extract data first, then click **📝 Generate Draft**.")


# ─── Tab 4: Evaluation ──────────────────────────────────

elif active_tab == "📊 Evaluation":
    st.subheader("📊 Evaluation Tools")

    # Section 1: LLM-as-judge
    with st.expander("🧠 LLM-as-Judge Evaluation", expanded=True):
        if st.session_state.extracted:
            if st.button("📊 Run LLM-as-Judge"):

                st.info("⏳ Running evaluation (30-60 sec)...")
                result = call_api("/evaluate", json_data={
                    "extracted": st.session_state.extracted,
                    "ground_truth": st.session_state.extracted,
                })
                if result and "error" not in result:
                    c1, c2 = st.columns(2)
                    metrics = [
                        ("🎯 Context Relevance", result.get("context_relevance", 0), True),
                        ("📖 Answer Faithfulness", result.get("answer_faithfulness", 0), True),
                        ("💡 Answer Relevance", result.get("answer_relevance", 0), True),
                        ("⚠️ Hallucination Rate", result.get("hallucination_rate", 0), False),
                    ]
                    for (label, val, higher_is_better), col in zip(metrics, [c1, c1, c2, c2]):
                        pct = int(val * 100)
                        if higher_is_better:
                            col.metric(label, f"{pct}%")
                            col.progress(val)
                            col.caption("Target: >80%")
                        else:
                            col.metric(label, f"{pct}%")
                            col.progress(val)
                            col.caption("Target: <10%")

                    with st.expander("📋 Full Results"):
                        st.json(result)
                elif result and "error" in result:
                    st.error(result["error"])
        else:
            st.info("Extract data first, then run LLM-as-Judge.")

    # ─── helper to display any benchmark result dict as key-value rows ───
    def _show_keyvals(data: dict, skip_keys: set | None = None):
        """Display a flat dict as labeled rows. Skip structural keys."""
        skip = skip_keys or {"per_document", "per_doc_type", "per_clause_type",
                               "per_field", "keyword_coverage", "note", "error",
                               "mode"}
        for k, v in data.items():
            if k in skip or k.startswith("_"):
                continue
            if isinstance(v, (int, float)):
                label = k.replace("_", " ").title()
                if isinstance(v, float):
                    st.metric(label, f"{v:.4f}" if v < 0.01 else f"{v:.2%}")
                else:
                    st.metric(label, v)

    def _show_note(result: dict):
        note = result.get("note")
        if note:
            st.caption(note)

    # ─── Section 2: Classifier Benchmark ─────────────────
    with st.expander("📚 Classifier Benchmark (CUAD)", expanded=False):
        st.markdown(
            "Runs the document classifier against **13,155 labeled clauses** "
            "from real SEC-filed contracts. Tests classifier accuracy across **41 clause types**."
        )
        st.caption("Downloads 1.81 MB dataset on first run. Cached afterward.")

        cnum = st.number_input("Sample size", min_value=10, max_value=500, value=200, step=10, key="clf_samples")

        if st.button("🚀 Run Classifier Benchmark", type="primary", key="clf_btn"):
            st.info(f"⏳ Running on {cnum} CUAD clauses...")
            result = call_api("/benchmark", json_data={"mode": "classifier", "num_samples": cnum})
            if result and "error" not in result:
                # Top-level metrics row
                acc = result.get("overall_accuracy", 0)
                col1, col2, col3 = st.columns(3)
                col1.metric("Overall Accuracy", f"{int(acc*100)}%")
                col2.metric("Samples", result.get("num_samples", 0))
                col3.metric("Mode", "classifier")
                st.progress(acc)
                _show_note(result)

                # Per-doc-type breakdown
                per_type = result.get("per_doc_type", {})
                if per_type:
                    st.subheader("Per-Document-Type Accuracy")
                    cols = st.columns(len(per_type))
                    for i, (dt, info) in enumerate(sorted(per_type.items())):
                        p = int(info["accuracy"] * 100)
                        cols[i].metric(dt, f"{p}%", help=f"{info['correct']}/{info['count']} correct")

                # Per-clause breakdown (top 15)
                per_clause = result.get("per_clause_type", {})
                if per_clause:
                    st.subheader("Per-Clause-Type Accuracy (top 15)")
                    clause_data = []
                    for ct, info in list(per_clause.items())[:15]:
                        clause_data.append({
                            "Clause Type": ct,
                            "Accuracy": f"{int(info['accuracy'] * 100)}%",
                            "Correct": info['correct'],
                            "Total": info['count'],
                        })
                    st.dataframe(clause_data, use_container_width=True)

                # Keyword coverage
                kw = result.get("keyword_coverage", {})
                if kw:
                    st.subheader("Keyword Coverage (theoretical max)")
                    kw_rows = []
                    for dt, info in sorted(kw.items()):
                        kw_rows.append({
                            "Doc Type": dt,
                            "Coverage": f"{int(info['keyword_coverage'] * 100)}%",
                            "Samples": info["samples"],
                        })
                    st.dataframe(kw_rows, use_container_width=True)

                with st.expander("📋 Raw JSON"):
                    st.json(result)
            elif result and "error" in result:
                st.error(result["error"])

    # ─── Section 3: OCR Benchmark ────────────────────────
    with st.expander("📝 OCR Accuracy Benchmark", expanded=False):
        st.markdown(
            "Measures **PaddleOCR accuracy** against real CUAD contracts "
            "with HuggingFace ground-truth text. Computes **CER** (Character Error Rate) "
            "and **WER** (Word Error Rate)."
        )
        st.caption("Downloads up to 10 CUAD PDFs (~200 MB) on first run. Cached by HF datasets.")
        st.warning("Each PDF takes 10-30s to OCR. Total: 1-5 min.", icon="⏳")

        ocr_docs = st.number_input("PDFs to OCR", min_value=1, max_value=10, value=5, step=1, key="ocr_docs")

        if st.button("📊 Run OCR Benchmark", type="primary", key="ocr_btn"):
            st.info(f"⏳ OCR-ing {ocr_docs} CUAD PDFs... (this may take several minutes)")
            bar = st.progress(0, text="Starting...")
            result = call_api("/benchmark", json_data={"mode": "ocr", "max_docs": ocr_docs})
            bar.progress(100, text="Done")

            if result and "error" not in result:
                # Top-level metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg CER", f"{result.get('average_cer', 1.0):.2%}",
                            delta_color="inverse", help="Character Error Rate (0% = perfect)")
                col2.metric("Avg WER", f"{result.get('average_wer', 1.0):.2%}",
                            delta_color="inverse", help="Word Error Rate (0% = perfect)")
                col3.metric("Avg Accuracy", f"{result.get('average_accuracy', 0):.2%}",
                            help="1 - CER, higher is better")

                col4, col5, col6 = st.columns(3)
                med = result.get("median_cer")
                p95 = result.get("p95_cer")
                p99 = result.get("p99_cer")
                col4.metric("Median CER", f"{med:.2%}" if med is not None else "N/A",
                            delta_color="inverse")
                col5.metric("P95 CER", f"{p95:.2%}" if p95 is not None else "N/A",
                            delta_color="inverse")
                col6.metric("P99 CER", f"{p99:.2%}" if p99 is not None else "N/A",
                            delta_color="inverse")

                st.metric("Documents", f"{result.get('num_success', 0)} / {result.get('num_docs', 0)}")
                _show_note(result)

                # Per-document table
                per_doc = result.get("per_document", [])
                if per_doc:
                    st.subheader("Per-Document Results")
                    rows = []
                    for d in per_doc:
                        rows.append({
                            "File": d.get("file_name", "?"),
                            "CER": f"{d.get('cer', 1.0):.2%}",
                            "WER": f"{d.get('wer', 1.0):.2%}",
                            "Accuracy": f"{d.get('accuracy', 0):.2%}",
                            "OCR Conf": f"{d.get('ocr_confidence', 0):.0%}",
                            "Chars": d.get("ocr_chars", 0),
                            "Doc Type": d.get("doc_type", "?"),
                            "Latency": f"{d.get('latency_sec', 0):.0f}s",
                        })
                    st.dataframe(rows, use_container_width=True)

                with st.expander("📋 Raw JSON"):
                    st.json(result)
            elif result and "error" in result:
                st.error(result["error"])

    # ─── Section 4: Extraction Benchmark ─────────────────
    with st.expander("🧠 Full Pipeline Extraction Benchmark", expanded=False):
        st.markdown(
            "Runs the full pipeline (**OCR → classify → LLM extract**) on CUAD PDFs "
            "and checks whether extracted field values are **grounded in the source text**."
        )
        st.caption(
            "Measures: grounding_rate (fraction of fields found in source), "
            "hallucination_rate (fraction fabricated). "
            "Processes up to 5 documents (LLM is the bottleneck)."
        )
        st.warning("Each doc takes 30-90s with LLM. Total: 3-8 min.", icon="⏳")

        ext_docs = st.number_input("Docs to process", min_value=1, max_value=5, value=3, step=1, key="ext_docs")

        if st.button("🧪 Run Extraction Benchmark", type="primary", key="ext_btn"):
            st.info(f"⏳ Running full pipeline on {ext_docs} CUAD PDFs... (3-8 min)")
            bar = st.progress(0, text="Starting...")
            result = call_api("/benchmark", json_data={"mode": "extraction", "max_docs": ext_docs})
            bar.progress(100, text="Done")

            if result and "error" not in result:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Grounding Rate", f"{result.get('grounding_rate', 0):.2%}",
                            help="% of extracted fields traceable to source text")
                col2.metric("Hallucination Rate", f"{result.get('hallucination_rate', 0):.2%}",
                            delta_color="inverse",
                            help="% of fields that are fabricated")
                col3.metric("Overall Quality", f"{result.get('overall_extraction_quality', 0):.2%}",
                            help="grounding_rate * (1 - hallucination_rate)")
                col4.metric("Mode", "extraction")

                st.metric(
                    "Fields Checked",
                    f"{result.get('total_fields_checked', 0)} "
                    f"({result.get('total_fields_present', 0)} present, "
                    f"{result.get('total_fields_grounded', 0)} grounded, "
                    f"{result.get('total_fields_hallucinated', 0)} hallucinated)"
                )
                st.metric("Documents", f"{result.get('num_success', 0)} / {result.get('num_docs', 0)}")
                _show_note(result)

                # Per-field breakdown
                per_field = result.get("per_field", {})
                if per_field:
                    st.subheader("Per-Field Reliability")
                    field_rows = []
                    for fname, info in sorted(per_field.items()):
                        field_rows.append({
                            "Field": fname,
                            "Occurrences": info["occurrences"],
                            "Present": info["present"],
                            "Grounded": info["grounded"],
                            "Grounding Rate": f"{info['grounding_rate']:.0%}",
                        })
                    st.dataframe(field_rows, use_container_width=True)

                # Per-doc breakdown
                per_doc = result.get("per_document", [])
                if per_doc:
                    st.subheader("Per-Document Results")
                    doc_rows = []
                    for d in per_doc:
                        doc_rows.append({
                            "File": d.get("file_name", "?"),
                            "Type": d.get("doc_type", "?"),
                            "Checked": d.get("fields_checked", 0),
                            "Present": d.get("fields_present", 0),
                            "Grounded": d.get("fields_grounded", 0),
                            "Hallucinated": d.get("hallucinated", 0),
                            "Grounding": f"{d.get('grounding_rate', 0):.0%}",
                            "OCR Lat": f"{d.get('ocr_latency_sec', 0):.0f}s",
                            "Extract Lat": f"{d.get('extract_latency_sec', 0):.0f}s",
                        })
                    st.dataframe(doc_rows, use_container_width=True)

                with st.expander("📋 Raw JSON"):
                    st.json(result)
            elif result and "error" in result:
                st.error(result["error"])

    # ─── Section 5: Run All ──────────────────────────────
    with st.expander("🏃 Run All Benchmarks", expanded=False):
        st.markdown(
            "Runs **all three benchmarks** sequentially and returns a "
            "combined report. This takes **5-15 minutes** total."
        )
        st.warning("Long-running operation. Do not navigate away.", icon="⏳")

        if st.button("🏆 Run All Benchmarks", type="primary", key="all_btn"):
            st.info("⏳ Running all benchmarks (5-15 min)...")
            bar = st.progress(0, text="Starting...")
            result = call_api("/benchmark", json_data={"mode": "all"})
            bar.progress(100, text="Done")

            if result and "error" not in result:
                st.success(f"All benchmarks complete in {result.get('total_time_sec', 0):.0f}s")

                clf = result.get("classifier", {})
                ocr = result.get("ocr", {})
                ext = result.get("extraction", {})

                # Summary row
                cols = st.columns(3)
                if "error" not in clf:
                    acc = clf.get("overall_accuracy", 0)
                    cols[0].metric("Classifier Accuracy", f"{int(acc * 100)}%",
                                   help=f"{clf.get('num_samples', 0)} clauses")
                if "error" not in ocr:
                    cer = ocr.get("average_cer", 1.0)
                    cols[1].metric("OCR Avg CER", f"{cer:.2%}",
                                   delta_color="inverse",
                                   help=f"{ocr.get('num_success', 0)}/{ocr.get('num_docs', 0)} docs")
                if "error" not in ext:
                    quality = ext.get("overall_extraction_quality", 0)
                    cols[2].metric("Extraction Quality", f"{quality:.2%}",
                                   help=f"{ext.get('num_success', 0)}/{ext.get('num_docs', 0)} docs")

                # Detail expanders for each sub-benchmark
                for label, sub in [("Classifier Details", clf),
                                    ("OCR Details", ocr),
                                    ("Extraction Details", ext)]:
                    if "error" not in sub:
                        with st.expander(label, expanded=False):
                            _show_keyvals(sub)
                            # Show per-doc if present
                            per_doc = sub.get("per_document", [])
                            if per_doc:
                                st.subheader("Per-Document")
                                st.dataframe(
                                    [{k: v for k, v in d.items()
                                      if not isinstance(v, (list, dict))}
                                     for d in per_doc],
                                    use_container_width=True,
                                )
                            st.json(sub)

                with st.expander("📋 Combined Raw JSON"):
                    st.json(result)
            elif result and "error" in result:
                st.error(result["error"])
