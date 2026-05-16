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
                if value:
                    display = json.dumps(value, indent=2) if not isinstance(value, str) else value
                    st.text(f"{key}: {display[:300]}")
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
            current_val = st.session_state.edited_fields.get(key, value)
            if isinstance(value, list):
                new_val = st.text_area(
                    key, value=json.dumps(current_val, indent=2) if isinstance(current_val, list) else str(current_val),
                    key=f"edit_{key}", height=80,
                )
                try:
                    edited[key] = json.loads(new_val) if new_val.startswith("[") else new_val
                except json.JSONDecodeError:
                    edited[key] = new_val
            elif isinstance(value, (int, float)):
                edited[key] = st.number_input(key, value=current_val, key=f"edit_{key}")
            else:
                edited[key] = st.text_input(key, value=str(current_val), key=f"edit_{key}")

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
    if st.session_state.extracted and st.button("📊 Run Evaluation"):
        result = call_api("/evaluate", json_data={
            "extracted": st.session_state.extracted,
            "ground_truth": st.session_state.extracted,
        })
        if result:
            st.json(result)
    else:
        st.info("Extract data first, then click **📊 Run Evaluation**.")
