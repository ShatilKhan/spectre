"""Spectre frontend — Streamlit UI for legal document processing."""

import os
import json
import time
from pathlib import Path

import streamlit as st
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Spectre — Legal Document Processor",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Spectre — Legal Document Processor")
st.markdown(
    "Upload legal documents to extract structured data, "
    "review and edit fields, and generate grounded draft memos."
)

# ─── Session state ──────────────────────────────────────

if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = None
if "extracted" not in st.session_state:
    st.session_state.extracted = None
if "draft" not in st.session_state:
    st.session_state.draft = None
if "edited_fields" not in st.session_state:
    st.session_state.edited_fields = {}

# ─── Sidebar ────────────────────────────────────────────

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload a legal document (fee proposal, engagement letter, contract, etc.)",
)

col1, col2 = st.sidebar.columns(2)
extract_btn = col1.button("🚀 Extract", use_container_width=True)
clear_btn = col2.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state.ocr_result = None
    st.session_state.extracted = None
    st.session_state.draft = None
    st.session_state.edited_fields = {}
    st.rerun()

# ─── API helpers ─────────────────────────────────────────

def call_api(endpoint: str, file_bytes, filename: str) -> dict:
    """Call a backend endpoint with file upload and return parsed JSON."""
    files = {"file": (filename, file_bytes, "application/pdf")}
    with st.spinner(f"Calling {endpoint}..."):
        try:
            resp = requests.post(
                f"{API_BASE_URL}{endpoint}",
                files=files,
                timeout=300,
            )
            if resp.status_code == 200:
                return resp.json()
            detail = resp.json().get("detail", str(resp.text))
            st.error(f"API error ({resp.status_code}): {detail}")
            return None
        except requests.exceptions.Timeout:
            st.error("Request timed out. The LLM may still be processing on first run.")
            return None
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to backend at {API_BASE_URL}")
            return None


# ─── Process upload ──────────────────────────────────────

if extract_btn and uploaded_file is not None:
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name

    # Step 1: Upload / OCR
    ocr_result = call_api("/upload", file_bytes, filename)
    if ocr_result:
        st.session_state.ocr_result = ocr_result
        st.session_state.extracted = None  # reset old extraction
        st.session_state.draft = None

        # Step 2: Extract
        ext_result = call_api("/extract", file_bytes, filename)
        if ext_result:
            st.session_state.extracted = ext_result.get("extracted", {})
            st.sidebar.success("Extraction complete!")
        else:
            st.sidebar.warning("OCR done but extraction failed.")
elif extract_btn and uploaded_file is None:
    st.sidebar.error("Please upload a PDF file first.")


# ─── API health check ────────────────────────────────────

try:
    r = requests.get(f"{API_BASE_URL}/health", timeout=5)
    if r.status_code == 200:
        st.sidebar.success("✅ Backend connected")
    else:
        st.sidebar.error("❌ Backend error")
except Exception:
    st.sidebar.error("❌ Backend unreachable")


# ─── Tabs ────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Extraction",
    "✏️ Review & Edit",
    "📝 Draft",
    "📊 Evaluation",
])

# ─── Tab 1: Extraction ──────────────────────────────────

with tab1:
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

with tab2:
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
                    f"{key}",
                    value=json.dumps(current_val, indent=2) if isinstance(current_val, list) else str(current_val),
                    key=f"edit_{key}",
                    height=80,
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
                "document_type": st.session_state.ocr_result.get("doc_type", "unknown"),
            }
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/feedback",
                    json=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    st.success("✅ Corrections submitted — the system will improve.")
                else:
                    st.error(f"Submission failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Extracted data will appear here for review and correction after extraction.")

# ─── Tab 3: Draft ────────────────────────────────────────

with tab3:
    if st.session_state.extracted and st.button("📝 Generate Draft"):
        ocr = st.session_state.ocr_result
        with st.spinner("Generating grounded draft memo..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/draft",
                    json={
                        "extracted_data": st.session_state.extracted,
                        "source_passages": [
                            {"document": p.get("text", ""), "metadata": {"page": p.get("page", 0)}}
                            for p in (ocr.get("pages", []) if ocr else [])
                        ],
                    },
                    timeout=300,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.draft = data.get("draft", str(data))
                else:
                    st.error(f"Draft failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.draft:
        st.subheader("📝 Generated Draft Memo")
        st.markdown(st.session_state.draft)

        st.download_button(
            "💾 Download Draft",
            st.session_state.draft,
            file_name="draft_memo.md",
            mime="text/markdown",
        )
    else:
        st.info("Extract data first, then click **📝 Generate Draft**.")

# ─── Tab 4: Evaluation ──────────────────────────────────

with tab4:
    if st.session_state.extracted and st.button("📊 Run Evaluation"):
        with st.spinner("Running LLM-as-judge evaluation..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/evaluate",
                    json={
                        "extracted": st.session_state.extracted,
                        "ground_truth": st.session_state.extracted,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    st.json(resp.json())
                else:
                    st.error(f"Evaluation failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Extract data first, then click **📊 Run Evaluation**.")
