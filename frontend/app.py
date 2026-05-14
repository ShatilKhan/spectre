"""Spectre frontend — Streamlit UI for legal document processing."""

import os
import streamlit as st

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

# Sidebar
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload a legal document (fee proposal, engagement letter, contract, etc.)",
)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Extraction",
    "✏️ Review & Edit",
    "📝 Draft",
    "📊 Evaluation",
])

with tab1:
    st.info("Upload a PDF using the sidebar to begin extraction.")

with tab2:
    st.info("Extracted fields will appear here for review and correction.")

with tab3:
    st.info("Generated draft memos will appear here.")

with tab4:
    st.info("Evaluation metrics and benchmark results will appear here.")
