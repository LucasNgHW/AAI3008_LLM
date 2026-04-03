"""
app/onboarding.py
-----------------
First-run upload gate for course materials.
"""

from __future__ import annotations

import streamlit as st

from pipeline.material_ingestion import ingest_all_materials
from storage.materials_db import count_materials

from app.ui_helpers import save_uploaded_pdfs


def render_material_upload_gate() -> None:
    """
    Stop the app and show the PDF uploader when the materials database is empty.
    """
    if count_materials() != 0:
        return

    st.subheader("Upload Course Materials")
    st.write(
        "Before using the assistant, upload your course PDFs. "
        "They will be stored in the database and then indexed into Qdrant."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Store PDFs and Build Knowledge Base", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Saving PDFs and indexing them..."):
                save_uploaded_pdfs(uploaded_files)
                ingest_all_materials(recreate=True)
            st.success("Course materials stored in the database and indexed successfully.")
            st.rerun()

    st.stop()
