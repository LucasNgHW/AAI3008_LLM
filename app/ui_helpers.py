"""
app/ui_helpers.py
-----------------
Small shared helpers for the Streamlit UI.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from personalisation.user_profile import UserProfile
from storage.materials_db import store_material


def warmup_models() -> None:
    """Warm the embedding and reranker models once per Streamlit session."""
    if st.session_state.get("models_warmed"):
        return

    with st.spinner("Loading models…"):
        from pipeline.embedder import warmup as embed_warmup
        from rag.reranker import warmup as rerank_warmup

        embed_warmup()
        rerank_warmup()

    st.session_state["models_warmed"] = True


def ensure_messages_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def get_profile(user_id: str) -> UserProfile:
    if (
        "profile_obj" not in st.session_state
        or st.session_state.get("profile_id") != user_id
        or st.session_state.get("profile_stale", False)
    ):
        st.session_state["profile_obj"] = UserProfile(user_id)
        st.session_state["profile_id"] = user_id
        st.session_state["profile_stale"] = False
    return st.session_state["profile_obj"]


def build_source_label(source: str | None) -> str:
    if not source:
        return "Course PDF"
    if source.startswith("db://materials/"):
        filename = source.rsplit("/", 1)[-1]
        return f"Course PDF: {filename}"
    filename = Path(source).name or source
    return f"Course PDF: {filename}"


def build_material_source(material: dict) -> str:
    return f"db://materials/{material['id']}/{material['filename']}"


def save_uploaded_pdfs(uploaded_files: list) -> None:
    for uploaded_file in uploaded_files:
        store_material(
            filename=uploaded_file.name,
            content=uploaded_file.getvalue(),
            mime_type=uploaded_file.type or "application/pdf",
        )


def render_sources(sources: list[dict]) -> None:
    with st.expander("📄 Retrieved sources", expanded=False):
        for i, src in enumerate(sources, start=1):
            score_val = src.get("rerank_score", src.get("score", "?"))
            score_str = f"{score_val:.3f}" if isinstance(score_val, float) else str(score_val)

            st.markdown(
                f"**[{i}]** `{src.get('difficulty', '?')}` | "
                f"score: `{score_str}` | "
                f"*{build_source_label(src.get('source'))}*"
            )

            preview = src.get("text", "")
            preview = preview[:300] + ("…" if len(preview) > 300 else "")
            st.caption(preview)


def render_timings(timings: dict) -> None:
    if timings:
        total = sum(timings.values())
        parts = [f"**{k}:** {v*1000:.0f} ms" for k, v in timings.items()]
        parts.append(f"**total:** {total*1000:.0f} ms")
        st.caption(" · ".join(parts))
