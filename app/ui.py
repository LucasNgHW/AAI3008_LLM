"""
app/ui.py
---------
Streamlit chat interface for the NLP Learning Assistant.

Features
--------
- Chat loop with full session state and message history display
- Multi-turn dialogue: last 5 turns injected into the RAG prompt
- Personalisation sidebar: inferred difficulty, top topics, recommended topics
- Optional topic and difficulty filters for targeted retrieval
- Retrieved source display (expandable) below each assistant answer
- Streaming generation: answer renders token-by-token via st.write_stream
- Per-component latency display (opt-in toggle)

Session state keys
------------------
  messages       list[dict]   full conversation history
  profile_obj    UserProfile  cached profile object
  profile_id     str          user_id the cache belongs to
  profile_stale  bool         if True, reload profile on next get_profile() call
  models_warmed  bool         True after one-time model warmup has run

Run with:
    streamlit run app/ui.py
"""

import os
import re
import sys
import time
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from rag.retriever            import retrieve_with_context
from rag.reranker             import rerank
from rag.generator            import generate_answer, generate_answer_stream
from personalisation.user_profile import UserProfile
from pipeline.material_ingestion import (
    delete_all_materials_everywhere,
    delete_material_everywhere,
    ingest_all_materials,
)
from storage.materials_db import count_materials, init_db, list_materials, store_material

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Learning Assistant",
    page_icon="📚",
    layout="wide",
)

init_db()

# ── One-time model warmup ──────────────────────────────────────────────────────
# Pays the cold-start cost (~3 s) before the first query arrives.

if not st.session_state.get("models_warmed"):
    with st.spinner("Loading models…"):
        from pipeline.embedder import warmup as embed_warmup
        from rag.reranker       import warmup as rerank_warmup
        embed_warmup()
        rerank_warmup()
    st.session_state["models_warmed"] = True

# ── Profile helpers ────────────────────────────────────────────────────────────

def get_profile(user_id: str) -> UserProfile:
    if (
        "profile_obj" not in st.session_state
        or st.session_state.get("profile_id") != user_id
        or st.session_state.get("profile_stale", False)
    ):
        st.session_state["profile_obj"]   = UserProfile(user_id)
        st.session_state["profile_id"]    = user_id
        st.session_state["profile_stale"] = False
    return st.session_state["profile_obj"]


def refresh_profile_cache(_: str) -> None:
    st.session_state["profile_stale"] = True


def build_source_label(source: str | None) -> str:
    if not source:
        return "Course PDF"
    if source.startswith("db://materials/"):
        filename = source.rsplit("/", 1)[-1]
        return f"Course PDF: {filename}"
    filename = Path(source).name or source
    return f"Course PDF: {filename}"


def save_uploaded_pdfs(uploaded_files: list) -> None:
    for uploaded_file in uploaded_files:
        store_material(
            filename=uploaded_file.name,
            content=uploaded_file.getvalue(),
            mime_type=uploaded_file.type or "application/pdf",
        )


def build_material_source(material: dict) -> str:
    return f"db://materials/{material['id']}/{material['filename']}"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 NLP Assistant")
    st.divider()

    materials_notice = st.session_state.pop("materials_notice", None)
    if materials_notice:
        getattr(st, materials_notice["level"])(materials_notice["message"])

    user_id = st.text_input("Your student ID", value="student_1", key="user_id_input")

    @st.cache_resource
    def load_profile(uid: str) -> UserProfile:
        return UserProfile(uid)

    profile = load_profile(user_id)

    st.subheader("Your Learning Profile")
    difficulty_label = profile.preferred_difficulty.capitalize()
    colours = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
    st.markdown(f"**Current level:** {colours.get(difficulty_label, '⚪')} {difficulty_label}")

    top = profile.top_topics
    st.markdown(f"**Top topics:** {', '.join(top) if top else 'None yet — start asking!'}")

    recs = profile.recommended_topics
    if recs:
        st.markdown(f"**Suggested to explore:** {', '.join(recs)}")

    st.divider()
    st.subheader("Stored Materials")
    materials = list_materials()
    if materials:
        st.caption(f"{len(materials)} PDF(s) stored in the database")
        selected_materials: list[dict] = []

        if st.button("Delete All Materials", key="delete_all_materials"):
            with st.spinner("Deleting all materials from the database and Qdrant..."):
                deleted_count = delete_all_materials_everywhere()
            st.session_state["materials_notice"] = {
                "level": "success",
                "message": f"Deleted {deleted_count} material(s) and cleared the Qdrant collection.",
            }
            st.rerun()

        for material in materials[:6]:
            checked = st.checkbox(
                material["filename"],
                key=f"select_material_{material['id']}",
            )
            if checked:
                selected_materials.append(material)

        if st.button("Delete Selected", key="delete_selected_materials"):
            if not selected_materials:
                st.session_state["materials_notice"] = {
                    "level": "warning",
                    "message": "Select at least one material to delete.",
                }
                st.rerun()

            deleted_names: list[str] = []
            with st.spinner("Deleting selected materials from the database and Qdrant..."):
                for material in selected_materials:
                    deleted = delete_material_everywhere(material["id"])
                    if deleted:
                        deleted_names.append(material["filename"])
                    st.session_state.pop(f"select_material_{material['id']}", None)

            st.session_state["materials_notice"] = {
                "level": "success" if deleted_names else "warning",
                "message": (
                    f"Deleted {len(deleted_names)} material(s): {', '.join(deleted_names)}."
                    if deleted_names
                    else "No selected materials were deleted."
                ),
            }
            st.rerun()
    else:
        st.caption("No course PDFs stored yet.")

    st.divider()
    st.subheader("Retrieval Filters")
    st.caption("Optionally restrict answers by stored course material or difficulty.")

    material_options = {"Any": None}
    for material in materials:
        material_options[material["filename"]] = build_material_source(material)

    material_sel = st.selectbox("Course Material", list(material_options.keys()))

    diff_sel     = st.selectbox("Difficulty", ["Any", "beginner", "intermediate", "advanced"])
    use_reranker = st.toggle("Enable reranker (more accurate, slower)", value=True)
    show_timings = st.toggle("Show latency breakdown", value=False)

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Reset my learning profile"):
        profile_path = f"./profiles/{user_id}.json"
        if os.path.exists(profile_path):
            os.remove(profile_path)
        st.cache_resource.clear()
        st.success("Profile reset.")
        st.rerun()

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("NLP Learning Assistant")
st.caption(
    "Ask anything about your NLP course. "
    "I'll retrieve the most relevant material and tailor the answer to your level."
)

if count_materials() == 0:
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

# Welcome message when chat is empty
if not st.session_state.messages:
    st.info("👋 Welcome! Ask me anything about your NLP course. Try starting with a topic like *transformers*, *tokenisation*, or *sentiment analysis*.")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sources"):
                render_sources(msg["sources"])
            if show_timings and msg.get("timings"):
                render_timings(msg["timings"])

# ── Handle new query ───────────────────────────────────────────────────────────
if query := st.chat_input("Ask about your course material..."):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        profile = get_profile(user_id)
        timings: dict = {}

        filter_kwargs: dict = {}
        selected_source = material_options.get(material_sel)
        if selected_source is not None:
            filter_kwargs["source_filter"] = selected_source
        if diff_sel != "Any":
            filter_kwargs["difficulty_filter"] = diff_sel

        # Stage 1: retrieve
        first_stage_k = 15 if use_reranker else 5
        progress = st.progress(0, text="Searching course materials…")
        t0 = time.perf_counter()
        chunks = retrieve_with_context(
            query=query,
            conversation_history=st.session_state.messages[:-1],
            user_profile=profile.to_dict(),
            top_k=first_stage_k,
            **filter_kwargs,
        )
        timings["retrieve"] = time.perf_counter() - t0

        # Stage 2: rerank
        progress.progress(50, text="Reranking results…")
        if use_reranker and chunks:
            t0 = time.perf_counter()
            chunks = rerank(query, chunks, top_n=5)
            timings["rerank"] = time.perf_counter() - t0

        progress.progress(100, text="Generating answer…")
        progress.empty()

        best_score = chunks[0].get("rerank_score", chunks[0].get("score", 0)) if chunks else 0
        if not chunks:
            st.warning("⚠️ I couldn't find relevant course material for that question. Try rephrasing or removing filters.")
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ No relevant material found."})
            st.rerun()

        # Stage 3: stream generation — renders tokens as they arrive
        t0 = time.perf_counter()
        answer = st.write_stream(
            generate_answer_stream(
                query=query,
                chunks=chunks,
                user_profile=profile.to_dict(),
                conversation_history=st.session_state.messages[:-1],
            )
        )

        if isinstance(answer, str):
            answer = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", answer)
            answer = re.sub(r"\s+([.,;:!?])", r"\1", answer)
            answer = re.sub(r" {2,}", " ", answer)

        timings["generate"] = time.perf_counter() - t0

        # Rating buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👍 Helpful", key=f"up_{len(st.session_state.messages)}"):
                st.toast("Thanks for the feedback!")
        with col2:
            if st.button("👎 Not helpful", key=f"down_{len(st.session_state.messages)}"):
                st.toast("We'll try to do better!")

        if chunks:
            render_sources(chunks)
        if show_timings:
            render_timings(timings)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": chunks,
        "timings": timings,
    })

    if chunks:
        from pipeline.chunker import infer_difficulty
        top_chunk = chunks[0]
        from pipeline.chunker import infer_difficulty
        profile.log_interaction(
            query=query,
            topic=top_chunk.get("topic", "general"),
            difficulty=infer_difficulty(query),  # ← query-based
        )

    st.rerun()
