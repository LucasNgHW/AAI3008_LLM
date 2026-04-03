"""Main Streamlit page for the learning assistant."""

import os
import re
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st

from app.onboarding import render_material_upload_gate
from app.sidebar import render_sidebar
from app.ui_helpers import (
    ensure_messages_state,
    get_profile,
    render_sources,
    render_timings,
    warmup_models,
)
from rag.reflection import rewrite_query_for_retry, should_retry_retrieval
from rag.router import route_query
from rag.retriever import retrieve_with_context
from rag.reranker import rerank
from rag.generator import generate_answer_stream, stream_direct_answer
from storage.materials_db import init_db


st.set_page_config(
    page_title="Learning Assistant",
    page_icon="📚",
    layout="wide",
)

init_db()
warmup_models()
ensure_messages_state()

sidebar_state = render_sidebar()
user_id = sidebar_state.user_id
material_options = sidebar_state.material_options
material_sel = sidebar_state.material_sel
diff_sel = sidebar_state.diff_sel
use_reranker = sidebar_state.use_reranker
show_timings = sidebar_state.show_timings


def _clean_streamed_answer(answer: str) -> str:
    """Normalise minor formatting artefacts from streamed model output."""
    answer = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", answer)
    answer = re.sub(r"\s+([.,;:!?])", r"\1", answer)
    answer = re.sub(r" {2,}", " ", answer)
    return answer


def _build_filter_kwargs(selected_source: str | None, difficulty: str) -> dict:
    filter_kwargs: dict = {}
    if selected_source is not None:
        filter_kwargs["source_filter"] = selected_source
    if difficulty != "Any":
        filter_kwargs["difficulty_filter"] = difficulty
    return filter_kwargs


def _render_no_material_warning() -> None:
    st.warning(
        "⚠️ I couldn't find relevant course material for that question. "
        "Try rephrasing or removing filters."
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": "⚠️ No relevant material found."}
    )
    st.rerun()


st.title("Learning Assistant")
st.caption(
    "Ask anything about your uploaded course materials. "
    "I'll retrieve the most relevant material and tailor the answer to your level."
)

render_material_upload_gate()


if not st.session_state.messages:
    st.info(
        "👋 Welcome! Ask me anything about your uploaded course materials. "
        "I can only answer questions based on what you have uploaded."
    )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sources"):
                render_sources(msg["sources"])
            if show_timings and msg.get("timings"):
                render_timings(msg["timings"])


if query := st.chat_input("Ask about your uploaded materials..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        profile = get_profile(user_id)
        timings: dict = {}

        selected_source = material_options.get(material_sel)
        filter_kwargs = _build_filter_kwargs(selected_source, diff_sel)

        route = route_query(
            query=query,
            selected_source=selected_source,
            conversation_history=st.session_state.messages[:-1],
        )
        chunks: list[dict] = []

        if route == "direct":
            st.caption("Route: direct answer")
            t0 = time.perf_counter()
            answer = st.write_stream(
                stream_direct_answer(
                    query=query,
                    user_profile=profile.to_dict(),
                    conversation_history=st.session_state.messages[:-1],
                )
            )
            timings["generate"] = time.perf_counter() - t0
        else:
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

            retried_query: str | None = None
            if should_retry_retrieval(chunks):
                # First-pass retrieval can miss when the question is too vague
                # ("how does it work?"). Retry once with a clearer standalone query.
                progress.progress(25, text="Retrying with a clarified query…")
                retried_query = rewrite_query_for_retry(
                    query=query,
                    conversation_history=st.session_state.messages[:-1],
                )
                if retried_query and retried_query != query:
                    t0 = time.perf_counter()
                    retry_chunks = retrieve_with_context(
                        query=retried_query,
                        conversation_history=None,
                        user_profile=profile.to_dict(),
                        top_k=first_stage_k,
                        **filter_kwargs,
                    )
                    timings["retrieve_retry"] = time.perf_counter() - t0
                    if retry_chunks:
                        chunks = retry_chunks

            progress.progress(50, text="Reranking results…")
            if use_reranker and chunks:
                t0 = time.perf_counter()
                chunks = rerank(query, chunks, top_n=5)
                timings["rerank"] = time.perf_counter() - t0

            progress.progress(100, text="Generating answer…")
            progress.empty()

            # Relevance gate: refuse to generate if no chunks or all scores are too low.
            # Cosine similarity below this threshold means the retrieved material
            # is unrelated to the question (e.g. baking PDFs for an NLP question).
            _LOW_RELEVANCE_THRESHOLD = 0.30
            best_score = (
                max((c.get("score", 0) for c in chunks), default=0) if chunks else 0
            )
            if not chunks or best_score < _LOW_RELEVANCE_THRESHOLD:
                _render_no_material_warning()

            if retried_query and retried_query != query:
                st.caption(f"Retried search with a clarified query: `{retried_query}`")

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
                answer = _clean_streamed_answer(answer)

            timings["generate"] = time.perf_counter() - t0

        if chunks:
            render_sources(chunks)
        if show_timings:
            render_timings(timings)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": chunks,
            "timings": timings,
            "route": route,
        }
    )

    if route == "rag" and chunks:
        from pipeline.chunker import infer_difficulty

        top_chunk = chunks[0]
        profile.log_interaction(
            query=query,
            topic=top_chunk.get("topic", "general"),
            difficulty=infer_difficulty(query),
        )

    st.rerun()
