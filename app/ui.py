"""
app/ui.py
---------
Streamlit chat interface for the NLP Learning Assistant.

Features:
  - Chat loop with full session state and message history display
  - Multi-turn dialogue: last 5 user+assistant turns injected into RAG prompt
  - Personalisation sidebar: inferred difficulty, top topics, recommended topics
  - Optional topic and difficulty filters for targeted retrieval
  - Retrieved source display (expandable) below each assistant answer

Session state keys
------------------
  messages      list[dict]   full conversation history
  profile_cache dict         cached profile.to_dict() — refreshed after each turn
  profile_id    str          user_id the cache belongs to (invalidated on ID change)

Run with:
    streamlit run app/ui.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from rag.retriever          import retrieve_with_context
from rag.reranker           import rerank
from rag.generator          import generate_answer
from personalisation.user_profile import UserProfile

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Learning Assistant",
    page_icon="📚",
    layout="wide",
)

# ── Helper: get or refresh the cached UserProfile ─────────────────────────────

def get_profile(user_id: str) -> UserProfile:
    """
    Return a UserProfile, loading from disk only when the user_id changes
    or when explicitly invalidated (profile_stale flag).
    """
    if (
        "profile_obj" not in st.session_state
        or st.session_state.get("profile_id") != user_id
        or st.session_state.get("profile_stale", False)
    ):
        st.session_state["profile_obj"]   = UserProfile(user_id)
        st.session_state["profile_id"]    = user_id
        st.session_state["profile_stale"] = False
    return st.session_state["profile_obj"]


def refresh_profile_cache(user_id: str) -> None:
    """Force a disk reload on the next get_profile() call."""
    st.session_state["profile_stale"] = True


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 NLP Assistant")
    st.divider()

    user_id = st.text_input("Your student ID", value="student_1", key="user_id_input")
    profile = get_profile(user_id)

    st.subheader("Your Learning Profile")
    st.markdown(f"**Current level:** {profile.preferred_difficulty.capitalize()}")

    top = profile.top_topics
    st.markdown(f"**Top topics:** {', '.join(top) if top else 'None yet — start asking!'}")

    recs = profile.recommended_topics
    if recs:
        st.markdown(f"**Suggested to explore:** {', '.join(recs)}")

    st.divider()
    st.subheader("Retrieval Filters")
    st.caption("Optionally restrict answers to specific content.")

    TOPICS = [
        "Any", "tokenisation", "embeddings", "language_models", "transformers",
        "sentiment", "named_entity", "parsing", "text_classification",
    ]
    topic_sel    = st.selectbox("Topic",       TOPICS)
    diff_sel     = st.selectbox("Difficulty",  ["Any", "beginner", "intermediate", "advanced"])
    use_reranker = st.toggle("Enable reranker (more accurate, slower)", value=True)

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("NLP Learning Assistant")
st.caption(
    "Ask anything about your NLP course. "
    "I'll retrieve the most relevant course material and tailor my answer to your level."
)


def render_sources(sources: list[dict]) -> None:
    """Render an expandable source panel."""
    with st.expander("📄 Retrieved sources", expanded=False):
        for i, src in enumerate(sources, start=1):
            score_val = src.get("rerank_score", src.get("score", "?"))
            score_str = f"{score_val:.3f}" if isinstance(score_val, float) else str(score_val)
            st.markdown(
                f"**[{i}]** `{src.get('topic', '?')}` | "
                f"`{src.get('difficulty', '?')}` | "
                f"score: `{score_str}` | "
                f"*{src.get('source', 'unknown')}*"
            )
            preview = src["text"][:300] + ("…" if len(src["text"]) > 300 else "")
            st.caption(preview)


# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# ── Handle new query ───────────────────────────────────────────────────────────
if query := st.chat_input("Ask about NLP..."):

    # Show student message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching course materials…"):

            # Reload profile in case another session wrote to disk
            profile = get_profile(user_id)

            # Build filter kwargs from sidebar selections
            filter_kwargs: dict = {}
            if topic_sel != "Any":
                filter_kwargs["topic_filter"] = topic_sel
            if diff_sel != "Any":
                filter_kwargs["difficulty_filter"] = diff_sel

            # Stage 1: retrieve (over-fetch if reranker is active)
            first_stage_k = 15 if use_reranker else 5
            chunks = retrieve_with_context(
                query=query,
                conversation_history=st.session_state.messages[:-1],
                user_profile=profile.to_dict(),
                top_k=first_stage_k,
                **filter_kwargs,
            )

            # Stage 2: optional rerank
            if use_reranker and chunks:
                chunks = rerank(query, chunks, top_n=5)

            # Stage 3: generate
            answer = generate_answer(
                query=query,
                chunks=chunks,
                user_profile=profile.to_dict(),
                conversation_history=st.session_state.messages[:-1],
            )

        st.markdown(answer)
        if chunks:
            render_sources(chunks)

    # Persist assistant message (with sources for history re-render)
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": chunks,
    })

    # Log interaction and mark profile cache stale so sidebar refreshes next render
    if chunks:
        top_chunk = chunks[0]
        profile.log_interaction(
            query=query,
            topic=top_chunk.get("topic", "general"),
            difficulty=top_chunk.get("difficulty", "intermediate"),
        )
        refresh_profile_cache(user_id)

    # Rerun only to refresh the sidebar profile stats — no double-render because
    # the new messages are already in session_state and rendered above.
    st.rerun()
