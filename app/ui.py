"""
app/ui.py
---------
Streamlit chat interface for the NLP Learning Assistant.

Features:
  - Chat loop with full session state and message history display
  - Multi-turn dialogue: last 5 turns injected into the RAG generation prompt
  - Personalisation sidebar: inferred difficulty, top topics, recommended topics
  - Optional topic and difficulty filters for targeted retrieval
  - Retrieved source display (expandable) below each assistant answer

Run with:
    streamlit run app/ui.py
"""
import os
import sys

# Add project root (parent of /app) to Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
import streamlit as st
from rag.retriever import retrieve_with_context
from rag.reranker  import rerank
from rag.generator import generate_answer
from personalisation.user_profile import UserProfile

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Learning Assistant",
    page_icon="📚",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 NLP Assistant")
    st.divider()

    user_id = st.text_input("Your student ID", value="student_1", key="user_id_input")
    profile = UserProfile(user_id)

    st.subheader("Your Learning Profile")
    difficulty_label = profile.preferred_difficulty.capitalize()
    st.markdown(f"**Current level:** {difficulty_label}")

    top = profile.top_topics
    if top:
        st.markdown(f"**Top topics:** {', '.join(top)}")
    else:
        st.markdown("**Top topics:** None yet — start asking questions!")

    recs = profile.recommended_topics
    if recs:
        st.markdown(f"**Suggested to explore:** {', '.join(recs)}")

    st.divider()
    st.subheader("Retrieval Filters")
    st.caption("Optionally restrict answers to specific content.")

    TOPICS = ["Any", "tokenisation", "embeddings", "language_models", "transformers",
              "sentiment", "named_entity", "parsing", "text_classification"]
    topic_sel = st.selectbox("Topic", TOPICS)
    diff_sel  = st.selectbox("Difficulty", ["Any", "beginner", "intermediate", "advanced"])
    use_reranker = st.toggle("Enable reranker (more accurate, slower)", value=True)

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("NLP Learning Assistant")
st.caption("Ask anything about your NLP course. I'll retrieve the most relevant course material and tailor my answer to your level.")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show retrieved sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Retrieved sources", expanded=False):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(
                        f"**[{i}]** `{src.get('topic', '?')}` | "
                        f"`{src.get('difficulty', '?')}` | "
                        f"score: `{src.get('rerank_score', src.get('score', '?'))}` | "
                        f"*{src.get('source', 'unknown')}*"
                    )
                    st.caption(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))

# ── Handle new query ──────────────────────────────────────────────────────────
if query := st.chat_input("Ask about NLP..."):

    # Display the student's message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching course materials..."):

            # Build filter kwargs
            filter_kwargs = {}
            if topic_sel != "Any":
                filter_kwargs["topic_filter"] = topic_sel
            if diff_sel != "Any":
                filter_kwargs["difficulty_filter"] = diff_sel

            # Retrieve: over-fetch if reranker is enabled
            first_stage_k = 15 if use_reranker else 5
            chunks = retrieve_with_context(
                query=query,
                conversation_history=st.session_state.messages[:-1],  # exclude current
                user_profile=profile.to_dict(),
                top_k=first_stage_k,
                **filter_kwargs,
            )

            # Optionally rerank
            if use_reranker and chunks:
                chunks = rerank(query, chunks, top_n=5)

            # Generate answer
            answer = generate_answer(
                query=query,
                chunks=chunks,
                user_profile=profile.to_dict(),
                conversation_history=st.session_state.messages[:-1],
            )

        st.markdown(answer)

        # Show sources inline
        if chunks:
            with st.expander("📄 Retrieved sources", expanded=False):
                for i, src in enumerate(chunks, start=1):
                    st.markdown(
                        f"**[{i}]** `{src.get('topic', '?')}` | "
                        f"`{src.get('difficulty', '?')}` | "
                        f"score: `{src.get('rerank_score', src.get('score', '?'))}` | "
                        f"*{src.get('source', 'unknown')}*"
                    )
                    st.caption(src["text"][:300] + ("..." if len(src["text"]) > 300 else ""))

    # Persist assistant message with sources attached
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": chunks,
    })

    # Log interaction to user profile
    if chunks:
        top_chunk = chunks[0]
        profile.log_interaction(
            query=query,
            topic=top_chunk.get("topic", "general"),
            difficulty=top_chunk.get("difficulty", "intermediate"),
        )

    # Reload sidebar to reflect updated profile
    st.rerun()
