"""
rag/generator.py
----------------
Assembles the RAG prompt and calls the Gemini API.

Two generation modes
--------------------
generate_answer()        — blocking, returns a complete string.
                           Used by evaluate.py and tests.

stream_answer()          — yields text chunks as they arrive via the Gemini
                           streaming API.  Use this in the UI so the answer
                           renders token-by-token instead of after a full
                           round-trip wait.

Prompt context budget
---------------------
Each chunk is up to ~800 chars.  With 5 chunks + history the prompt sits at
~5-6 k tokens — well within Gemini Flash's context window and short enough
that time-to-first-token stays low.

Environment
-----------
  GEMINI_API_KEY  — required; set before running the app.
"""

import os
from pathlib import Path
from typing import Iterator
from google import genai

MODEL          = "gemini-2.5-flash-lite"    
HISTORY_WINDOW = 3                     # 5 turns was ~2k extra tokens; 3 is enough for context
MAX_CHUNK_CHARS = 500                  # truncate each chunk to cap prompt size
MAX_HISTORY_CHARS = 600               # total chars of history injected

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


# ── Prompt assembly ────────────────────────────────────────────────────────────

def build_system_prompt(user_profile: dict | None = None) -> str:
    difficulty = (user_profile or {}).get("preferred_difficulty", "intermediate")
    top_topics = (user_profile or {}).get("top_topics", [])
    topic_str  = ", ".join(top_topics) if top_topics else "none recorded yet"

    depth_guide = {
        "beginner":     (
            "Use plain language, avoid jargon, and build concepts step by step. "
            "Use relatable analogies to make abstract ideas concrete."
        ),
        "intermediate": (
            "Use correct NLP terminology, explain the intuition behind each idea, "
            "and connect new concepts to ones the student has already seen."
        ),
        "advanced":     (
            "Use precise technical language, discuss trade-offs and edge cases, "
            "and go deep into mechanisms without over-explaining basics."
        ),
    }.get(difficulty, "Explain concepts clearly with appropriate depth.")

    lines = [
        "You are a personalised NLP learning assistant for university students.",
        "Answer questions about Natural Language Processing using the course material "
        "provided in the <context> block. You may synthesise across multiple chunks.",
        "",
        "Student profile:",
        f"  - Current level: {difficulty}",
        f"  - Most studied topics: {topic_str}",
        "",
        "Writing guidelines:",
        f"  - Depth: {depth_guide}",
        "  - Structure: open with a direct 1-2 sentence answer, then explain the "
        "concept fully, then give a concrete example from the course material.",
        "  - Length: write as much as the topic requires. A good conceptual answer "
        "is typically 150-400 words — do not truncate early or pad with filler.",
        "  - Synthesis: if multiple context chunks are relevant, weave them into a "
        "single coherent explanation rather than addressing each chunk separately.",
        "  - If the context genuinely lacks the information needed, say so briefly "
        "— do not speculate or invent facts not present in the material.",
        "  - Do not repeat the question. Do not end with generic sign-off phrases.",
    ]
    return "\n".join(lines)


def build_user_prompt(
    query: str,
    chunks: list[dict],
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Build the user-turn message:
      1. Recent conversation history (all roles, capped to HISTORY_WINDOW)
      2. Retrieved course material as numbered XML context blocks
      3. The student's current question
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source_name = Path(chunk.get("source") or "unknown").name
        meta = (
            f"[{chunk.get('topic', 'general')} | "
            f"{chunk.get('difficulty', '?')} | "
            f"{source_name}]"
        )
        text = chunk["text"][:MAX_CHUNK_CHARS]  # cap to stay within token budget
        context_blocks.append(f"[{i}] {meta}\n{text}")
    context_str = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."

    history_str = ""
    if conversation_history:
        recent = conversation_history[-HISTORY_WINDOW:]
        lines  = [
            f"{'Student' if t['role'] == 'user' else 'Assistant'}: {t['content'].strip()}"
            for t in recent
            if t.get("content", "").strip()
        ]
        if lines:
            history_str = "\n".join(lines)[-MAX_HISTORY_CHARS:]  # keep most recent chars

    parts: list[str] = []
    if history_str:
        parts.append(f"<conversation_history>\n{history_str}\n</conversation_history>")
    parts.append(f"<context>\n{context_str}\n</context>")
    parts.append(f"Student question: {query}")

    return "\n\n".join(parts)


def _make_prompt(
    query: str,
    chunks: list[dict],
    user_profile: dict | None,
    conversation_history: list[dict] | None,
) -> str:
    return (
        build_system_prompt(user_profile)
        + "\n\n"
        + build_user_prompt(query, chunks, conversation_history)
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def stream_answer(
    query: str,
    chunks: list[dict],
    user_profile: dict | None = None,
    conversation_history: list[dict] | None = None,
) -> Iterator[str]:
    """
    Stream the answer token-by-token using the Gemini streaming API.

    Yields text fragments as they arrive so the UI can render them
    incrementally — the student sees the first words in ~300-500 ms
    instead of waiting for the full response (~1500 ms).

    Usage in Streamlit:
        with st.chat_message("assistant"):
            answer = st.write_stream(
                stream_answer(query, chunks, user_profile, history)
            )

    Args:
        query:                The student's current question.
        chunks:               Retrieved (and optionally reranked) context chunks.
        user_profile:         Dict from UserProfile.to_dict().
        conversation_history: Full conversation as list of role/content dicts.

    Yields:
        str fragments of the model response.
    """
    client = get_client()
    prompt = _make_prompt(query, chunks, user_profile, conversation_history)

    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=prompt,
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield text
    except Exception as exc:
        err = str(exc)
        if "429" in err or "quota" in err.lower() or "rate" in err.lower():
            yield (
                "\n\n⚠️ **Gemini free-tier rate limit reached.** "
                "Wait 60 seconds and try again, or reduce query frequency. "
                "Upgrading to a paid API key removes this limit."
            )
        else:
            raise


def generate_answer(
    query: str,
    chunks: list[dict],
    user_profile: dict | None = None,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Blocking generation — returns the complete response as a string.
    Use stream_answer() in the UI for better perceived latency.

    Args:
        query:                The student's current question.
        chunks:               Retrieved (and optionally reranked) context chunks.
        user_profile:         Dict from UserProfile.to_dict().
        conversation_history: Full conversation as list of role/content dicts.

    Returns:
        The model's response as a plain string.
    """
    return "".join(stream_answer(query, chunks, user_profile, conversation_history))