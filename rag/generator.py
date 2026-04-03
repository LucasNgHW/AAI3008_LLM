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
import re
from pathlib import Path
from typing import Iterator
from google import genai

MODEL          = "gemini-2.5-flash-lite"    
MAX_TOKENS   = 2048   
HISTORY_WINDOW = 3                     # 5 turns was ~2k extra tokens; 3 is enough for context
MAX_CHUNK_CHARS = 500                  # truncate each chunk to cap prompt size
MAX_HISTORY_CHARS = 600               # total chars of history injected

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY before running the app."
            )
        _client = genai.Client(api_key=api_key)
    return _client

def clean_answer_text(text: str) -> str:
    # Remove citation-style markers like [1], [2], [1, 3]
    text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text)
    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    # Clean repeated spaces
    text = re.sub(r" {2,}", " ", text)
    return text

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
        "Answer questions about Natural Language Processing using only the course material "
        "provided in the <context> block. You may synthesise across multiple chunks, "
        "but do not add outside facts unless clearly labeled as general background.",
        "",
        "Student profile:",
        f"  - Current level: {difficulty}",
        f"  - Most studied topics: {topic_str}",
        "",
        "Writing guidelines:",
        f"  - Depth: {depth_guide}",
        "  - Repetition: do not repeat the same point or restate the question.",
        "  - Avoid repeating definitions after already giving a direct answer.",
        "  - Formatting: use bullet points for comparisons, pros/cons, differences, or lists. For comparison questions, present each item clearly under its own bullet.",
        "  - Summary: when helpful, end with one brief sentence starting with 'In short:'.",
        "  - Prefer phrasing that stays close to the retrieved course material instead of broad general commentary.",
        "  - Do not introduce examples, applications, or claims unless they are supported by the retrieved material.",
        "  - Do not include citation markers like [1], [2], or reference numbers in the answer.",
        "  - Do not end with generic sign-off phrases.",
    ]
    return "\n".join(lines)


def build_direct_system_prompt(user_profile: dict | None = None) -> str:
    difficulty = (user_profile or {}).get("preferred_difficulty", "intermediate")
    return "\n".join([
        "You are a helpful university learning assistant.",
        "This response is for a direct conversational/help request, not a course-content retrieval question.",
        "Answer briefly and clearly.",
        f"Adapt your tone to the student's level: {difficulty}.",
        "If the student asks for course-specific content, tell them to ask a question about their uploaded materials.",
        "Do not invent references to slides or PDFs.",
    ])


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
    if not chunks:
        context_str = "No relevant course material was retrieved for this question."
    else:
        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            raw_source = chunk.get("source") or "unknown"
            if isinstance(raw_source, str) and raw_source.startswith("db://materials/"):
                source_name = raw_source.rsplit("/", 1)[-1]
            else:
                source_name = Path(raw_source).name
            difficulty = chunk.get("difficulty") or "unknown"
            topic = chunk.get("topic") or "general"
            section_title = chunk.get("section_title")
            slide = chunk.get("slide")
            meta_parts = [
                f"source: {source_name}",
                f"topic: {topic}",
                f"difficulty: {difficulty}",
            ]
            if section_title:
                meta_parts.append(f"section: {section_title}")
            if slide is not None:
                meta_parts.append(f"slide: {slide}")
            meta = f"({', '.join(meta_parts)})"
            clean_chunk_text = clean_answer_text(chunk["text"])
            context_blocks.append(f"Source {i}: {meta}\n{clean_chunk_text}")
        context_str = "\n\n".join(context_blocks)

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


def _make_direct_prompt(
    query: str,
    user_profile: dict | None,
    conversation_history: list[dict] | None,
) -> str:
    history_str = ""
    if conversation_history:
        recent = conversation_history[-2:]
        lines = [
            f"{'Student' if t['role'] == 'user' else 'Assistant'}: {t['content'].strip()}"
            for t in recent
            if t.get("content", "").strip()
        ]
        if lines:
            history_str = "\n".join(lines)[-MAX_HISTORY_CHARS:]

    parts = [build_direct_system_prompt(user_profile)]
    if history_str:
        parts.append(f"<conversation_history>\n{history_str}\n</conversation_history>")
    parts.append(f"Student message: {query}")
    return "\n\n".join(parts)


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
                yield clean_answer_text(text)
    except Exception as exc:
        err = str(exc)
        if "429" in err or "quota" in err.lower() or "rate" in err.lower():
            yield (
                "\n\n⚠️ **Gemini quota limit reached.** "
                "Please try again later, use a different API key/project, "
                "or switch to a paid/billed API setup."
            )
        else:
            raise


def stream_direct_answer(
    query: str,
    user_profile: dict | None = None,
    conversation_history: list[dict] | None = None,
) -> Iterator[str]:
    """
    Stream a short direct response for conversational/help queries that do not
    require retrieval.
    """
    client = get_client()
    prompt = _make_direct_prompt(query, user_profile, conversation_history)

    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=prompt,
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield clean_answer_text(text)
    except Exception as exc:
        err = str(exc)
        if "429" in err or "quota" in err.lower() or "rate" in err.lower():
            yield (
                "\n\n⚠️ **Gemini quota limit reached.** "
                "Please try again later, use a different API key/project, "
                "or switch to a paid/billed API setup."
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


def generate_answer_stream(
    query: str,
    chunks: list[dict],
    user_profile: dict | None = None,
    conversation_history: list[dict] | None = None,
) -> Iterator[str]:
    """
    Streaming version of generate_answer using Gemini.
    Yields text chunks as they arrive.
    Use with st.write_stream() in Streamlit.
    """
    client = get_client()

    system_prompt = build_system_prompt(user_profile)
    user_prompt   = build_user_prompt(query, chunks, conversation_history)

    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    response = client.models.generate_content_stream(
        model=MODEL,
        contents=full_prompt,
    )

    for chunk in response:
        text = getattr(chunk, "text", None)
        if text:
            yield clean_answer_text(text)
