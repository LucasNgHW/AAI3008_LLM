"""
rag/generator.py
----------------
Assembles the RAG generation prompt and calls the Anthropic Claude API.

Prompt design principles:
  1. Grounds the model strictly in retrieved course material
  2. Explicitly instructs the model to decline if context is insufficient
  3. Adapts explanation depth to the student's inferred difficulty level
  4. Includes the last N conversation turns for multi-turn coherence

Environment:
  ANTHROPIC_API_KEY  — required; set before running the app
"""

import os
from pathlib import Path
from google import genai

MODEL          = "gemini-2.5-flash"
HISTORY_WINDOW = 5  # conversation turns to inject into each prompt

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

    return f"""You are a personalised NLP learning assistant for university students.
Your role is to answer questions about Natural Language Processing using ONLY the \
course material provided in the <context> block below.

Student profile:
- Current level: {difficulty}
- Most studied topics: {topic_str}

Guidelines:
- Tailor your explanation depth and terminology to a {difficulty}-level student.
- Structure your answer clearly: lead with a direct response, then expand with detail.
- Use concrete examples from the course material where available.
- If the provided context does not contain enough information to answer reliably, \
say so explicitly — do NOT speculate or draw on external knowledge.
- Do not repeat the question back to the student.
- Keep answers focused and avoid unnecessary padding."""


def build_user_prompt(
    query: str,
    chunks: list[dict],
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Build the user-turn message by injecting:
      1. Recent conversation history (user turns only, for conciseness)
      2. Retrieved course material as XML-tagged context blocks
      3. The student's current question
    """
    # Format retrieved chunks — use basename to avoid leaking filesystem paths
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source_name = Path(chunk.get("source") or "unknown").name
        meta = (
            f"[{chunk.get('topic', 'general')} | "
            f"{chunk.get('difficulty', '?')} | "
            f"{source_name}]"
        )
        context_blocks.append(f"[{i}] {meta}\n{chunk['text']}")
    context_str = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."

    # Inject only user turns from recent history to keep augmentation tight
    history_str = ""
    if conversation_history:
        recent = conversation_history[-HISTORY_WINDOW:]
        history_lines = [
            f"{'Student' if t['role'] == 'user' else 'Assistant'}: {t['content'].strip()}"
            for t in recent
            if t.get("content", "").strip()
        ]
        if history_lines:
            history_str = "\n".join(history_lines)

    parts: list[str] = []
    if history_str:
        parts.append(f"<conversation_history>\n{history_str}\n</conversation_history>")
    parts.append(f"<context>\n{context_str}\n</context>")
    parts.append(f"Student question: {query}")

    return "\n\n".join(parts)


# ── Main generation function ───────────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: list[dict],
    user_profile: dict | None = None,
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Generate a grounded answer using the Claude API.

    Args:
        query:                The student's current question.
        chunks:               Retrieved (and optionally reranked) context chunks.
        user_profile:         Dict from UserProfile.to_dict().
        conversation_history: Full conversation so far as list of role/content dicts.

    Returns:
        The model's response as a plain string.
    """
    client        = get_client()
    system_prompt = build_system_prompt(user_profile)
    user_prompt   = build_user_prompt(query, chunks, conversation_history)

    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    response = client.models.generate_content(
        model=MODEL,
        contents=full_prompt,
    )
    return response.text
