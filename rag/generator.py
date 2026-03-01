"""
rag/generator.py
----------------
Assembles the RAG generation prompt and calls the Claude API.

Prompt design principles:
  1. Grounds the model strictly in retrieved course material
  2. Explicitly instructs the model to decline if context is insufficient
  3. Adapts explanation depth to the student's inferred difficulty level
  4. Includes the last N conversation turns for multi-turn coherence
"""

import anthropic

MODEL        = "claude-sonnet-4-6"
MAX_TOKENS   = 1024
HISTORY_WINDOW = 5   # number of previous turns to include in the prompt

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


# ── Prompt assembly ───────────────────────────────────────────────────────────

def build_system_prompt(user_profile: dict | None = None) -> str:
    difficulty  = (user_profile or {}).get("preferred_difficulty", "intermediate")
    top_topics  = (user_profile or {}).get("top_topics", [])
    topic_str   = ", ".join(top_topics) if top_topics else "none recorded yet"

    return f"""You are a personalised NLP learning assistant for university students.
Your role is to answer questions about Natural Language Processing using ONLY the course material provided in the <context> block below.

Student profile:
- Current level: {difficulty}
- Most studied topics: {topic_str}

Guidelines:
- Tailor your explanation depth and terminology to a {difficulty}-level student.
- Structure your answer clearly: lead with a direct response, then expand with detail.
- Use concrete examples from the course material where available.
- If the provided context does not contain enough information to answer reliably, say so explicitly — do NOT speculate or draw on external knowledge.
- Do not repeat the question back to the student.
- Keep answers focused and avoid unnecessary padding."""


def build_user_prompt(
    query: str,
    chunks: list[dict],
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Construct the user-turn prompt by injecting:
      1. Retrieved course material as XML-tagged context
      2. The last HISTORY_WINDOW conversation turns (for multi-turn coherence)
      3. The student's current question
    """
    # Format retrieved chunks as numbered context blocks
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = f"[{chunk.get('topic', 'general')} | {chunk.get('difficulty', '?')} | {chunk.get('source', '?')}]"
        context_blocks.append(f"[{i}] {meta}\n{chunk['text']}")
    context_str = "\n\n".join(context_blocks)

    # Format recent conversation history
    history_str = ""
    if conversation_history:
        recent = conversation_history[-HISTORY_WINDOW:]
        history_lines = []
        for turn in recent:
            role    = "Student" if turn["role"] == "user" else "Assistant"
            content = turn["content"].strip()
            history_lines.append(f"{role}: {content}")
        history_str = "\n".join(history_lines)

    parts = []
    if history_str:
        parts.append(f"<conversation_history>\n{history_str}\n</conversation_history>")
    parts.append(f"<context>\n{context_str}\n</context>")
    parts.append(f"Student question: {query}")

    return "\n\n".join(parts)


# ── Main generation function ──────────────────────────────────────────────────

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
        user_profile:         Dict from UserProfile containing difficulty level etc.
        conversation_history: Full conversation so far as list of role/content dicts.

    Returns:
        The model's response as a plain string.
    """
    client = get_client()

    system_prompt = build_system_prompt(user_profile)
    user_prompt   = build_user_prompt(query, chunks, conversation_history)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text
