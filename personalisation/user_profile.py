"""
personalisation/user_profile.py
--------------------------------
Tracks each student's interaction history and infers their current learning level.

Storage:
  - One JSON file per student at profiles/{user_id}.json
  - Written on every interaction (lightweight — profile files stay small)

Difficulty inference:
  - Maps difficulty labels to numeric scores: beginner=1, intermediate=2, advanced=3
  - Takes a rolling average of the last ROLLING_WINDOW interactions
  - Rounds back to a label for prompt injection

Topic recommendations:
  - Compares the student's interaction history against ALL_TOPICS
  - Surfaces topics with zero or very low engagement as suggestions
"""

import json
import os
import logging
from collections import Counter
from pathlib import Path

from project_paths import PROFILES_DIR as DEFAULT_PROFILES_DIR

logger = logging.getLogger(__name__)

PROFILES_DIR   = DEFAULT_PROFILES_DIR
ROLLING_WINDOW = 10   # recent interactions used for difficulty inference

ALL_TOPICS = [
    "tokenisation", "embeddings", "language_models", "transformers",
    "sentiment", "named_entity", "parsing", "text_classification", "general",
]

DIFFICULTY_SCORE = {"beginner": 1, "intermediate": 2, "advanced": 3}
SCORE_LABEL      = {1: "beginner", 2: "intermediate", 3: "advanced"}

_BLANK_PROFILE = {
    "query_history":     [],
    "topic_counts":      {},
    "difficulty_scores": [],
}


def _blank(user_id: str) -> dict:
    return {
        "user_id":           user_id,
        "query_history":     [],
        "topic_counts":      {},
        "difficulty_scores": [],
    }


class UserProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.path    = Path(PROFILES_DIR) / f"{user_id}.json"
        self.data    = self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        """
        Load profile from disk. Returns a blank profile on any error
        (missing file, malformed JSON, unexpected schema) so the app
        never crashes due to a corrupted profile file.
        """
        if not self.path.exists():
            return _blank(self.user_id)

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load profile %s: %s. Starting fresh.", self.path, exc)
            return _blank(self.user_id)

        # Ensure all required keys exist (handles profiles from older versions)
        for key, default in _BLANK_PROFILE.items():
            data.setdefault(key, type(default)())   # [] or {}
        data.setdefault("user_id", self.user_id)

        # Sanitise: difficulty_scores must be ints 1–3
        data["difficulty_scores"] = [
            s for s in data["difficulty_scores"]
            if isinstance(s, (int, float)) and 1 <= s <= 3
        ]

        return data

    def save(self) -> None:
        os.makedirs(PROFILES_DIR, exist_ok=True)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except OSError as exc:
            logger.error("Failed to save profile %s: %s", self.path, exc)

    # ── Interaction logging ────────────────────────────────────────────────────

    def log_interaction(self, query: str, topic: str, difficulty: str) -> None:
        """
        Record a single student interaction.

        Args:
            query:      The student's question text.
            topic:      Topic label of the top retrieved chunk.
            difficulty: Difficulty label of the top retrieved chunk.
        """
        self.data["query_history"].append(query)

        counts = self.data["topic_counts"]
        counts[topic] = counts.get(topic, 0) + 1

        score = DIFFICULTY_SCORE.get(difficulty, 2)
        self.data["difficulty_scores"].append(score)

        self.save()

    # ── Inference ──────────────────────────────────────────────────────────────

    @property
    def preferred_difficulty(self) -> str:
        """Infer current level from a rolling average of recent interactions."""
        scores = self.data["difficulty_scores"]
        if not scores:
            return "intermediate"
        recent  = scores[-ROLLING_WINDOW:]
        avg     = sum(recent) / len(recent)
        clamped = max(1, min(3, round(avg)))
        return SCORE_LABEL[clamped]

    @property
    def top_topics(self) -> list[str]:
        """Return the 3 most frequently studied topics."""
        counts = self.data["topic_counts"]
        return [t for t, _ in Counter(counts).most_common(3)]

    @property
    def recommended_topics(self) -> list[str]:
        """Topics with zero or low engagement — suggested areas to explore."""
        counts = self.data["topic_counts"]
        ranked = sorted(
            [t for t in ALL_TOPICS if t != "general"],
            key=lambda t: counts.get(t, 0),
        )
        return ranked[:3]

    def to_dict(self) -> dict:
        """Return a summary dict for injection into prompts."""
        return {
            "user_id":              self.user_id,
            "preferred_difficulty": self.preferred_difficulty,
            "top_topics":           self.top_topics,
            "recommended_topics":   self.recommended_topics,
            "total_interactions":   len(self.data["query_history"]),
        }
