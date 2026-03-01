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
from collections import Counter
from pathlib import Path

PROFILES_DIR   = "./profiles"
ROLLING_WINDOW = 10   # number of recent interactions used for difficulty inference

ALL_TOPICS = [
    "tokenisation", "embeddings", "language_models", "transformers",
    "sentiment", "named_entity", "parsing", "text_classification", "general",
]

DIFFICULTY_SCORE = {"beginner": 1, "intermediate": 2, "advanced": 3}
SCORE_LABEL      = {1: "beginner", 2: "intermediate", 3: "advanced"}


class UserProfile:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.path    = Path(PROFILES_DIR) / f"{user_id}.json"
        self.data    = self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, "r") as f:
                return json.load(f)
        return {
            "user_id":          self.user_id,
            "query_history":    [],
            "topic_counts":     {},
            "difficulty_scores": [],
        }

    def save(self) -> None:
        os.makedirs(PROFILES_DIR, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    # ── Interaction logging ───────────────────────────────────────────────────

    def log_interaction(self, query: str, topic: str, difficulty: str) -> None:
        """
        Record a single student interaction.

        Args:
            query:      The student's question text.
            topic:      Topic label of the top retrieved chunk.
            difficulty: Difficulty label of the top retrieved chunk.
        """
        self.data["query_history"].append(query)

        # Increment topic counter
        counts = self.data["topic_counts"]
        counts[topic] = counts.get(topic, 0) + 1

        # Append numeric difficulty score
        score = DIFFICULTY_SCORE.get(difficulty, 2)
        self.data["difficulty_scores"].append(score)

        self.save()

    # ── Inference ────────────────────────────────────────────────────────────

    @property
    def preferred_difficulty(self) -> str:
        """
        Infer the student's current level from a rolling average of their
        last ROLLING_WINDOW interactions.
        """
        scores = self.data["difficulty_scores"]
        if not scores:
            return "intermediate"

        recent   = scores[-ROLLING_WINDOW:]
        avg      = sum(recent) / len(recent)
        rounded  = round(avg)
        clamped  = max(1, min(3, rounded))
        return SCORE_LABEL[clamped]

    @property
    def top_topics(self) -> list[str]:
        """Return the 3 most frequently studied topics."""
        counts = self.data["topic_counts"]
        return [t for t, _ in Counter(counts).most_common(3)]

    @property
    def recommended_topics(self) -> list[str]:
        """
        Return topics with zero or low engagement, suggesting areas the
        student has not yet explored.
        """
        counts  = self.data["topic_counts"]
        # Sort all topics by engagement ascending; exclude 'general'
        ranked  = sorted(
            [t for t in ALL_TOPICS if t != "general"],
            key=lambda t: counts.get(t, 0),
        )
        # Return the lowest-engagement topics (up to 3)
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
