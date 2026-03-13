"""
tests/test_integration.py
--------------------------
Integration tests for the NLP Learning Assistant pipeline.

These tests cover the full data flow without requiring live Qdrant,
or ML models — all external I/O is patched with lightweight stubs.

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_integration.py
"""

import sys
import os
import json
import tempfile
import types
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Stub all heavy deps before any project module is imported ─────────────────
# This prevents ImportError for packages not installed in the test environment.

def _stub(name, **attrs):
    m = sys.modules.get(name) or types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

_st = _stub("sentence_transformers", SentenceTransformer=MagicMock)
_st_ce = _stub("sentence_transformers.cross_encoder", CrossEncoder=MagicMock)

_qc = _stub("qdrant_client", QdrantClient=MagicMock)
_qm = _stub("qdrant_client.models",
            Filter=MagicMock, FieldCondition=MagicMock,
            MatchValue=MagicMock, Range=MagicMock,
            Distance=MagicMock, VectorParams=MagicMock,
            PointStruct=MagicMock, PayloadSchemaType=MagicMock)


class _SimpleSplitter:
    def __init__(self, chunk_size=800, chunk_overlap=150, separators=None, **kw):
        self.chunk_size = chunk_size
    def split_text(self, text):
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)
                if text[i:i+self.chunk_size].strip()]

_lts = _stub("langchain_text_splitters",
             RecursiveCharacterTextSplitter=_SimpleSplitter)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_chunk(text="NLP text about tokenisation and BPE.", topic="tokenisation",
               difficulty="intermediate", source="/data/raw/lec1.pdf", score=0.85):
    return {
        "text": text, "score": score, "topic": topic,
        "difficulty": difficulty, "content_type": "pdf",
        "week": 1, "source": source, "slide": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

class TestChunker(unittest.TestCase):


    def test_infer_topic_tokenisation(self):
        from pipeline.chunker import infer_topic
        result = infer_topic("BPE and WordPiece are subword tokenisation algorithms.")
        self.assertEqual(result, "tokenisation")

    def test_infer_topic_transformers(self):
        from pipeline.chunker import infer_topic
        result = infer_topic("The transformer uses multi-head self-attention and positional encoding.")
        self.assertEqual(result, "transformers")

    def test_infer_topic_general_fallback(self):
        from pipeline.chunker import infer_topic
        result = infer_topic("The quick brown fox jumps over the lazy dog.")
        self.assertEqual(result, "general")

    def test_infer_difficulty_beginner(self):
        from pipeline.chunker import infer_difficulty
        result = infer_difficulty("Introduction to NLP: a basic overview and simple examples.")
        self.assertEqual(result, "beginner")

    def test_infer_difficulty_advanced(self):
        from pipeline.chunker import infer_difficulty
        result = infer_difficulty(
            "The loss function uses cross-entropy with backpropagation. "
            "Fine-tuning with contrastive loss requires careful learning rate scheduling."
        )
        self.assertEqual(result, "advanced")

    def test_infer_week_from_filename(self):
        from pipeline.chunker import infer_week
        self.assertEqual(infer_week("lecture_03_transformers.pdf"), 3)
        self.assertEqual(infer_week("wk5_embeddings.pptx"), 5)
        self.assertEqual(infer_week("week-12-summary.pdf"), 12)
        self.assertIsNone(infer_week("introduction.pdf"))

    def test_chunk_document_splits_text(self):
        from pipeline.chunker import chunk_document
        doc = {
            "text": "Tokenisation splits text into tokens. " * 60,
            "source": "wk2_tokenisation.pdf",
            "content_type": "pdf",
        }
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 1, "Long text should produce multiple chunks")
        for c in chunks:
            self.assertIn("text",     c)
            self.assertIn("metadata", c)
            self.assertEqual(c["metadata"]["week"], 2)
            self.assertIn(c["metadata"]["topic"], list(__import__("pipeline.chunker", fromlist=["TOPIC_KEYWORDS"]).TOPIC_KEYWORDS.keys()) + ["general"])

    def test_chunk_document_empty_text_returns_empty(self):
        from pipeline.chunker import chunk_document
        chunks = chunk_document({"text": "   ", "source": "empty.pdf", "content_type": "pdf"})
        self.assertEqual(chunks, [])

    def test_chunk_documents_aggregates(self):
        from pipeline.chunker import chunk_documents
        docs = [
            {"text": "BERT uses masked language modelling. " * 30, "source": "wk3.pdf", "content_type": "pdf"},
            {"text": "Sentiment analysis classifies polarity. " * 30, "source": "wk4.pdf", "content_type": "pdf"},
        ]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 2. USER PROFILE
# ══════════════════════════════════════════════════════════════════════════════

class TestUserProfile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from personalisation import user_profile as _up
        self._up = _up

    def _profile(self, user_id):
        from personalisation.user_profile import UserProfile
        with patch.object(self._up, "PROFILES_DIR", self.tmpdir):
            p = UserProfile.__new__(UserProfile)
            p.user_id = user_id
            p.path = Path(self.tmpdir) / f"{user_id}.json"
            p.data = p._load()
        return p

    def test_blank_profile_defaults(self):
        from personalisation.user_profile import UserProfile
        with patch.object(self._up, "PROFILES_DIR", self.tmpdir):
            p = UserProfile("new_student")
        self.assertEqual(p.preferred_difficulty, "intermediate")
        self.assertEqual(p.top_topics, [])
        self.assertEqual(len(p.recommended_topics), 3)

    def test_log_interaction_updates_counts(self):
        from personalisation.user_profile import UserProfile
        with patch.object(self._up, "PROFILES_DIR", self.tmpdir):
            p = UserProfile("counts_test")
            self.assertEqual(p.data["topic_counts"], {}, "Should start blank")
            p.log_interaction("what is BERT?",         "transformers", "intermediate")
            p.log_interaction("explain attention",      "transformers", "advanced")
            p.log_interaction("what is tokenisation?", "tokenisation", "beginner")
            self.assertEqual(p.data["topic_counts"]["transformers"], 2)
            self.assertEqual(p.data["topic_counts"]["tokenisation"], 1)
            self.assertEqual(p.top_topics[0], "transformers")

    def test_difficulty_inference_rolling_window(self):
        from personalisation.user_profile import UserProfile
        with patch.object(self._up, "PROFILES_DIR", self.tmpdir):
            p = UserProfile("s2")
            for _ in range(8):
                p.log_interaction("q", "transformers", "advanced")
        self.assertEqual(p.preferred_difficulty, "advanced")

    def test_corrupted_json_returns_blank(self):
        """A corrupted profile file must not crash the app."""
        from personalisation.user_profile import UserProfile
        corrupt_path = Path(self.tmpdir) / "bad_student.json"
        corrupt_path.write_text("{ this is not json !!!", encoding="utf-8")
        with patch.object(self._up, "PROFILES_DIR", self.tmpdir):
            p = UserProfile("bad_student")
        self.assertEqual(p.preferred_difficulty, "intermediate")
        self.assertEqual(p.data["query_history"], [])

    def test_partial_profile_migrated(self):
        """Profile missing new keys should be backfilled with defaults."""
        from personalisation.user_profile import UserProfile
        partial = {"user_id": "old", "query_history": ["hello"]}
        partial_path = Path(self.tmpdir) / "old.json"
        partial_path.write_text(json.dumps(partial), encoding="utf-8")
        with patch.object(self._up, "PROFILES_DIR", self.tmpdir):
            p = UserProfile("old")
        self.assertIn("topic_counts",      p.data)
        self.assertIn("difficulty_scores", p.data)

    def test_to_dict_shape(self):
        from personalisation.user_profile import UserProfile
        with patch.object(self._up, "PROFILES_DIR", self.tmpdir):
            p = UserProfile("s3")
        d = p.to_dict()
        for key in ("user_id", "preferred_difficulty", "top_topics",
                    "recommended_topics", "total_interactions"):
            self.assertIn(key, d)


# ══════════════════════════════════════════════════════════════════════════════
# 3. RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

class TestRetriever(unittest.TestCase):

    def _mock_qdrant_result(self, text="token text", topic="tokenisation",
                             difficulty="intermediate", score=0.9):
        point = MagicMock()
        point.score = score
        point.payload = {
            "text": text, "topic": topic, "difficulty": difficulty,
            "content_type": "pdf", "week": 1, "source": "/data/lec1.pdf", "slide": None,
        }
        return point

    def test_retrieve_returns_flat_dicts(self):
        import numpy as np
        import rag.retriever as ret_mod

        mock_client = MagicMock()
        point = self._mock_qdrant_result()
        result_obj = MagicMock(); result_obj.points = [point]
        mock_client.query_points.return_value = result_obj

        with patch.object(ret_mod, "get_client",  return_value=mock_client), \
             patch.object(ret_mod, "embed_query", return_value=np.zeros(384)):
            results = ret_mod.retrieve("what is tokenisation?", top_k=3)

        self.assertEqual(len(results), 1)
        self.assertIn("text",  results[0])
        self.assertIn("score", results[0])
        self.assertIn("topic", results[0])

    def test_retrieve_with_context_no_auto_difficulty_filter(self):
        """retrieve_with_context must NOT inject difficulty from profile."""
        import numpy as np
        import rag.retriever as ret_mod

        mock_client = MagicMock()
        result_obj = MagicMock(); result_obj.points = []
        mock_client.query_points.return_value = result_obj

        with patch.object(ret_mod, "get_client",  return_value=mock_client), \
             patch.object(ret_mod, "embed_query", return_value=np.zeros(384)):
            ret_mod.retrieve_with_context(
                query="tell me about transformers",
                user_profile={"preferred_difficulty": "beginner"},
            )

        call_kwargs = mock_client.query_points.call_args.kwargs
        self.assertIsNone(call_kwargs.get("query_filter"))

    def test_retrieve_with_context_query_augmentation(self):
        """Query augmentation should prepend recent user turn text, user-only."""
        import numpy as np
        import rag.retriever as ret_mod

        captured = {}
        def fake_embed(query):
            captured["query"] = query
            return np.zeros(384)

        mock_client = MagicMock()
        result_obj = MagicMock(); result_obj.points = []
        mock_client.query_points.return_value = result_obj

        history = [
            {"role": "user",      "content": "tell me about BERT"},
            {"role": "assistant", "content": "BERT is a transformer model..."},
        ]

        with patch.object(ret_mod, "get_client",  return_value=mock_client), \
             patch.object(ret_mod, "embed_query", side_effect=fake_embed):
            ret_mod.retrieve_with_context(query="how does it work?",
                                          conversation_history=history)

        self.assertIn("BERT", captured["query"])
        self.assertIn("how does it work?", captured["query"])
        self.assertNotIn("transformer model...", captured["query"])


# ══════════════════════════════════════════════════════════════════════════════
# 4. RERANKER
# ══════════════════════════════════════════════════════════════════════════════

class TestReranker(unittest.TestCase):

    def test_rerank_sorts_descending(self):
        import rag.reranker as rr
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.3, 0.9, 0.1, 0.7]
        with patch.object(rr, "get_reranker", return_value=mock_ce):
            chunks = [make_chunk(text=f"chunk {i}") for i in range(4)]
            result = rr.rerank("query", chunks, top_n=3)
        self.assertEqual(len(result), 3)
        scores = [r["rerank_score"] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(result[0]["rerank_score"], 0.9)

    def test_rerank_does_not_mutate_input(self):
        import rag.reranker as rr
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.5, 0.8]
        with patch.object(rr, "get_reranker", return_value=mock_ce):
            original = [make_chunk(text="a"), make_chunk(text="b")]
            _ = rr.rerank("q", original, top_n=2)
        self.assertNotIn("rerank_score", original[0])
        self.assertNotIn("rerank_score", original[1])

    def test_rerank_empty_input(self):
        import rag.reranker as rr
        self.assertEqual(rr.rerank("q", []), [])


# ══════════════════════════════════════════════════════════════════════════════
# 5. GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerator(unittest.TestCase):

    def test_build_system_prompt_difficulty_injection(self):
        from rag.generator import build_system_prompt
        prompt = build_system_prompt({"preferred_difficulty": "advanced", "top_topics": ["transformers"]})
        self.assertIn("advanced", prompt)
        self.assertIn("transformers", prompt)

    def test_build_system_prompt_defaults(self):
        from rag.generator import build_system_prompt
        prompt = build_system_prompt(None)
        self.assertIn("intermediate", prompt)

    def test_build_user_prompt_includes_context(self):
        from rag.generator import build_user_prompt
        chunks = [make_chunk(text="BPE is a subword tokenisation method.")]
        prompt = build_user_prompt("what is BPE?", chunks)
        self.assertIn("BPE is a subword tokenisation method.", prompt)
        self.assertIn("<context>", prompt)

    def test_build_user_prompt_uses_basename_not_full_path(self):
        """Filesystem paths should not leak into the student-visible prompt."""
        from rag.generator import build_user_prompt
        chunks = [make_chunk(source="/home/user/secret/data/raw/lec1.pdf")]
        prompt = build_user_prompt("question", chunks)
        self.assertNotIn("/home/user/secret", prompt)
        self.assertIn("lec1.pdf", prompt)

    def test_build_user_prompt_conversation_history_user_only(self):
        from rag.generator import build_user_prompt
        history = [
            {"role": "user",      "content": "tell me about BERT"},
            {"role": "assistant", "content": "BERT stands for Bidirectional..."},
        ]
        prompt = build_user_prompt("what about GPT?", [], conversation_history=history)
        self.assertIn("tell me about BERT", prompt)
        self.assertIn("<conversation_history>", prompt)

    def test_build_user_prompt_no_history(self):
        from rag.generator import build_user_prompt
        prompt = build_user_prompt("hello", [], conversation_history=None)
        self.assertNotIn("<conversation_history>", prompt)

    @patch("rag.generator.get_client")
    def test_generate_answer_calls_gemini(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Tokenisation splits text into tokens."
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        from rag.generator import generate_answer
        answer = generate_answer(
            query="what is tokenisation?",
            chunks=[make_chunk()],
            user_profile={"preferred_difficulty": "beginner", "top_topics": []},
        )
        self.assertEqual(answer, "Tokenisation splits text into tokens.")
        mock_client.models.generate_content.assert_called_once()

    @patch("rag.generator.get_client")
    def test_generate_answer_empty_chunks_still_works(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "I don't have enough context."
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        from rag.generator import generate_answer
        answer = generate_answer("random question", chunks=[])
        self.assertIsInstance(answer, str)


# ══════════════════════════════════════════════════════════════════════════════
# 6. END-TO-END PIPELINE INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline(unittest.TestCase):
    """
    Smoke-tests the full retrieve → rerank → generate flow with all
    external I/O (Qdrant, embedding model, Claude API) mocked out.
    """

    def test_full_rag_pipeline(self):
        import numpy as np
        import rag.retriever as ret_mod
        import rag.reranker  as rr_mod
        import rag.generator as gen_mod

        # Stub Qdrant
        mock_qdrant = MagicMock()
        point = MagicMock()
        point.score = 0.88
        point.payload = {
            "text": "Self-attention allows the model to weigh token importance.",
            "topic": "transformers", "difficulty": "intermediate",
            "content_type": "pdf", "week": 3,
            "source": "/data/raw/lec3.pdf", "slide": None,
        }
        result_obj = MagicMock(); result_obj.points = [point] * 5
        mock_qdrant.query_points.return_value = result_obj

        # Stub reranker
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.95, 0.80, 0.70, 0.60, 0.50]

        # Stub Gemini
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Self-attention computes weighted token representations."
        mock_llm.models.generate_content.return_value = mock_response

        with patch.object(ret_mod, "get_client",    return_value=mock_qdrant), \
             patch.object(ret_mod, "embed_query",   return_value=np.zeros(384)), \
             patch.object(rr_mod,  "get_reranker",  return_value=mock_ce), \
             patch.object(gen_mod, "get_client",    return_value=mock_llm):

            query   = "Explain self-attention in simple terms"
            history = [{"role": "user", "content": "what are transformers?"}]
            profile = {"preferred_difficulty": "beginner", "top_topics": ["transformers"]}

            chunks  = ret_mod.retrieve_with_context(query, conversation_history=history,
                                                    user_profile=profile, top_k=15)
            chunks  = rr_mod.rerank(query, chunks, top_n=5)
            answer  = gen_mod.generate_answer(query, chunks, user_profile=profile,
                                               conversation_history=history)

        self.assertEqual(len(chunks), 5)
        self.assertIn("rerank_score", chunks[0])
        self.assertGreater(chunks[0]["rerank_score"], chunks[-1]["rerank_score"])
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

        mock_llm.models.generate_content.assert_called_once()
        call_args = mock_llm.models.generate_content.call_args
        self.assertIn("gemini", call_args.kwargs.get("model", ""))

    def test_chunker_to_embed_shape_contract(self):
        """chunk_document output must contain all keys expected by index_chunks()."""
        from pipeline.chunker import chunk_document
        doc = {
            "text": "Attention is all you need. Transformers use self-attention. " * 20,
            "source": "wk3_transformers.pdf",
            "content_type": "pdf",
        }
        chunks = chunk_document(doc)
        required_keys = {
            "source", "topic", "difficulty", "content_type",
            "week", "slide", "cell_type", "chunk_index",
        }
        for chunk in chunks:
            missing = required_keys - chunk["metadata"].keys()
            self.assertFalse(missing, f"Chunk missing metadata keys: {missing}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATE SCRIPT
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluate(unittest.TestCase):

    def _get_evaluate_module(self):
        import importlib, sys
        # Clear cached module so fresh import picks up stubs
        for k in list(sys.modules.keys()):
            if k in ("evaluate", "rag.retriever", "rag.reranker", "pipeline.indexer",
                     "pipeline.embedder", "pipeline.chunker"):
                del sys.modules[k]
        import evaluate as ev
        return ev

    def test_is_relevant_by_source(self):
        ev = self._get_evaluate_module()
        chunk = {"source": "/data/raw/AAI3008_Lec2_TextProcessing.pdf", "topic": "general"}
        item  = {"relevant_sources": ["Lec2_TextProcessing"], "relevant_topics": []}
        self.assertTrue(ev.is_relevant(chunk, item))

    def test_is_relevant_by_topic(self):
        ev = self._get_evaluate_module()
        chunk = {"source": "/data/raw/some_other_file.pdf", "topic": "tokenisation"}
        item  = {"relevant_sources": [], "relevant_topics": ["tokenisation"]}
        self.assertTrue(ev.is_relevant(chunk, item))

    def test_is_not_relevant(self):
        ev = self._get_evaluate_module()
        chunk = {"source": "/data/raw/lec5.pdf", "topic": "sentiment"}
        item  = {"relevant_sources": ["lec1"], "relevant_topics": ["embeddings"]}
        self.assertFalse(ev.is_relevant(chunk, item))

    def test_evaluate_mrr_perfect(self):
        """MRR = 1.0 when every query returns the relevant chunk first."""
        ev = self._get_evaluate_module()
        eval_set = [
            {"question": "What is tokenisation?",
             "relevant_sources": ["lec2.pdf"], "relevant_topics": ["tokenisation"]},
        ]
        with patch.object(ev, "retrieve", return_value=[
            {"source": "lec2.pdf", "topic": "tokenisation", "text": "..."},
        ]):
            metrics = ev.evaluate(eval_set, top_k=5, verbose=False)
        self.assertAlmostEqual(metrics["mrr"],      1.0)
        self.assertAlmostEqual(metrics["recall@5"], 1.0)

    def test_evaluate_mrr_miss(self):
        """MRR = 0 when no relevant chunk is ever returned."""
        ev = self._get_evaluate_module()
        eval_set = [
            {"question": "What is tokenisation?",
             "relevant_sources": ["lec2.pdf"], "relevant_topics": ["tokenisation"]},
        ]
        with patch.object(ev, "retrieve", return_value=[
            {"source": "lec9.pdf", "topic": "sentiment", "text": "..."},
        ]):
            metrics = ev.evaluate(eval_set, top_k=5, verbose=False)
        self.assertAlmostEqual(metrics["mrr"],      0.0)
        self.assertAlmostEqual(metrics["recall@5"], 0.0)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)