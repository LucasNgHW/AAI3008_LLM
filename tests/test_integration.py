"""
tests/test_integration.py
--------------------------
Integration tests for the NLP Learning Assistant pipeline.

All external I/O (Qdrant, ML models, Gemini API) is patched with lightweight
stubs so the suite runs without any installed ML dependencies.

Run with:
    python tests/test_integration.py
or:
    python -m pytest tests/ -v
"""

import sys
import os
import json
import time
import tempfile
import types
import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Stub all heavy deps before any project module is imported ─────────────────

def _stub(name, **attrs):
    m = sys.modules.get(name) or types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

_stub("sentence_transformers", SentenceTransformer=MagicMock)
_stub("sentence_transformers.cross_encoder", CrossEncoder=MagicMock)
_stub("qdrant_client", QdrantClient=MagicMock)
_stub("qdrant_client.models",
      Filter=MagicMock, FieldCondition=MagicMock,
      MatchValue=MagicMock, Range=MagicMock,
      Distance=MagicMock, VectorParams=MagicMock,
      PointStruct=MagicMock, PayloadSchemaType=MagicMock)


class _SimpleSplitter:
    def __init__(self, chunk_size=800, chunk_overlap=150, separators=None, **kw):
        self.chunk_size = chunk_size

    def split_text(self, text):
        return [
            text[i:i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
            if text[i:i + self.chunk_size].strip()
        ]

_stub("langchain_text_splitters", RecursiveCharacterTextSplitter=_SimpleSplitter)


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
        self.assertEqual(infer_topic("BPE and WordPiece are subword tokenisation algorithms."), "tokenisation")

    def test_infer_topic_transformers(self):
        from pipeline.chunker import infer_topic
        self.assertEqual(infer_topic("The transformer uses multi-head self-attention and positional encoding."), "transformers")

    def test_infer_topic_general_fallback(self):
        from pipeline.chunker import infer_topic
        self.assertEqual(infer_topic("The quick brown fox jumps over the lazy dog."), "general")

    def test_infer_difficulty_beginner(self):
        from pipeline.chunker import infer_difficulty
        self.assertEqual(infer_difficulty("Introduction to NLP: a basic overview and simple examples."), "beginner")

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
        doc = {"text": "Tokenisation splits text into tokens. " * 60,
               "source": "wk2_tokenisation.pdf", "content_type": "pdf"}
        chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertIn("text",     c)
            self.assertIn("metadata", c)
            self.assertEqual(c["metadata"]["week"], 2)

    def test_chunk_document_empty_text_returns_empty(self):
        from pipeline.chunker import chunk_document
        self.assertEqual(
            chunk_document({"text": "   ", "source": "empty.pdf", "content_type": "pdf"}), []
        )

    def test_chunk_documents_aggregates(self):
        from pipeline.chunker import chunk_documents
        docs = [
            {"text": "BERT uses masked language modelling. " * 30, "source": "wk3.pdf", "content_type": "pdf"},
            {"text": "Sentiment analysis classifies polarity. " * 30, "source": "wk4.pdf", "content_type": "pdf"},
        ]
        self.assertGreater(len(chunk_documents(docs, chunk_size=200, chunk_overlap=20)), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 2. USER PROFILE
# ══════════════════════════════════════════════════════════════════════════════

class TestUserProfile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from personalisation import user_profile as _up
        self._up = _up

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
            self.assertEqual(p.data["topic_counts"], {})
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
        from personalisation.user_profile import UserProfile
        (Path(self.tmpdir) / "bad.json").write_text("{ bad json", encoding="utf-8")
        with patch.object(self._up, "PROFILES_DIR", self.tmpdir):
            p = UserProfile("bad")
        self.assertEqual(p.preferred_difficulty, "intermediate")
        self.assertEqual(p.data["query_history"], [])

    def test_partial_profile_migrated(self):
        from personalisation.user_profile import UserProfile
        (Path(self.tmpdir) / "old.json").write_text(
            json.dumps({"user_id": "old", "query_history": ["hi"]}), encoding="utf-8"
        )
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
# 3. EMBEDDER — warmup
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbedder(unittest.TestCase):

    def test_warmup_calls_encode(self):
        import pipeline.embedder as emb_mod
        mock_model = MagicMock()
        with patch.object(emb_mod, "get_model", return_value=mock_model):
            emb_mod.warmup()
        mock_model.encode.assert_called_once()
        args = mock_model.encode.call_args
        self.assertEqual(args[0][0], ["warmup"])

    def test_embed_query_uses_batch_size_1(self):
        """embed_query must pass batch_size=1 to skip padding overhead."""
        import pipeline.embedder as emb_mod
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, 384))
        with patch.object(emb_mod, "get_model", return_value=mock_model):
            emb_mod.embed_query("hello")
        call_kwargs = mock_model.encode.call_args.kwargs
        self.assertEqual(call_kwargs.get("batch_size"), 1)
        self.assertFalse(call_kwargs.get("show_progress_bar", True))


# ══════════════════════════════════════════════════════════════════════════════
# 4. RETRIEVER
# ══════════════════════════════════════════════════════════════════════════════

class TestRetriever(unittest.TestCase):

    def _mock_point(self, text="token text", topic="tokenisation",
                    difficulty="intermediate", score=0.9):
        p = MagicMock()
        p.score = score
        p.payload = {"text": text, "topic": topic, "difficulty": difficulty,
                     "content_type": "pdf", "week": 1,
                     "source": "/data/lec1.pdf", "slide": None}
        return p

    def test_retrieve_returns_flat_dicts(self):
        import numpy as np, rag.retriever as ret
        mock_client = MagicMock()
        r = MagicMock(); r.points = [self._mock_point()]
        mock_client.query_points.return_value = r
        with patch.object(ret, "get_client",  return_value=mock_client), \
             patch.object(ret, "embed_query", return_value=np.zeros(384)):
            results = ret.retrieve("tokenisation?", top_k=3)
        self.assertEqual(len(results), 1)
        for key in ("text", "score", "topic", "difficulty", "source"):
            self.assertIn(key, results[0])

    def test_no_auto_difficulty_filter(self):
        """retrieve_with_context must NOT inject difficulty from profile."""
        import numpy as np, rag.retriever as ret
        mock_client = MagicMock()
        r = MagicMock(); r.points = []
        mock_client.query_points.return_value = r
        with patch.object(ret, "get_client",  return_value=mock_client), \
             patch.object(ret, "embed_query", return_value=np.zeros(384)):
            ret.retrieve_with_context("transformers",
                                      user_profile={"preferred_difficulty": "beginner"})
        self.assertIsNone(mock_client.query_points.call_args.kwargs.get("query_filter"))

    def test_query_augmentation_user_turns_only(self):
        import numpy as np, rag.retriever as ret
        captured = {}
        def fake_embed(q):
            captured["query"] = q
            return np.zeros(384)
        mock_client = MagicMock()
        r = MagicMock(); r.points = []
        mock_client.query_points.return_value = r
        history = [
            {"role": "user",      "content": "tell me about BERT"},
            {"role": "assistant", "content": "BERT is a transformer model..."},
        ]
        with patch.object(ret, "get_client",  return_value=mock_client), \
             patch.object(ret, "embed_query", side_effect=fake_embed):
            ret.retrieve_with_context("how does it work?",
                                      conversation_history=history)
        self.assertIn("BERT", captured["query"])
        self.assertIn("how does it work?", captured["query"])
        self.assertNotIn("transformer model...", captured["query"])


# ══════════════════════════════════════════════════════════════════════════════
# 5. RERANKER — cache behaviour + warmup
# ══════════════════════════════════════════════════════════════════════════════

class TestReranker(unittest.TestCase):

    def setUp(self):
        # Clear the module-level cache before each test
        import rag.reranker as rr
        rr.clear_cache()

    def test_rerank_sorts_descending(self):
        import rag.reranker as rr
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.3, 0.9, 0.1, 0.7]
        with patch.object(rr, "get_reranker", return_value=mock_ce):
            result = rr.rerank("query", [make_chunk(text=f"chunk {i}") for i in range(4)], top_n=3)
        self.assertEqual(len(result), 3)
        scores = [r["rerank_score"] for r in result]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(result[0]["rerank_score"], 0.9)

    def test_rerank_does_not_mutate_input(self):
        import rag.reranker as rr
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.5, 0.8]
        original = [make_chunk(text="a"), make_chunk(text="b")]
        with patch.object(rr, "get_reranker", return_value=mock_ce):
            rr.rerank("q", original, top_n=2)
        self.assertNotIn("rerank_score", original[0])

    def test_rerank_empty_input(self):
        import rag.reranker as rr
        self.assertEqual(rr.rerank("q", []), [])

    def test_rerank_cache_hit_skips_model(self):
        """Second identical call must not invoke the cross-encoder again."""
        import rag.reranker as rr
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.7, 0.4]
        chunks = [make_chunk(text="a"), make_chunk(text="b")]
        with patch.object(rr, "get_reranker", return_value=mock_ce):
            r1 = rr.rerank("same query", chunks, top_n=2)
            r2 = rr.rerank("same query", chunks, top_n=2)
        # predict should have been called exactly once despite two rerank calls
        self.assertEqual(mock_ce.predict.call_count, 1)
        self.assertEqual(r1[0]["rerank_score"], r2[0]["rerank_score"])

    def test_rerank_cache_miss_on_different_query(self):
        """Different queries must each invoke the model."""
        import rag.reranker as rr
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.5]
        chunks = [make_chunk(text="x")]
        with patch.object(rr, "get_reranker", return_value=mock_ce):
            rr.rerank("query A", chunks, top_n=1)
            rr.rerank("query B", chunks, top_n=1)
        self.assertEqual(mock_ce.predict.call_count, 2)

    def test_rerank_cache_returns_copies(self):
        """Mutating a cached result must not affect the next call."""
        import rag.reranker as rr
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.9]
        chunks = [make_chunk(text="z")]
        with patch.object(rr, "get_reranker", return_value=mock_ce):
            r1 = rr.rerank("q", chunks, top_n=1)
            r1[0]["rerank_score"] = -999   # mutate returned copy
            r2 = rr.rerank("q", chunks, top_n=1)
        self.assertAlmostEqual(r2[0]["rerank_score"], 0.9)

    def test_warmup_calls_predict(self):
        import rag.reranker as rr
        mock_ce = MagicMock()
        with patch.object(rr, "get_reranker", return_value=mock_ce):
            rr.warmup()
        mock_ce.predict.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# 6. GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerator(unittest.TestCase):

    def test_build_system_prompt_difficulty_injection(self):
        from rag.generator import build_system_prompt
        prompt = build_system_prompt({"preferred_difficulty": "advanced", "top_topics": ["transformers"]})
        self.assertIn("advanced", prompt)
        self.assertIn("transformers", prompt)

    def test_build_system_prompt_defaults(self):
        from rag.generator import build_system_prompt
        self.assertIn("intermediate", build_system_prompt(None))

    def test_build_user_prompt_includes_context(self):
        from rag.generator import build_user_prompt
        chunks = [make_chunk(text="BPE is a subword tokenisation method.")]
        prompt = build_user_prompt("what is BPE?", chunks)
        self.assertIn("BPE is a subword tokenisation method.", prompt)
        self.assertIn("<context>", prompt)

    def test_build_user_prompt_uses_basename(self):
        from rag.generator import build_user_prompt
        chunks = [make_chunk(source="/home/user/secret/data/raw/lec1.pdf")]
        prompt = build_user_prompt("q", chunks)
        self.assertNotIn("/home/user/secret", prompt)
        self.assertIn("lec1.pdf", prompt)

    def test_build_user_prompt_no_history(self):
        from rag.generator import build_user_prompt
        self.assertNotIn("<conversation_history>",
                         build_user_prompt("hello", [], conversation_history=None))

    def test_build_user_prompt_includes_history(self):
        from rag.generator import build_user_prompt
        history = [
            {"role": "user",      "content": "tell me about BERT"},
            {"role": "assistant", "content": "BERT stands for..."},
        ]
        prompt = build_user_prompt("what about GPT?", [], conversation_history=history)
        self.assertIn("<conversation_history>", prompt)
        self.assertIn("tell me about BERT", prompt)

    @patch("rag.generator.get_client")
    def test_generate_answer_calls_gemini(self, mock_get_client):
        # generate_answer now delegates to stream_answer which uses
        # generate_content_stream — mock that to yield one chunk.
        mock_client = MagicMock()
        mock_chunk  = MagicMock()
        mock_chunk.text = "Tokenisation splits text into tokens."
        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])
        mock_get_client.return_value = mock_client

        from rag.generator import generate_answer
        answer = generate_answer("what is tokenisation?", chunks=[make_chunk()],
                                 user_profile={"preferred_difficulty": "beginner", "top_topics": []})
        self.assertEqual(answer, "Tokenisation splits text into tokens.")
        mock_client.models.generate_content_stream.assert_called_once()

    @patch("rag.generator.get_client")
    def test_generate_answer_empty_chunks(self, mock_get_client):
        mock_client = MagicMock()
        mock_chunk  = MagicMock()
        mock_chunk.text = "No context available."
        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])
        mock_get_client.return_value = mock_client

        from rag.generator import generate_answer
        answer = generate_answer("question", chunks=[])
        self.assertIsInstance(answer, str)

    @patch("rag.generator.get_client")
    def test_stream_answer_yields_fragments(self, mock_get_client):
        """stream_answer must yield individual text fragments, not one big string."""
        mock_client  = MagicMock()
        fragments    = ["Self-attention ", "is a mechanism ", "for weighing tokens."]
        mock_chunks  = [MagicMock(text=t) for t in fragments]
        mock_client.models.generate_content_stream.return_value = iter(mock_chunks)
        mock_get_client.return_value = mock_client

        from rag.generator import stream_answer
        result = list(stream_answer("explain self-attention", chunks=[make_chunk()]))
        self.assertEqual(result, fragments)
        self.assertEqual("".join(result), "Self-attention is a mechanism for weighing tokens.")


# ══════════════════════════════════════════════════════════════════════════════
# 7. END-TO-END PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline(unittest.TestCase):

    def setUp(self):
        import rag.reranker as rr
        rr.clear_cache()

    def test_full_rag_pipeline(self):
        import numpy as np
        import rag.retriever as ret_mod
        import rag.reranker  as rr_mod
        import rag.generator as gen_mod

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

        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.95, 0.80, 0.70, 0.60, 0.50]

        mock_llm    = MagicMock()
        mock_chunk  = MagicMock()
        mock_chunk.text = "Self-attention computes weighted representations."
        mock_llm.models.generate_content_stream.return_value = iter([mock_chunk])

        with patch.object(ret_mod, "get_client",   return_value=mock_qdrant), \
             patch.object(ret_mod, "embed_query",  return_value=np.zeros(384)), \
             patch.object(rr_mod,  "get_reranker", return_value=mock_ce), \
             patch.object(gen_mod, "get_client",   return_value=mock_llm):

            query   = "Explain self-attention in simple terms"
            history = [{"role": "user", "content": "what are transformers?"}]
            profile = {"preferred_difficulty": "beginner", "top_topics": ["transformers"]}

            chunks = ret_mod.retrieve_with_context(query, conversation_history=history,
                                                   user_profile=profile, top_k=15)
            chunks = rr_mod.rerank(query, chunks, top_n=5)
            answer = gen_mod.generate_answer(query, chunks, user_profile=profile,
                                             conversation_history=history)

        self.assertEqual(len(chunks), 5)
        self.assertIn("rerank_score", chunks[0])
        self.assertGreater(chunks[0]["rerank_score"], chunks[-1]["rerank_score"])
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        mock_llm.models.generate_content_stream.assert_called_once()
        self.assertIn("gemini", mock_llm.models.generate_content_stream.call_args.kwargs.get("model", ""))

    def test_full_pipeline_cache_skips_rerank_on_repeat(self):
        """The second call with the same query should skip the cross-encoder."""
        import numpy as np
        import rag.retriever as ret_mod
        import rag.reranker  as rr_mod

        mock_qdrant = MagicMock()
        point = MagicMock(); point.score = 0.7
        point.payload = {"text": "t", "topic": "transformers", "difficulty": "intermediate",
                         "content_type": "pdf", "week": 1, "source": "lec.pdf", "slide": None}
        r = MagicMock(); r.points = [point]
        mock_qdrant.query_points.return_value = r

        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.8]

        with patch.object(ret_mod, "get_client",   return_value=mock_qdrant), \
             patch.object(ret_mod, "embed_query",  return_value=np.zeros(384)), \
             patch.object(rr_mod,  "get_reranker", return_value=mock_ce):

            chunks1 = ret_mod.retrieve_with_context("same q", top_k=5)
            rr_mod.rerank("same q", chunks1, top_n=1)

            chunks2 = ret_mod.retrieve_with_context("same q", top_k=5)
            rr_mod.rerank("same q", chunks2, top_n=1)

        self.assertEqual(mock_ce.predict.call_count, 1)

    def test_chunker_metadata_contract(self):
        """All keys required by index_chunks must be present in chunk metadata."""
        from pipeline.chunker import chunk_document
        doc = {"text": "Transformers use self-attention. " * 20,
               "source": "wk3_transformers.pdf", "content_type": "pdf"}
        required = {"source", "topic", "difficulty", "content_type",
                    "week", "slide", "cell_type", "chunk_index"}
        for chunk in chunk_document(doc):
            missing = required - chunk["metadata"].keys()
            self.assertFalse(missing, f"Missing metadata keys: {missing}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATE SCRIPT
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluate(unittest.TestCase):

    def _get_ev(self):
        for k in list(sys.modules.keys()):
            if k in ("evaluate", "rag.retriever", "rag.reranker",
                     "pipeline.indexer", "pipeline.embedder", "pipeline.chunker"):
                del sys.modules[k]
        import evaluate as ev
        return ev

    def test_is_relevant_by_source(self):
        ev = self._get_ev()
        self.assertTrue(ev.is_relevant(
            {"source": "/data/raw/AAI3008_Lec2_TextProcessing.pdf", "topic": "general"},
            {"relevant_sources": ["Lec2_TextProcessing"], "relevant_topics": []}
        ))

    def test_is_relevant_by_topic(self):
        ev = self._get_ev()
        self.assertTrue(ev.is_relevant(
            {"source": "/data/raw/other.pdf", "topic": "tokenisation"},
            {"relevant_sources": [], "relevant_topics": ["tokenisation"]}
        ))

    def test_is_not_relevant(self):
        ev = self._get_ev()
        self.assertFalse(ev.is_relevant(
            {"source": "/data/raw/lec5.pdf", "topic": "sentiment"},
            {"relevant_sources": ["lec1"], "relevant_topics": ["embeddings"]}
        ))

    def test_evaluate_mrr_perfect(self):
        ev = self._get_ev()
        eval_set = [{"question": "tokenisation?",
                     "relevant_sources": ["lec2.pdf"], "relevant_topics": ["tokenisation"]}]
        with patch.object(ev, "retrieve", return_value=[
            {"source": "lec2.pdf", "topic": "tokenisation", "text": "..."}
        ]):
            m = ev.evaluate(eval_set, top_k=5, verbose=False)
        self.assertAlmostEqual(m["mrr"], 1.0)
        self.assertAlmostEqual(m["recall@5"], 1.0)

    def test_evaluate_mrr_miss(self):
        ev = self._get_ev()
        eval_set = [{"question": "tokenisation?",
                     "relevant_sources": ["lec2.pdf"], "relevant_topics": ["tokenisation"]}]
        with patch.object(ev, "retrieve", return_value=[
            {"source": "lec9.pdf", "topic": "sentiment", "text": "..."}
        ]):
            m = ev.evaluate(eval_set, top_k=5, verbose=False)
        self.assertAlmostEqual(m["mrr"], 0.0)
        self.assertAlmostEqual(m["recall@5"], 0.0)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)