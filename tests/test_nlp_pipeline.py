"""
Test Suite for NLP RAG Document Assistant
Validates chunking, BM25 indexing, vector store, hybrid retrieval, and grounding.
"""

import unittest
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.nlp.text_extractor import DocumentExtractor
from backend.nlp.chunker import RecursiveChunker
from backend.nlp.analyzer import DocumentAnalyzer
from backend.nlp.bm25 import BM25Engine
from backend.nlp.vector_store import VectorStore
from backend.nlp.hybrid_retriever import HybridRetriever
from backend.rag.engine import RAGEngine


class TestNLPRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "The Transformer model uses Scaled Dot-Product Attention to replace recurrence. "
            "It computes attention as softmax(QK^T / sqrt(d_k))V. "
            "The multi-head attention mechanism splits the queries into 8 parallel attention heads. "
            "On the WMT 2014 translation benchmark, it achieved an unprecedented 28.4 BLEU score. "
            "Positional encodings are added using sine and cosine functions of different frequencies."
        )

    def test_text_extractor_clean(self):
        cleaned = DocumentExtractor._clean_text("Hello   World!\r\n\r\n\n\nTest.")
        self.assertIn("Hello   World!\n\nTest.", cleaned)

    def test_chunker(self):
        chunker = RecursiveChunker(chunk_size=120, chunk_overlap=30)
        chunks = chunker.split_document("doc1", "test.txt", self.sample_text)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].doc_id, "doc1")
        self.assertIn("Transformer", chunks[0].text)

    def test_analyzer(self):
        analysis = DocumentAnalyzer.analyze(self.sample_text, "test.txt")
        self.assertIn("keywords", analysis)
        self.assertIn("entities", analysis)
        self.assertIn("readability", analysis)
        self.assertIn("summary", analysis)
        self.assertGreater(len(analysis["keywords"]), 0)

    def test_bm25_search(self):
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
        chunks = chunker.split_document("doc1", "test.txt", self.sample_text)
        
        bm25 = BM25Engine()
        bm25.fit(chunks)
        results = bm25.search("BLEU score benchmark", top_k=2)
        self.assertGreater(len(results), 0)
        top_chunk, score = results[0]
        self.assertIn("BLEU", top_chunk.text)

    def test_vector_store(self):
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
        chunks = chunker.split_document("doc1", "test.txt", self.sample_text)
        
        vs = VectorStore()
        vs.fit(chunks)
        results = vs.search("mathematical attention formula softmax", top_k=2)
        self.assertGreater(len(results), 0)
        top_chunk, score = results[0]
        self.assertIn("attention", top_chunk.text.lower())

    def test_hybrid_retriever(self):
        chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
        chunks = chunker.split_document("doc1", "test.txt", self.sample_text)
        
        bm25 = BM25Engine()
        bm25.fit(chunks)
        vs = VectorStore()
        vs.fit(chunks)
        
        retriever = HybridRetriever(bm25, vs)
        results = retriever.retrieve("parallel attention heads mechanism", top_k=3, mode="hybrid")
        self.assertGreater(len(results), 0)
        self.assertIsNotNone(results[0].fused_score)

    def test_rag_engine_end_to_end(self):
        engine = RAGEngine()
        ingest_res = engine.ingest_file(self.sample_text.encode("utf-8"), "sample.txt")
        self.assertIn("doc_id", ingest_res)
        
        answer_res = engine.answer_query("What BLEU score did the model achieve?")
        self.assertIn("answer", answer_res)
        self.assertGreater(len(answer_res["citations"]), 0)
        self.assertEqual(answer_res["confidence"], "high")


if __name__ == "__main__":
    unittest.main()
