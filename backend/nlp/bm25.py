"""
NLP Document Assistant - Okapi BM25 Sparse Search Engine
Pure Python high-performance BM25 implementation with stopword filtering.
"""

import math
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
from backend.nlp.analyzer import STOPWORDS


class BM25Engine:
    """Okapi BM25 index and retrieval engine for document chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths = []
        self.doc_freqs = {}  # term -> number of documents containing term
        self.term_freqs = []  # list of Counter per document chunk
        self.chunk_ids = []
        self.chunks = []

    def fit(self, chunks: List[Any]):
        """Index a list of Chunk objects."""
        self.chunks = chunks
        self.corpus_size = len(chunks)
        self.doc_lengths = []
        self.doc_freqs = {}
        self.term_freqs = []
        self.chunk_ids = [c.chunk_id for c in chunks]

        if not chunks:
            self.avg_doc_len = 0.0
            return

        total_len = 0
        for chunk in chunks:
            tokens = self.tokenize(chunk.text)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            
            for term in tf.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avg_doc_len = total_len / max(1, self.corpus_size)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """Search indexed chunks using Okapi BM25 scoring."""
        if not self.chunks or self.corpus_size == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.doc_freqs:
                continue

            # Compute Robertson-Spärck Jones IDF
            n_q = self.doc_freqs[term]
            idf = math.log(1.0 + (self.corpus_size - n_q + 0.5) / (n_q + 0.5))

            for idx in range(self.corpus_size):
                tf = self.term_freqs[idx].get(term, 0)
                if tf == 0:
                    continue

                doc_len = self.doc_lengths[idx]
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                numerator = tf * (self.k1 + 1.0)
                scores[idx] += idf * (numerator / denominator)

        # Normalize scores to 0.0 - 1.0
        max_score = max(scores) if scores else 1.0
        ranked = []
        for idx, score in enumerate(scores):
            if score > 0.0:
                normalized = round(score / max(1.0, max_score), 4)
                ranked.append((self.chunks[idx], normalized))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize, lowercase, and filter stopwords."""
        words = re.findall(r"\b[A-Za-z0-9_-]{2,}\b", text.lower())
        return [w for w in words if w not in STOPWORDS]
