"""
NLP Document Assistant - Dense Vector Store
Supports both Local LSA/TF-IDF semantic embeddings (100% offline) and Google Gemini text-embedding-004.
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

try:
    from google import genai
except ImportError:
    genai = None


class VectorStore:
    """Manages dense vector embeddings, cosine indexing, and semantic retrieval."""

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.chunks: List[Any] = []
        self.embeddings: Optional[np.ndarray] = None
        self.local_vectorizer: Optional[TfidfVectorizer] = None
        self.local_svd: Optional[TruncatedSVD] = None
        self.embedding_model = "local-semantic"

        if self.gemini_api_key and genai:
            try:
                self.client = genai.Client(api_key=self.gemini_api_key)
                self.embedding_model = "text-embedding-004"
            except Exception:
                self.client = None
        else:
            self.client = None

    def set_api_key(self, api_key: str):
        """Update API key dynamically at runtime."""
        self.gemini_api_key = api_key
        if api_key and genai:
            try:
                self.client = genai.Client(api_key=api_key)
                self.embedding_model = "text-embedding-004"
            except Exception:
                self.client = None
                self.embedding_model = "local-semantic"
        else:
            self.client = None
            self.embedding_model = "local-semantic"

    def fit(self, chunks: List[Any]):
        """Embed and index all document chunks."""
        self.chunks = chunks
        if not chunks:
            self.embeddings = None
            return

        texts = [c.text for c in chunks]

        # Check if Gemini neural embeddings are available
        if self.client and self.gemini_api_key:
            try:
                vecs = self._embed_with_gemini(texts)
                self.embeddings = np.array(vecs, dtype=np.float32)
                self._normalize_embeddings()
                self.embedding_model = "text-embedding-004 (Google GenAI)"
                return
            except Exception as e:
                print(f"[VectorStore] Gemini embedding failed ({e}), falling back to local semantic model.")

        # Local semantic vectorizer (TF-IDF + LSA SVD)
        self._fit_local_embeddings(texts)
        self.embedding_model = "local-semantic (TF-IDF + LSA SVD)"

    def _fit_local_embeddings(self, texts: List[str]):
        """Generate normalized semantic dense vectors using TF-IDF and SVD."""
        max_df = 1.0 if len(texts) <= 3 else 0.95
        self.local_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=max_df,
            min_df=1,
            sublinear_tf=True
        )
        tfidf_matrix = self.local_vectorizer.fit_transform(texts)

        # If we have enough chunks, apply SVD for latent semantic representation
        n_features = tfidf_matrix.shape[1]
        n_samples = len(texts)
        n_components = min(128, n_features - 1, n_samples - 1)

        if n_components >= 4:
            self.local_svd = TruncatedSVD(n_components=n_components, random_state=42)
            dense_vectors = self.local_svd.fit_transform(tfidf_matrix)
        else:
            self.local_svd = None
            dense_vectors = tfidf_matrix.toarray()

        self.embeddings = np.array(dense_vectors, dtype=np.float32)
        self._normalize_embeddings()

    def _embed_with_gemini(self, texts: List[str]) -> List[List[float]]:
        """Call Gemini text-embedding-004 API."""
        embeddings = []
        # Batch in chunks of 20
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.models.embed_content(
                model="text-embedding-004",
                contents=batch,
            )
            for emb in response.embeddings:
                embeddings.append(emb.values)
        return embeddings

    def _normalize_embeddings(self):
        """L2 normalize rows of embedding matrix for fast cosine similarity dot product."""
        if self.embeddings is not None and len(self.embeddings) > 0:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings = self.embeddings / norms

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """Retrieve most similar chunks using cosine similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_vec = self._embed_query(query)
        if query_vec is None:
            return []

        # Cosine similarity is dot product when vectors are L2-normalized
        sims = np.dot(self.embeddings, query_vec)
        # Clip between 0 and 1
        sims = np.clip(sims, 0.0, 1.0)

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score > 0.01:
                results.append((self.chunks[idx], round(score, 4)))

        return results

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        """Embed a single query string."""
        if self.client and self.gemini_api_key and "text-embedding-004" in self.embedding_model:
            try:
                res = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=query
                )
                vec = np.array(res.embeddings[0].values, dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / (norm if norm > 0 else 1.0)
            except Exception:
                pass

        # Fallback local
        if self.local_vectorizer is None:
            return None

        tfidf_vec = self.local_vectorizer.transform([query])
        if self.local_svd:
            dense_vec = self.local_svd.transform(tfidf_vec)[0]
        else:
            dense_vec = tfidf_vec.toarray()[0]

        norm = np.linalg.norm(dense_vec)
        if norm > 0:
            dense_vec = dense_vec / norm
        return dense_vec
