"""
NLP Document Assistant - Hybrid Retriever with Reciprocal Rank Fusion (RRF)
Combines Okapi BM25 sparse lexical search and Dense Vector semantic embeddings.
"""

from typing import List, Dict, Any, Tuple
from backend.nlp.bm25 import BM25Engine
from backend.nlp.vector_store import VectorStore


class RetrievedResult:
    """Encapsulates retrieved chunk with score attribution and provenance."""

    def __init__(
        self,
        chunk: Any,
        fused_score: float,
        bm25_score: float = 0.0,
        vector_score: float = 0.0,
        bm25_rank: int = -1,
        vector_rank: int = -1,
        match_type: str = "hybrid"
    ):
        self.chunk = chunk
        self.fused_score = fused_score
        self.bm25_score = bm25_score
        self.vector_score = vector_score
        self.bm25_rank = bm25_rank
        self.vector_rank = vector_rank
        self.match_type = match_type

    def to_dict(self) -> Dict[str, Any]:
        data = self.chunk.to_dict()
        data.update({
            "fused_score": round(self.fused_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "vector_score": round(self.vector_score, 4),
            "bm25_rank": self.bm25_rank,
            "vector_rank": self.vector_rank,
            "match_type": self.match_type
        })
        return data


class HybridRetriever:
    """Orchestrates multi-index retrieval using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, bm25_engine: BM25Engine, vector_store: VectorStore):
        self.bm25 = bm25_engine
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",  # "hybrid", "vector", "bm25"
        alpha: float = 0.5,     # weight for BM25 vs Vector in hybrid
        rrf_k: int = 60,
        min_score_threshold: float = 0.05
    ) -> List[RetrievedResult]:
        """Executes retrieval across BM25 and Vector indices and fuses rankings."""
        candidate_pool_size = max(top_k * 3, 15)

        if mode == "bm25":
            bm25_hits = self.bm25.search(query, top_k=top_k)
            return [
                RetrievedResult(
                    chunk=ch,
                    fused_score=score,
                    bm25_score=score,
                    bm25_rank=r + 1,
                    match_type="bm25_only"
                )
                for r, (ch, score) in enumerate(bm25_hits)
                if score >= min_score_threshold
            ]

        if mode == "vector":
            vector_hits = self.vector_store.search(query, top_k=top_k)
            return [
                RetrievedResult(
                    chunk=ch,
                    fused_score=score,
                    vector_score=score,
                    vector_rank=r + 1,
                    match_type="vector_only"
                )
                for r, (ch, score) in enumerate(vector_hits)
                if score >= min_score_threshold
            ]

        # Hybrid RRF Retrieval
        bm25_hits = self.bm25.search(query, top_k=candidate_pool_size)
        vector_hits = self.vector_store.search(query, top_k=candidate_pool_size)

        rrf_scores = {}  # chunk_id -> dict
        
        # Ingest BM25 rankings
        for rank, (chunk, score) in enumerate(bm25_hits, start=1):
            cid = chunk.chunk_id
            if cid not in rrf_scores:
                rrf_scores[cid] = {
                    "chunk": chunk,
                    "rrf": 0.0,
                    "bm25_score": score,
                    "vector_score": 0.0,
                    "bm25_rank": rank,
                    "vector_rank": -1
                }
            rrf_contribution = alpha * (1.0 / (rrf_k + rank))
            rrf_scores[cid]["rrf"] += rrf_contribution

        # Ingest Vector rankings
        for rank, (chunk, score) in enumerate(vector_hits, start=1):
            cid = chunk.chunk_id
            if cid not in rrf_scores:
                rrf_scores[cid] = {
                    "chunk": chunk,
                    "rrf": 0.0,
                    "bm25_score": 0.0,
                    "vector_score": score,
                    "bm25_rank": -1,
                    "vector_rank": rank
                }
            else:
                rrf_scores[cid]["vector_score"] = score
                rrf_scores[cid]["vector_rank"] = rank

            rrf_contribution = (1.0 - alpha) * (1.0 / (rrf_k + rank))
            rrf_scores[cid]["rrf"] += rrf_contribution

        # Normalize fused scores
        max_rrf = max((item["rrf"] for item in rrf_scores.values()), default=1.0)
        results = []

        for cid, item in rrf_scores.items():
            norm_score = item["rrf"] / max(1e-9, max_rrf)
            if norm_score < min_score_threshold:
                continue

            # Determine match provenance
            if item["bm25_rank"] != -1 and item["vector_rank"] != -1:
                match_type = "hybrid_consensus"
            elif item["bm25_rank"] != -1:
                match_type = "bm25_keyword"
            else:
                match_type = "vector_semantic"

            res = RetrievedResult(
                chunk=item["chunk"],
                fused_score=norm_score,
                bm25_score=item["bm25_score"],
                vector_score=item["vector_score"],
                bm25_rank=item["bm25_rank"],
                vector_rank=item["vector_rank"],
                match_type=match_type
            )
            results.append(res)

        # Sort by final fused score
        results.sort(key=lambda x: x.fused_score, reverse=True)
        return results[:top_k]
