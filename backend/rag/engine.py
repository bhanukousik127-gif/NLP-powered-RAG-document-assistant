"""
NLP Document Assistant - RAG Pipeline Engine
Coordinates ingestion, chunking, indexing, hybrid retrieval, grounding, and response generation.
"""

import os
import uuid
import time
from typing import List, Dict, Any, Optional

from backend.nlp.text_extractor import DocumentExtractor
from backend.nlp.chunker import RecursiveChunker, Chunk
from backend.nlp.analyzer import DocumentAnalyzer
from backend.nlp.bm25 import BM25Engine
from backend.nlp.vector_store import VectorStore
from backend.nlp.hybrid_retriever import HybridRetriever, RetrievedResult
from backend.rag.providers import GeminiProvider, LocalNLPSynthesizer


SYSTEM_PROMPT = """You are RAGNova, an elite NLP-Powered Document Assistant and Research Companion.
Your mission is to provide accurate, comprehensive, and strictly evidence-grounded answers based ONLY on the provided document excerpts.

STRICT OPERATIONAL GUIDELINES:
1. GROUNDING & FIDELITY: Answer using ONLY facts directly mentioned in the Context. Do NOT speculate, infer without evidence, or bring outside assumptions.
2. CITATIONS: Cite the specific chunk and page for every factual assertion using the format: [SourceDoc, Page X, Chunk ID].
3. ANTI-HALLUCINATION: If the provided excerpts do not contain sufficient evidence to answer the question, clearly state: "The uploaded documents do not contain sufficient information to answer this question." Then specify what related topics ARE covered.
4. STRUCTURE: Use clear formatting, bullet points, headers, or bold text for key terms to make your answers easily readable.
5. TONE: Professional, concise, authoritative, and helpful.
"""


class RAGEngine:
    """End-to-end RAG orchestrator."""

    def __init__(self):
        # Configuration
        self.chunk_size = 600
        self.chunk_overlap = 120
        self.top_k = 4
        self.retrieval_mode = "hybrid"  # "hybrid", "vector", "bm25"
        self.alpha = 0.5                # 0.0 = pure vector, 1.0 = pure BM25
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = "gemini-2.5-flash"

        # State storage
        self.documents: Dict[str, Dict[str, Any]] = {}  # doc_id -> metadata & text
        self.chunks: List[Chunk] = []                   # flat list of all chunks
        self.doc_analyses: Dict[str, Dict[str, Any]] = {} # doc_id -> NLP analysis

        # Engines
        self.bm25 = BM25Engine()
        self.vector_store = VectorStore(self.gemini_api_key)
        self.retriever = HybridRetriever(self.bm25, self.vector_store)
        self._init_provider()

    def _init_provider(self):
        """Initialize LLM provider based on API key availability."""
        if self.gemini_api_key and self.gemini_api_key.strip():
            self.provider = GeminiProvider(self.gemini_api_key, self.gemini_model)
            self.provider_name = f"Google Gemini ({self.gemini_model})"
        else:
            self.provider = LocalNLPSynthesizer()
            self.provider_name = "Built-in Local NLP Synthesizer (Offline Mode)"

    def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update tuning parameters dynamically."""
        if "chunk_size" in settings:
            self.chunk_size = int(settings["chunk_size"])
        if "chunk_overlap" in settings:
            self.chunk_overlap = int(settings["chunk_overlap"])
        if "top_k" in settings:
            self.top_k = int(settings["top_k"])
        if "retrieval_mode" in settings:
            self.retrieval_mode = settings["retrieval_mode"]
        if "alpha" in settings:
            self.alpha = float(settings["alpha"])
        if "gemini_model" in settings:
            self.gemini_model = settings["gemini_model"]
        if "gemini_api_key" in settings:
            new_key = settings["gemini_api_key"].strip()
            self.gemini_api_key = new_key
            self.vector_store.set_api_key(new_key)
            self._init_provider()

        # Re-chunk and re-index if chunk parameters changed
        if "chunk_size" in settings or "chunk_overlap" in settings:
            self._reindex_all_documents()

        return self.get_settings()

    def get_settings(self) -> Dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "retrieval_mode": self.retrieval_mode,
            "alpha": self.alpha,
            "gemini_model": self.gemini_model,
            "has_api_key": bool(self.gemini_api_key),
            "provider_name": self.provider_name,
            "embedding_model": self.vector_store.embedding_model,
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks)
        }

    def ingest_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Ingest a document from raw bytes."""
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        # 1. Extract text and base metadata
        extracted = DocumentExtractor.extract_from_bytes(file_bytes, filename)
        text = extracted["text"]
        metadata = extracted["metadata"]
        metadata["doc_id"] = doc_id
        metadata["upload_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # 2. Deep NLP Analysis
        analysis = DocumentAnalyzer.analyze(text, filename)
        self.doc_analyses[doc_id] = analysis

        # 3. Store document
        self.documents[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "text": text,
            "metadata": metadata,
            "analysis": analysis
        }

        # 4. Chunk document
        chunker = RecursiveChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        new_chunks = chunker.split_document(doc_id, filename, text)
        
        # Add to global chunk list
        self.chunks.extend(new_chunks)

        # 5. Update index
        self._reindex_all_documents()

        return {
            "doc_id": doc_id,
            "filename": filename,
            "chunks_count": len(new_chunks),
            "metadata": metadata,
            "analysis": analysis
        }

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and rebuild indices."""
        if doc_id not in self.documents:
            return False

        del self.documents[doc_id]
        if doc_id in self.doc_analyses:
            del self.doc_analyses[doc_id]

        # Re-filter chunks
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        self._reindex_all_documents()
        return True

    def _reindex_all_documents(self):
        """Re-chunks and updates BM25 and vector indices."""
        # If chunks need to be recomputed for all docs
        all_chunks = []
        chunker = RecursiveChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        for doc_id, doc in self.documents.items():
            doc_chunks = chunker.split_document(doc_id, doc["filename"], doc["text"])
            all_chunks.extend(doc_chunks)

        self.chunks = all_chunks

        # Update sparse and dense indices
        self.bm25.fit(self.chunks)
        self.vector_store.fit(self.chunks)

    def answer_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Processes user query through retrieval, grounding, and LLM generation."""
        start_time = time.time()
        
        if not self.chunks:
            return {
                "answer": "No documents have been uploaded yet. Please upload a PDF, DOCX, TXT, or JSON document to begin querying.",
                "citations": [],
                "retrieved_chunks": [],
                "latency_ms": 0,
                "confidence": "none",
                "suggested_questions": []
            }

        # 1. Retrieve most relevant chunks
        retrieved_results: List[RetrievedResult] = self.retriever.retrieve(
            query=query,
            top_k=self.top_k,
            mode=self.retrieval_mode,
            alpha=self.alpha,
            min_score_threshold=0.04
        )

        # Anti-hallucination guardrail: evaluate top relevance score
        if not retrieved_results:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "answer": "The uploaded documents do not contain relevant information or evidence to answer this question. Try rephrasing your search or querying a topic covered in the uploaded files.",
                "citations": [],
                "retrieved_chunks": [],
                "latency_ms": latency_ms,
                "confidence": "low",
                "suggested_questions": self._get_global_suggested_questions()
            }

        # 2. Build Grounded Context Prompt
        context_blocks = []
        citations_data = []

        for idx, res in enumerate(retrieved_results, start=1):
            chunk = res.chunk
            context_blocks.append(
                f"[Chunk ID: {chunk.chunk_id}]\n"
                f"[Source: {chunk.filename}]\n"
                f"[Page: {chunk.page_num}]\n"
                f"{chunk.text}\n"
            )
            citations_data.append({
                "citation_id": f"[{chunk.filename}, Page {chunk.page_num}, {chunk.chunk_id}]",
                "chunk_id": chunk.chunk_id,
                "filename": chunk.filename,
                "page_num": chunk.page_num,
                "section_title": chunk.section_title,
                "snippet": chunk.text[:220] + "..." if len(chunk.text) > 220 else chunk.text,
                "full_text": chunk.text,
                "score": res.fused_score,
                "match_type": res.match_type
            })

        context_str = "\n".join(context_blocks)

        # Format prompt with conversational history if provided
        history_context = ""
        if conversation_history:
            turns = []
            for h in conversation_history[-3:]:
                role = "User" if h.get("role") == "user" else "Assistant"
                turns.append(f"{role}: {h.get('content', '')}")
            if turns:
                history_context = "PREVIOUS CONVERSATION HISTORY:\n" + "\n".join(turns) + "\n\n"

        prompt = (
            f"{history_context}"
            f"=== DOCUMENT CONTEXT EXCERPTS ===\n"
            f"{context_str}\n"
            f"=================================\n\n"
            f"USER QUERY: {query}\n\n"
            f"Provide a thorough, grounded answer directly supported by the context above. "
            f"Cite specific chunks using [SourceDoc, Page X, Chunk ID]."
        )

        # 3. Generate Answer
        raw_answer = self.provider.generate(prompt=prompt, system_instruction=SYSTEM_PROMPT)

        latency_ms = int((time.time() - start_time) * 1000)

        # Determine confidence score based on top retrieved score
        top_score = retrieved_results[0].fused_score if retrieved_results else 0.0
        if top_score > 0.45:
            confidence = "high"
        elif top_score > 0.15:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "answer": raw_answer,
            "citations": citations_data,
            "retrieved_chunks": [r.to_dict() for r in retrieved_results],
            "latency_ms": latency_ms,
            "confidence": confidence,
            "provider_used": self.provider_name,
            "suggested_questions": self._get_global_suggested_questions()
        }

    def _get_global_suggested_questions(self) -> List[str]:
        """Aggregate suggested questions across all ingested documents."""
        all_q = []
        for doc in self.documents.values():
            analysis = doc.get("analysis", {})
            for q in analysis.get("suggested_questions", []):
                if q not in all_q:
                    all_q.append(q)
        return all_q[:4]
