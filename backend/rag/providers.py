"""
NLP Document Assistant - Generation Providers
Supports Google Gemini LLM API (gemini-2.0-flash / gemini-1.5-flash) and
a Built-in Local NLP Synthesizer that works 100% offline with zero API keys.
"""

import os
import re
from typing import List, Dict, Any, Optional

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


class BaseProvider:
    def generate(self, prompt: str, system_instruction: str = "") -> str:
        raise NotImplementedError


class GeminiProvider(BaseProvider):
    """Google Gemini LLM provider using modern google-genai SDK."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        try:
            config = types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.9,
                system_instruction=system_instruction if system_instruction else None
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            return response.text or "I could not generate an answer from the provided context."
        except Exception as e:
            return f"Gemini API Error: {str(e)}. Please check your API key or network connection."


class LocalNLPSynthesizer(BaseProvider):
    """
    Built-in Local Extractive & Generative NLP Synthesizer.
    Generates grounded answers with exact source citations completely offline.
    """

    def __init__(self):
        self.name = "Local NLP Synthesizer (Offline Engine)"

    def generate(self, prompt: str, system_instruction: str = "") -> str:
        """
        Parses the RAG prompt containing query and context chunks,
        selects the most salient sentences, synthesizes them into structured answers,
        and attributes citations.
        """
        # Extract query and context sections from prompt
        query_match = re.search(r"USER QUERY:\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
        query = query_match.group(1).strip() if query_match else ""

        # Extract context chunks
        chunks = re.findall(r"\[Chunk ID:\s*(.+?)\]\s*\[Source:\s*(.+?)\]\s*\[Page:\s*(\d+)\]\s*\n(.*?)(?=\n\[Chunk ID:|\n={3,}|\Z)", prompt, re.DOTALL)
        
        if not chunks:
            return "No matching document excerpts were found to answer your inquiry."

        query_tokens = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", query.lower()))
        
        synthesized_points = []
        citations_used = set()

        for chunk_id, filename, page_str, content in chunks[:5]:
            clean_content = content.strip()
            # Split into individual sentences
            sentences = re.split(r"(?<=[.!?])\s+", clean_content)
            
            # Score sentences against query tokens
            scored_sentences = []
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 15:
                    continue
                s_tokens = set(re.findall(r"\b[A-Za-z0-9_-]{3,}\b", s_clean.lower()))
                overlap = len(query_tokens.intersection(s_tokens))
                scored_sentences.append((overlap, s_clean))

            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            
            if scored_sentences and scored_sentences[0][0] > 0:
                best_sentence = scored_sentences[0][1]
                citation = f"**[{filename}, Page {page_str}, {chunk_id}]**"
                citations_used.add(citation)
                synthesized_points.append(f"- {best_sentence} {citation}")
            elif len(clean_content) > 30 and len(synthesized_points) < 3:
                snippet = clean_content[:200].strip() + "..."
                citation = f"**[{filename}, Page {page_str}, {chunk_id}]**"
                citations_used.add(citation)
                synthesized_points.append(f"- Relevant excerpt: \"{snippet}\" {citation}")

        if not synthesized_points:
            # Fallback to top chunk snippet
            top_cid, top_fn, top_pg, top_txt = chunks[0]
            snippet = top_txt[:300].strip()
            return f"Based on **{top_fn}** (Page {top_pg}):\n\n> {snippet}...\n\n*(Note: For advanced conversational generation, connect a Gemini API Key in Settings).* "

        response_body = "\n\n".join(synthesized_points)
        citation_summary = ", ".join(sorted(list(citations_used)))

        return (
            f"### Grounded Response\n\n"
            f"{response_body}\n\n"
            f"---\n"
            f"**Sources Referenced:** {citation_summary}\n\n"
            f"*Generated locally via built-in NLP engine (no API key required).*"
        )
