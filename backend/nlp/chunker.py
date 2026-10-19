"""
NLP Document Assistant - Semantic & Recursive Chunker
Implements recursive splitting with boundary preservation and metadata tracking.
"""

import re
from typing import List, Dict, Any


class Chunk:
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        filename: str,
        text: str,
        start_char: int,
        end_char: int,
        page_num: int = 1,
        section_title: str = ""
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.filename = filename
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.page_num = page_num
        self.section_title = section_title
        self.char_count = len(text)
        self.word_count = len(text.split())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "page_num": self.page_num,
            "section_title": self.section_title,
            "char_count": self.char_count,
            "word_count": self.word_count
        }


class RecursiveChunker:
    """Recursively splits text into overlapping chunks honoring semantic boundaries."""

    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))

    def split_document(self, doc_id: str, filename: str, text: str) -> List[Chunk]:
        """Split complete document text into structured chunks with metadata."""
        if not text or not text.strip():
            return []

        # Find section headers to preserve context
        header_patterns = re.compile(r"^(#{1,6}\s+.+|[A-Z0-9\s]{4,}:)", re.MULTILINE)
        headers = []
        for match in header_patterns.finditer(text):
            headers.append((match.start(), match.group(0).strip()))

        raw_chunks = self._recursive_split(text, self.SEPARATORS)
        
        # Merge small fragments and apply overlap
        merged_chunks = self._merge_and_overlap(raw_chunks, text)

        result_chunks = []
        for idx, (chunk_text, start_pos, end_pos) in enumerate(merged_chunks):
            # Detect page number if embedded
            page_match = re.search(r"\[Page\s+(\d+)\]", chunk_text)
            page_num = int(page_match.group(1)) if page_match else 1
            if not page_match:
                # Approximate page based on 2500 chars per page
                page_num = max(1, (start_pos // 2500) + 1)

            # Detect closest section title
            section_title = ""
            for h_pos, h_title in reversed(headers):
                if h_pos <= start_pos:
                    section_title = h_title.lstrip("#").strip()
                    break

            chunk_obj = Chunk(
                chunk_id=f"{doc_id}_c{idx:03d}",
                doc_id=doc_id,
                filename=filename,
                text=chunk_text.strip(),
                start_char=start_pos,
                end_char=end_pos,
                page_num=page_num,
                section_title=section_title
            )
            result_chunks.append(chunk_obj)

        return result_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Split text recursively using descending priority separators."""
        final_pieces = []
        separator = separators[-1]
        new_separators = []

        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if re.search(re.escape(s), text):
                separator = s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        for s in splits:
            if not s:
                continue
            if len(s) <= self.chunk_size:
                final_pieces.append(s)
            elif new_separators:
                final_pieces.extend(self._recursive_split(s, new_separators))
            else:
                # Force chunk by character count
                for c in range(0, len(s), self.chunk_size):
                    final_pieces.append(s[c : c + self.chunk_size])

        return final_pieces

    def _merge_and_overlap(self, pieces: List[str], full_text: str) -> List[tuple]:
        """Combine fragments up to chunk_size and add overlap."""
        chunks = []
        current_chunk = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)
            if current_len + piece_len > self.chunk_size and current_chunk:
                joined = " ".join(current_chunk).strip()
                chunks.append(joined)
                
                # Create overlap buffer
                overlap_accum = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    overlap_accum.insert(0, item)
                    overlap_len += len(item)
                    if overlap_len >= self.chunk_overlap:
                        break
                current_chunk = overlap_accum
                current_len = sum(len(x) for x in current_chunk) + len(current_chunk)

            current_chunk.append(piece)
            current_len += piece_len + 1

        if current_chunk:
            joined = " ".join(current_chunk).strip()
            if joined:
                chunks.append(joined)

        # Map back to exact character positions in full_text
        mapped = []
        search_start = 0
        for ch in chunks:
            # Clean search snippet
            search_key = ch[:min(40, len(ch))]
            pos = full_text.find(search_key, search_start)
            if pos == -1:
                pos = search_start
            end_pos = min(len(full_text), pos + len(ch))
            mapped.append((ch, pos, end_pos))
            search_start = max(0, pos + len(ch) - self.chunk_overlap)

        return mapped
