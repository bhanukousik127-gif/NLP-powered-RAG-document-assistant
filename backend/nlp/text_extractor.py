"""
NLP Document Assistant - Text Extractor
Supports PDF, DOCX, TXT, MD, CSV, JSON with metadata extraction.
"""

import os
import json
import csv
import io
import re
from typing import Dict, Any, List

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None


class DocumentExtractor:
    """Extracts raw text and metadata from diverse file formats."""

    @classmethod
    def extract_from_file(cls, file_path: str, filename: str = None) -> Dict[str, Any]:
        """Extract text and metadata from a file path."""
        if not filename:
            filename = os.path.basename(file_path)
            
        ext = os.path.splitext(filename)[1].lower()
        
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            
        return cls.extract_from_bytes(file_bytes, filename, ext)

    @classmethod
    def extract_from_bytes(cls, file_bytes: bytes, filename: str, ext: str = None) -> Dict[str, Any]:
        """Extract text and structural metadata from file bytes."""
        if not ext:
            ext = os.path.splitext(filename)[1].lower()
            
        metadata = {
            "filename": filename,
            "extension": ext,
            "byte_size": len(file_bytes),
            "pages": 1,
            "sections": []
        }
        
        if ext == ".pdf":
            text, pages = cls._extract_pdf(file_bytes)
            metadata["pages"] = pages
        elif ext in [".docx", ".doc"]:
            text = cls._extract_docx(file_bytes)
        elif ext == ".json":
            text = cls._extract_json(file_bytes)
        elif ext == ".csv":
            text = cls._extract_csv(file_bytes)
        elif ext in [".md", ".markdown"]:
            text = cls._extract_markdown(file_bytes)
        else:
            # Fallback plain text / utf-8
            text = cls._extract_text(file_bytes)
            
        # Clean up whitespace and non-printable artifacts
        text = cls._clean_text(text)
        
        metadata["char_count"] = len(text)
        metadata["word_count"] = len(text.split()) if text else 0
        
        return {
            "text": text,
            "metadata": metadata
        }

    @classmethod
    def _extract_pdf(cls, file_bytes: bytes) -> (str, int):
        """Extract text from PDF using pypdf or regex fallback."""
        text_parts = []
        page_count = 1
        
        if pypdf:
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                page_count = len(reader.pages)
                for idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(f"[Page {idx + 1}]\n{page_text}")
                if text_parts:
                    return "\n\n".join(text_parts), page_count
            except Exception as e:
                pass
                
        # Fallback raw string extraction
        try:
            content = file_bytes.decode("latin1", errors="ignore")
            # Extract text streams in parentheses / BT...ET blocks
            matches = re.findall(r"\((.*?)\)\s*T[jJ]", content)
            if matches:
                return "\n".join(matches), 1
        except Exception:
            pass
            
        return file_bytes.decode("utf-8", errors="ignore"), 1

    @classmethod
    def _extract_docx(cls, file_bytes: bytes) -> str:
        """Extract text from DOCX file bytes."""
        if docx:
            try:
                doc = docx.Document(io.BytesIO(file_bytes))
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                # Also include table text
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                        if row_text:
                            paragraphs.append(row_text)
                return "\n\n".join(paragraphs)
            except Exception:
                pass
                
        # Fallback: unzip word/document.xml
        try:
            import zipfile
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                xml_content = z.read("word/document.xml").decode("utf-8", errors="ignore")
                # Strip XML tags
                text = re.sub(r"<[^>]+>", " ", xml_content)
                text = re.sub(r"\s+", " ", text)
                return text
        except Exception:
            return file_bytes.decode("utf-8", errors="ignore")

    @classmethod
    def _extract_json(cls, file_bytes: bytes) -> str:
        """Format JSON documents into coherent readable text blocks for RAG."""
        try:
            data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
            lines = []
            
            def flatten_json(obj, prefix=""):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        new_prefix = f"{prefix}.{k}" if prefix else str(k)
                        flatten_json(v, new_prefix)
                elif isinstance(obj, list):
                    for idx, item in enumerate(obj):
                        new_prefix = f"{prefix}[{idx}]"
                        flatten_json(item, new_prefix)
                else:
                    lines.append(f"{prefix}: {obj}")

            flatten_json(data)
            return "\n".join(lines)
        except Exception:
            return file_bytes.decode("utf-8", errors="ignore")

    @classmethod
    def _extract_csv(cls, file_bytes: bytes) -> str:
        """Format tabular CSV data into descriptive contextual sentences."""
        try:
            content = file_bytes.decode("utf-8", errors="ignore")
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)
            if not rows:
                return ""
            headers = [h.strip() for h in rows[0]]
            text_rows = []
            for row_idx, row in enumerate(rows[1:], start=1):
                item_desc = []
                for h_idx, val in enumerate(row):
                    h_name = headers[h_idx] if h_idx < len(headers) else f"Column {h_idx+1}"
                    val_str = val.strip()
                    if val_str:
                        item_desc.append(f"{h_name}: {val_str}")
                if item_desc:
                    text_rows.append(f"Record {row_idx}: " + ", ".join(item_desc))
            return "\n".join(text_rows)
        except Exception:
            return file_bytes.decode("utf-8", errors="ignore")

    @classmethod
    def _extract_markdown(cls, file_bytes: bytes) -> str:
        """Extract and normalize markdown content."""
        return file_bytes.decode("utf-8", errors="ignore")

    @classmethod
    def _extract_text(cls, file_bytes: bytes) -> str:
        """Decode plain text using common encodings."""
        for enc in ["utf-8", "utf-16", "latin1", "cp1252"]:
            try:
                return file_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="ignore")

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Normalize line endings, remove excessive blank lines and control characters."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove zero-width spaces and control chars
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        # Collapse > 2 consecutive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
