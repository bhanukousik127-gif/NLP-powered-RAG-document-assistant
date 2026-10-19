"""
NLP Document Assistant - Web Server & REST API
Exposes document ingestion, NLP intelligence, and RAG conversational endpoints.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so backend imports work cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from backend.rag.engine import RAGEngine

load_dotenv()

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Global singleton RAG Engine
rag_engine = RAGEngine()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/health", methods=["GET"])
def health():
    settings = rag_engine.get_settings()
    return jsonify({
        "status": "healthy",
        "app_name": "RAGNova NLP Document Assistant",
        "version": "1.0.0",
        "settings": settings
    })


@app.route("/api/documents", methods=["GET"])
def get_documents():
    docs_list = []
    for doc_id, doc in rag_engine.documents.items():
        docs_list.append({
            "doc_id": doc_id,
            "filename": doc["filename"],
            "metadata": doc["metadata"],
            "analysis": doc["analysis"]
        })
    return jsonify({"documents": docs_list, "total": len(docs_list)})


@app.route("/api/documents/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = file.filename
    file_bytes = file.read()

    try:
        result = rag_engine.ingest_file(file_bytes, filename)
        return jsonify({
            "message": f"Successfully ingested {filename}",
            "document": result
        }), 201
    except Exception as e:
        return jsonify({"error": f"Failed to process document: {str(e)}"}), 500


@app.route("/api/documents/sample", methods=["POST"])
def load_sample_documents():
    """Ingests pre-packaged sample documents for instant testing."""
    sample_dir = ROOT_DIR / "sample_docs"
    loaded = []
    if sample_dir.exists():
        for file_path in sample_dir.glob("*.*"):
            if file_path.is_file():
                filename = file_path.name
                # Avoid re-ingesting duplicate filenames
                if any(doc["filename"] == filename for doc in rag_engine.documents.values()):
                    continue
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                res = rag_engine.ingest_file(file_bytes, filename)
                loaded.append(res)

    return jsonify({
        "message": f"Loaded {len(loaded)} sample document(s)",
        "documents": loaded
    })


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    success = rag_engine.delete_document(doc_id)
    if success:
        return jsonify({"message": f"Document {doc_id} deleted successfully"})
    return jsonify({"error": "Document not found"}), 404


@app.route("/api/documents/<doc_id>/chunks", methods=["GET"])
def get_document_chunks(doc_id):
    chunks = [c.to_dict() for c in rag_engine.chunks if c.doc_id == doc_id]
    return jsonify({"doc_id": doc_id, "chunks": chunks, "total": len(chunks)})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    history = data.get("history", [])

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        response = rag_engine.answer_query(query=query, conversation_history=history)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Error generating answer: {str(e)}"}), 500


@app.route("/api/settings", methods=["GET", "POST"])
def handle_settings():
    if request.method == "POST":
        data = request.get_json() or {}
        updated = rag_engine.update_settings(data)
        return jsonify({"message": "Settings updated", "settings": updated})
    return jsonify(rag_engine.get_settings())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"[*] RAGNova NLP Document Assistant Running on http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
