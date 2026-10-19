"""
RAGNova NLP Document Assistant - One-Click Launcher
Starts the Flask server and opens the application in your default web browser.
"""

import os
import sys
import webbrowser
import threading
import time

def open_browser(url: str):
    time.sleep(1.2)
    try:
        webbrowser.open(url)
    except Exception:
        pass

def main():
    port = int(os.environ.get("PORT", 5000))
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    banner = rf"""
    ========================================================================
     ____       _     ____  _                       
    |  _ \     / \   / ___|| \ | | _____   ____ _  
    | |_) |   / _ \ | |  _ |  \| |/ _ \ \ / / _` | 
    |  _ <   / ___ \| |_| || |\  | (_) \ V / (_| | 
    |_| \_\ /_/   \_\\____||_| \_|\___/ \_/ \__,_| 
                                                    
     [*] NLP-Powered RAG Document Assistant & Intelligence Suite
     [+] Okapi BM25 + Dense Vectors + Reciprocal Rank Fusion + Citations
    ========================================================================
     -> Server running at: {url}
     -> Ready to ingest: PDF, DOCX, TXT, Markdown, CSV, JSON
     -> Press Ctrl+C in this terminal to stop the server.
    ========================================================================
    """
    print(banner)

    # Spawn auto-browser opener
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    from backend.app import app
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    main()
