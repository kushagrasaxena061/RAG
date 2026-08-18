# 🤠 Adaptive Token-Efficient AI Platform

A production-grade, multimodal Retrieval-Augmented Generation (RAG) engine designed to maximize answer quality while minimizing KLM context token usage.

	## ⌨ Key Features
- **Hybrid RRF Search:** Combines BM25 lexical search with dense vector embeddings via Reciprocal Rank Fusion.
- **Token Budget Manager:** Strictly enforces context limits to prevent token overflow and reduce inference costs.
- **Multimodal CLIP Retrieval:** Natively ingests, embeds, and retrieves images, charts, and figures.
- **Table Preservation:** Extracts and preserves tabular data structures in Markdown format.
- **Contradiction & Temporal Reasoning:** Detects cross-document discrepancies and sorts evidence chronologically.
- **Two-Tier Memory:** Manages both short-term sliding windows and long-term semantic conversation history.
- **Real-Time Streaming:** Sub-second Time-To-First-Token (TTFT) with live UI rendering.
- **Dynamic Model Routing:** Hot-swap local LLMs directly from the UI.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- Local LLM provider running (e.g., Ollama).
- System dependencies: `poppler-utils` and `tesseract-ocr` (for PDF parsing and OCR fallback).

### 2. Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Running the Application
Use the provided startup script to launch both the FastAPI backend and Streamlit UI simultaneously:
```bash
./start.sh
```
Access the web interface at: **http://localhost:8501**
