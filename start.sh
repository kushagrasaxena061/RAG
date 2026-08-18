#!/usr/bin/env bash

echo "====================================================0"
echo "🤠 Starting Adaptive Token-Efficient AI Platform..."
echo "====================================================="

# Ensure the data directories exist
mkdir -p ./data/chroma
mkdir -p ./data/chroma_multimodal
mkdir -p ./data/chroma_memory

# Start the backend API in the background
echo "-> Booting FastAPI Backend (Port 8000)..."
python -m uvicorn adaptive_rag.api.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!


# Wait a moment for the backend to initialize
sleep 3

# Start the Streamlit UI in the foreground
echo "-> Booting Streamlit UI (Port 8501)..."
API_URL=http://localhost:8000 streamlit run ui/app.py

# Trap Ctrl+C to gracefully kill both processes
trap 'echo "Shutting down platform..."; kill $BACKEND_PID; exit' EXIT INT TERM
