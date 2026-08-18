# ==============================================================================
# HOW TO RUN THIS REPOSITORY IN ONE COMMAND:
#
# Once cloned, run this single chained command in your terminal:
# docker build -t adaptive-rag . && docker run -p 8000:8000 -p 8501:8501 -p 11434:11434 adaptive-rag
# ==============================================================================

FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies (Poppler for PDFs, Tesseract for OCR, and curl for Ollama)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install the Ollama engine directly into the container
RUN curl -fsSL https://ollama.com/install.sh | sh

# 3. Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the entire repository codebase
COPY . .

# Ensure your startup script has executable permissions
RUN chmod +x start.sh

# 5. Create an all-in-one entrypoint script
# This script starts Ollama, pulls the required model, and then boots your platform.
RUN echo '#!/bin/bash\n\
echo "=========================================="\n\
echo "1. Starting Ollama Server..."\n\
echo "=========================================="\n\
ollama serve &\n\
sleep 5\n\
echo "=========================================="\n\
echo "2. Pulling the LLM (qwen3:latest)..."\n\
echo "=========================================="\n\
ollama pull qwen3:latest\n\
echo "=========================================="\n\
echo "3. Starting Backend & Frontend..."\n\
echo "=========================================="\n\
./start.sh\n\
wait' > /app/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Expose ports: FastAPI (8000), Streamlit (8501), and Ollama (11434)
EXPOSE 8000 8501 11434

# Execute the entrypoint script when the container launches
CMD ["/app/entrypoint.sh"]