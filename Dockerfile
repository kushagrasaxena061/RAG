FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y poppler-utils tesseract-ocr libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
