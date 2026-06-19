#!/usr/bin/env bash

set -e

ollama serve &

sleep 10

ollama pull qwen2.5:0.5b
ollama pull llama3.2:1b
ollama pull nomic-embed-text

streamlit run app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableXsrfProtection false \
  --server.enableCORS false \
  --server.maxUploadSize 200