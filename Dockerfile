FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    zstd \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fix Windows line endings in start.sh
RUN sed -i 's/\r$//' start.sh

ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_BASE_URL=http://localhost:11434

EXPOSE 7860

CMD ["bash", "start.sh"]