# Dockerfile for the Wikivoyage RAG Chatbot API (chatbot/main.py)
#
# Build (run from the project root – the build context must be the repo root
# so that the `chatbot/` package is available):
#
#   docker build -f chatbot/Dockerfile -t wikivoyage-chatbot .
#
# Run:
#
#   docker run --rm -p 8000:8000 --env-file .env wikivoyage-chatbot
#
# The container serves the FastAPI app + static UI on port 8000.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    # Cache the BGE-M3 model inside the image-friendly location
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

# Minimal build/runtime deps (curl is used by the HEALTHCHECK)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --retries 5 --timeout 100 \
       -r /app/requirements.txt

# Copy only what the chatbot service needs
COPY . /app/chatbot/

# Create a non-root user and hand over ownership
RUN useradd --create-home --home-dir /home/appuser appuser \
    && mkdir -p "$HF_HOME" \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

EXPOSE 8000

# Container-level health probe hitting the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Start the ASGI server. Use the module path so the `chatbot` package resolves.
CMD ["uvicorn", "chatbot.main:app", "--host", "0.0.0.0", "--port", "8000"]
