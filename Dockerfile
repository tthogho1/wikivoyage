# Minimal Dockerfile for running get_wikivoyage.py on AWS Fargate
# Target: ~400–600 MB

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install dependencies with timeout/retries and no cache
RUN pip install --upgrade pip setuptools wheel \
    && PIP_DEFAULT_TIMEOUT=100 \
       pip install --no-cache-dir \
       --retries 5 \
       --timeout 100 \
       -r /app/requirements.txt

# Copy application source
COPY . /app

# Create a non-root user and use it
RUN useradd --create-home --home-dir /home/appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

WORKDIR /app
ENTRYPOINT ["python", "get_wikivoyage.py"]