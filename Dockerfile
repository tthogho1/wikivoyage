# Minimal Dockerfile for running get_wikivoyage.py on AWS Fargate
# - Uses slim Python image
# - Installs common build deps for Python packages that may need compilation
# - Installs Python dependencies from requirements.txt
# - Copies the app and runs as an unprivileged user

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       libxml2-dev \
       libxslt1-dev \
       libbz2-dev \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY . /app

# Create a non-root user and use it
RUN useradd --create-home --home-dir /home/appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

# Default working directory and command
WORKDIR /app
ENTRYPOINT ["python", "get_wikivoyage.py"]
