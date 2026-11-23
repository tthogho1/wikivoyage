# article_db.api: FastAPI retrieval service (Qwen3-Embedding-0.6B + SigLIP2-so400m, CPU-only).
# Models are baked into the image at build time so the task starts instantly and needs no
# internet access to Hugging Face at runtime (only Zilliz Cloud + nothing else).
#
# Build from the repo root:  docker build -f article_db/Dockerfile -t wikivoyage-article-db-api .
#
# Image is ~7-8 GB, mostly the two models (~5.8 GB). Two things keep it from doubling:
#  - torch comes from the CPU-only wheel index (Fargate has no GPU; the default Linux wheel
#    pulls ~5 GB of NVIDIA CUDA libraries).
#  - the models are downloaded as appuser, never chown'ed afterwards (chown -R on a later
#    layer would copy all 5.8 GB of model files into a second layer).

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    HF_HOME=/opt/hf \
    PORT=8001

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY article_db/requirements-search.txt article_db/requirements-search.txt
# requirements-search.txt -r's requirements-embed.txt, which -r's requirements.txt: copy the whole chain.
COPY image_embedding/requirements-embed.txt image_embedding/requirements-embed.txt
COPY image_embedding/requirements.txt image_embedding/requirements.txt
# CPU-only torch first; the requirements' `torch>=2.3` is then already satisfied, so pip
# doesn't fetch the CUDA build from PyPI.
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --retries 5 --index-url https://download.pytorch.org/whl/cpu "torch>=2.3" \
    && pip install --no-cache-dir --retries 5 -r article_db/requirements-search.txt \
    && if pip list 2>/dev/null | grep -qi '^nvidia-'; then echo "CUDA packages got installed" >&2; exit 1; fi

RUN useradd --create-home --home-dir /home/appuser appuser \
    && mkdir -p /opt/hf \
    && chown appuser:appuser /app /opt/hf
USER appuser

# Bake the models into the image so no download happens at container start.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')" \
    && python -c "\
from transformers import AutoModel, AutoProcessor; \
AutoModel.from_pretrained('google/siglip2-so400m-patch16-384'); \
AutoProcessor.from_pretrained('google/siglip2-so400m-patch16-384')"

COPY --chown=appuser:appuser article_db/ article_db/
COPY --chown=appuser:appuser image_embedding/ image_embedding/
COPY --chown=appuser:appuser wikivoyage_stream/ wikivoyage_stream/

EXPOSE 8001
ENTRYPOINT ["sh", "-c", "uvicorn article_db.api:app --host 0.0.0.0 --port ${PORT}"]
