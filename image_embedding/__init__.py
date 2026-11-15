"""Wikivoyage image pipeline: dump → image refs → Commons metadata → thumbnails →
SigLIP2 embeddings → Zilliz collection ``wikivoyage_images``.

Run as ``python -m image_embedding <subcommand>`` from the repo root.
Design: ``design/image-embedding-design.md``.
"""
