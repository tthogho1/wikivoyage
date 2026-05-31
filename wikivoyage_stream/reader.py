"""Stream a bz2-compressed MediaWiki XML dump from a URL or local file."""
from __future__ import annotations

import bz2
import codecs
import logging
from typing import Iterable, Iterator, Optional

from .parser import NamespaceAwareXmlParser
from .models import Page

logger = logging.getLogger("wikivoyage.stream")

DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MiB


class WikivoyageDumpStreamReader:
    """Stream a bz2-compressed MediaWiki XML dump from a URL or local file."""

    def __init__(
        self,
        source: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        timeout: int = 60,
    ) -> None:
        self.source = source
        self.chunk_size = chunk_size
        self.timeout = timeout
        self._compressed_bytes = 0
        self._decompressed_bytes = 0
        self._page_count = 0

    @property
    def compressed_bytes(self) -> int:
        return self._compressed_bytes

    @property
    def decompressed_bytes(self) -> int:
        return self._decompressed_bytes

    @property
    def page_count(self) -> int:
        return self._page_count

    def _iter_compressed_chunks(self) -> Iterable[bytes]:
        if self.source.startswith(("http://", "https://")):
            # Import requests lazily so importing this module doesn't require it.
            import requests

            with requests.get(self.source, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        yield chunk
        elif self.source.startswith("s3://"):
            # Stream directly from S3 without downloading whole object.
            try:
                import boto3
            except Exception as e:  # pragma: no cover - dependency/runtime
                raise RuntimeError("boto3 is required to stream from s3:// URIs") from e

            bucket_and_key = self.source[5:]
            parts = bucket_and_key.split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            if not bucket or not key:
                raise ValueError(f"Invalid s3 URI: {self.source}")

            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            body = obj["Body"]
            # StreamingBody.read() will block until data available; iterate until EOF
            while True:
                chunk = body.read(self.chunk_size)
                if not chunk:
                    break
                yield chunk
        else:
            with open(self.source, "rb") as fh:
                while True:
                    chunk = fh.read(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk

    def iter_pages(self) -> Iterator[Page]:
        dec = bz2.BZ2Decompressor()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        xml_parser = NamespaceAwareXmlParser()

        chunk_index = -1
        try:
            for chunk_index, chunk in enumerate(self._iter_compressed_chunks()):
                self._compressed_bytes += len(chunk)
                try:
                    data = dec.decompress(chunk)
                except (OSError, EOFError) as e:
                    logger.error(
                        "Decompression error at chunk %d (last title=%r): %s",
                        chunk_index,
                        xml_parser.last_title,
                        e,
                    )
                    continue

                if not data:
                    continue
                self._decompressed_bytes += len(data)

                text = decoder.decode(data)
                if not text:
                    continue

                try:
                    xml_parser.feed(text)
                except Exception as e:
                    logger.error(
                        "XML parse error at chunk %d (last title=%r): %s",
                        chunk_index,
                        xml_parser.last_title,
                        e,
                    )
                    continue

                for page in xml_parser.iter_pages():
                    self._page_count += 1
                    yield page

        except Exception as e:
            logger.error(
                "Network/error at chunk %d after %d pages: %s",
                chunk_index,
                self._page_count,
                e,
            )
            raise
        finally:
            logger.info(
                "Done. compressed=%d B, decompressed=%d B, pages=%d, last_title=%r",
                self._compressed_bytes,
                self._decompressed_bytes,
                self._page_count,
                xml_parser.last_title,
            )
