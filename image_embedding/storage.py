"""Work-area abstraction over a local directory or an S3 prefix.

All progress is recorded as immutable *part files* that are written in one shot
(temp file + rename locally, a single PUT on S3) and merged last-wins when read,
so nothing ever needs appending and a crash never leaves a half-written part.
Part names start with the run id (UTC timestamp first), so sorting them by name
is chronological.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Iterator, Optional


class Store(ABC):
    @abstractmethod
    def uri(self, rel: str) -> str: ...

    @abstractmethod
    def exists(self, rel: str) -> bool: ...

    @abstractmethod
    def read_bytes(self, rel: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, rel: str, data: bytes) -> None:
        """Write atomically: readers see the old object or the complete new one."""

    @abstractmethod
    def put_file(self, rel: str, local_path: str) -> None:
        """Move/upload a finished local file to ``rel``."""

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """Sorted relative paths of all objects under ``prefix``."""

    @abstractmethod
    def delete(self, rel: str) -> None: ...

    def move(self, src: str, dst: str) -> None:
        self.write_bytes(dst, self.read_bytes(src))
        self.delete(src)

    @property
    def is_local(self) -> bool:
        return False

    # Convenience helpers -------------------------------------------------
    def read_json(self, rel: str) -> Any:
        return json.loads(self.read_bytes(rel).decode("utf-8"))

    def write_json(self, rel: str, obj: Any) -> None:
        self.write_bytes(rel, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))

    def iter_jsonl(self, rel: str) -> Iterator[dict]:
        for line in io.TextIOWrapper(io.BytesIO(self.read_bytes(rel)), encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)

    def iter_jsonl_parts(self, prefix: str, stem: str = "part-") -> Iterator[dict]:
        for rel in self.list(prefix):
            name = rel.rsplit("/", 1)[-1]
            if name.startswith(stem) and name.endswith(".jsonl"):
                yield from self.iter_jsonl(rel)


class LocalStore(Store):
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)

    @property
    def is_local(self) -> bool:
        return True

    def path(self, rel: str) -> str:
        return os.path.join(self.root, *rel.split("/"))

    def uri(self, rel: str) -> str:
        return self.path(rel)

    def exists(self, rel: str) -> bool:
        return os.path.exists(self.path(rel))

    def read_bytes(self, rel: str) -> bytes:
        with open(self.path(rel), "rb") as fh:
            return fh.read()

    def write_bytes(self, rel: str, data: bytes) -> None:
        dest = self.path(rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.chmod(tmp, 0o644)  # mkstemp creates 0600
            os.replace(tmp, dest)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def put_file(self, rel: str, local_path: str) -> None:
        dest = self.path(rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.abspath(local_path) != dest:
            shutil.move(local_path, dest)

    def list(self, prefix: str) -> list[str]:
        base = self.path(prefix)
        out: list[str] = []
        if not os.path.isdir(base):
            return out
        for dirpath, _dirs, files in os.walk(base):
            for f in files:
                if f.startswith(".tmp-"):
                    continue
                full = os.path.join(dirpath, f)
                out.append(os.path.relpath(full, self.root).replace(os.sep, "/"))
        return sorted(out)

    def delete(self, rel: str) -> None:
        p = self.path(rel)
        if os.path.exists(p):
            os.unlink(p)

    def move(self, src: str, dst: str) -> None:
        dest = self.path(dst)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(self.path(src), dest)


class S3Store(Store):
    def __init__(self, bucket: str, prefix: str = "", client: Any = None) -> None:
        if client is None:
            import boto3

            client = boto3.client("s3")
        self.s3 = client
        self.bucket = bucket
        self.prefix = (prefix.strip("/") + "/") if prefix.strip("/") else ""

    def key(self, rel: str) -> str:
        return self.prefix + rel

    def uri(self, rel: str) -> str:
        return f"s3://{self.bucket}/{self.key(rel)}"

    def exists(self, rel: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=self.key(rel))
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def read_bytes(self, rel: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=self.key(rel))["Body"].read()

    def write_bytes(self, rel: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self.key(rel), Body=data)

    def put_file(self, rel: str, local_path: str) -> None:
        self.s3.upload_file(local_path, self.bucket, self.key(rel))
        os.unlink(local_path)

    def list(self, prefix: str) -> list[str]:
        out: list[str] = []
        paginator = self.s3.get_paginator("list_objects_v2")
        full = self.key(prefix.rstrip("/") + "/") if prefix else self.prefix
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []) or []:
                out.append(obj["Key"][len(self.prefix):])
        return sorted(out)

    def delete(self, rel: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=self.key(rel))

    def move(self, src: str, dst: str) -> None:
        self.s3.copy_object(
            Bucket=self.bucket, Key=self.key(dst),
            CopySource={"Bucket": self.bucket, "Key": self.key(src)},
        )
        self.delete(src)


def open_store(workdir: str) -> Store:
    """``s3://bucket/prefix`` → S3Store, anything else → LocalStore."""
    if workdir.startswith("s3://"):
        rest = workdir[5:]
        bucket, _, prefix = rest.partition("/")
        if not bucket:
            raise ValueError(f"invalid S3 work dir: {workdir!r}")
        return S3Store(bucket, prefix)
    return LocalStore(workdir)


def last_wins(rows: Iterable[dict], key: Callable[[dict], Any]) -> dict:
    """Merge rows so the last row for each key wins."""
    merged: dict = {}
    for r in rows:
        merged[key(r)] = r
    return merged


class PartWriter:
    """Thread-safe buffered writer that emits ``<prefix>/<stem><run_id>-<seq>.jsonl``."""

    def __init__(
        self,
        store: Store,
        prefix: str,
        run_id: str,
        flush_every: int = 1000,
        stem: str = "part-",
        on_flush: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        self.store = store
        self.prefix = prefix.rstrip("/")
        self.run_id = run_id
        self.flush_every = flush_every
        self.stem = stem
        self.on_flush = on_flush
        self._rows: list[dict] = []
        self._seq = 0
        self._lock = threading.Lock()
        self.parts_written: list[str] = []
        self.total = 0

    def add(self, row: dict) -> None:
        with self._lock:
            self._rows.append(row)
            self.total += 1
            if len(self._rows) >= self.flush_every:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._rows:
            return
        rel = f"{self.prefix}/{self.stem}{self.run_id}-{self._seq:05d}.jsonl"
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self._rows)
        self.store.write_bytes(rel, data.encode("utf-8"))
        self.parts_written.append(rel)
        if self.on_flush:
            self.on_flush(rel, len(self._rows))
        self._seq += 1
        self._rows = []

    close = flush
