"""Stage ``dump``: download the dated pages-articles dump, resumably, and verify it."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from typing import Optional

from .config import DUMPS_BASE, ImagesConfig, StageError
from .http import HttpError, make_session, request_with_retry
from .storage import Store

logger = logging.getLogger("wikivoyage.images.dump")

CHUNK = 1024 * 1024


def _expected_checksum(session, cfg: ImagesConfig) -> tuple[str, str]:
    """Return (algorithm, hex digest) for the articles dump, from dumpstatus.json or *sums.txt."""
    date = cfg.require_dump_date()
    fname = cfg.dump_filename()
    base = f"{DUMPS_BASE}/{cfg.wiki}/{date}"
    try:
        resp = request_with_retry(session, "GET", f"{base}/dumpstatus.json", max_attempts=3)
        status = resp.json()
        job = status.get("jobs", {}).get("articlesdump", {})
        if job.get("status") and job.get("status") != "done":
            raise StageError(f"dump {date} is not finished yet (articlesdump status={job.get('status')!r})", 1)
        info = job.get("files", {}).get(fname, {})
        if info.get("sha1"):
            return "sha1", info["sha1"].lower()
        if info.get("md5"):
            return "md5", info["md5"].lower()
    except HttpError as e:
        if e.status == 404:
            raise StageError(f"dump directory not found: {base}/ (date {date})", 1) from e
        logger.warning("dumpstatus.json unavailable (%s); trying *sums.txt", e)
    except ValueError as e:
        logger.warning("dumpstatus.json unparsable (%s); trying *sums.txt", e)

    for algo in ("sha1", "md5"):
        url = f"{base}/{cfg.wiki}-{date}-{algo}sums.txt"
        try:
            text = request_with_retry(session, "GET", url, max_attempts=3).text
        except HttpError:
            continue
        for line in text.splitlines():
            m = re.match(r"^([0-9a-fA-F]+)\s+\*?(\S+)$", line.strip())
            if m and m.group(2) == fname:
                return algo, m.group(1).lower()
    raise StageError(f"no checksum published for {fname}", 1)


def _file_digest(path: str, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_resumable(session, url: str, part_path: str) -> None:
    have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        resp = request_with_retry(session, "GET", url, stream=True, headers=headers, timeout=(10, 120))
    except HttpError as e:
        if e.status == 416 and have:
            logger.info("server reports %s already complete (%d bytes)", part_path, have)
            return
        if e.status == 404:
            raise StageError(f"dump not found: {url}", 1) from e
        raise StageError(f"dump download failed: {e}", 1) from e
    with resp:
        mode = "ab"
        if have and resp.status_code != 206:
            logger.warning("server ignored Range; restarting download from byte 0")
            mode = "wb"
            have = 0
        elif have:
            logger.info("resuming download at byte %d", have)
        total = have
        with open(part_path, mode) as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
        logger.info("downloaded %s (%d bytes)", os.path.basename(part_path), total)


def run(cfg: ImagesConfig, store: Store, *, force: bool = False) -> dict:
    rel = cfg.dump_rel()
    marker = rel + ".verified"
    url = cfg.dump_url()
    if not force and store.exists(rel) and store.exists(marker):
        logger.info("dump already downloaded and verified: %s", store.uri(rel))
        return {"status": "cached", "path": store.uri(rel)}

    session = make_session(cfg.user_agent)
    algo, expected = _expected_checksum(session, cfg)

    # Download into a local file: the store path itself when local, a temp dir for S3.
    tmpdir: Optional[str] = None
    if store.is_local:
        local = store.uri(rel)
        os.makedirs(os.path.dirname(local), exist_ok=True)
    else:
        tmpdir = tempfile.mkdtemp(prefix="wv-dump-")
        local = os.path.join(tmpdir, cfg.dump_filename())
    part = local + ".part"
    if force and os.path.exists(part):
        os.unlink(part)

    logger.info("downloading %s", url)
    _download_resumable(session, url, part)

    actual = _file_digest(part, algo)
    if actual != expected:
        os.unlink(part)
        raise StageError(
            f"{algo} mismatch for {cfg.dump_filename()} (expected {expected}, got {actual}); "
            "the file was deleted, please re-run",
            2,
        )
    logger.info("%s verified (%s=%s)", cfg.dump_filename(), algo, actual)
    size = os.path.getsize(part)
    os.replace(part, local)
    if not store.is_local:
        store.put_file(rel, local)
        os.rmdir(tmpdir)
    store.write_bytes(marker, f"{algo} {actual}\n".encode())
    return {"status": "downloaded", "path": store.uri(rel), "bytes": size, algo: actual}
