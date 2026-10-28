"""Verified, invalidating read cache for provider-free source browsing only."""

import json
from functools import lru_cache
from pathlib import Path

from jev_rag.knowledge import KnowledgeStore, inside, product_store_path


def source_revision(root):
    """Watch catalog plus all raw/normalized inputs; drift always re-verifies."""
    root = Path(root).resolve()
    files = {root / name for name in ("sources.jsonl", "chunks.jsonl", "wiki.json", "index.json")
             if (root / name).exists()}
    for line in (root / "sources.jsonl").read_text("utf8").splitlines():
        row = json.loads(line)
        files.update(inside(root, row[key]) for key in ("snapshot", "normalized"))
    revision = []
    for path in sorted(files):
        stat = path.stat()
        revision.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(revision)


@lru_cache(maxsize=2)
def _verified_store(root, revision):
    return KnowledgeStore(Path(root))


def verified_store(workspace):
    root = product_store_path(Path(workspace)).resolve()
    return _verified_store(str(root), source_revision(root))
