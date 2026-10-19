"""Build a secret-free, provider-free demo from the live UI's shared presentation.

Run after UI/fixture changes; --check verifies the committed export is current.
No .env, live logs, private configuration, backend or model weights are exported.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontend.citation_display import article_text, article_title  # noqa: E402
from frontend.demo_catalog import (  # noqa: E402
    CASE_TOPICS,
    FEATURED_CASES,
    featured_recordings,
    wiki_articles,
)
from frontend.math_display import answer_markdown  # noqa: E402
from frontend.offline_replay import MODES, load_recordings  # noqa: E402

SITE = ROOT / "hosted-demo"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_catalog():
    recordings = []
    for row in featured_recordings(load_recordings()):
        result = row["result"]
        article = result.get("verification_scope") == "article_faithfulness_not_original_source_certification"
        evidence = []
        for source in result.get("evidence", []):
            evidence.append({
                "title": article_title(source["title"]) if article else source["title"],
                "text": article_text(source["text"]) if article else source["text"],
                "section": "" if article else source.get("section", ""),
                "page": source.get("page"), "locator": source.get("locator"),
                "kind": source.get("source_kind", "pdf"),
            })
        provenance = result.get("replay_provenance", {})
        # Strict field allowlist. No original request bodies, environment, keys,
        # local filesystem paths or provider wire data can enter the export.
        recordings.append({
            "id": row["case_id"] + "-" + row["knowledge_mode"],
            "title": FEATURED_CASES[row["case_id"]][0],
            "category": FEATURED_CASES[row["case_id"]][1],
            "wiki_ids": CASE_TOPICS[row["case_id"]],
            "question": row["question"], "mode": row["knowledge_mode"],
            "answer": answer_markdown(result.get("answer") or "未输出答案。"),
            "status": result["status"], "evidence": evidence,
            "timeline": result.get("timeline", []),
            "navigation": [{k: step.get(k, []) for k in ("wiki_pages", "seeds", "graph_neighbors")}
                           for step in result.get("knowledge_navigation", [])],
            "historical_calls": provenance.get("historical_calls_used"),
            "source_sha256": provenance.get("source_sha256"),
            "scope": "完整词条" if article else "原文资料",
        })
    return {"schema": 1, "modes": MODES, "recordings": recordings, "articles": wiki_articles()}


def source_hashes():
    paths = [ROOT / "frontend/fixtures/chat_replays.json", ROOT / "frontend/fixtures/chat_replays.sha256",
             ROOT / "frontend/math_display.py", ROOT / "frontend/citation_display.py",
             ROOT / "frontend/demo_catalog.py", ROOT / "frontend/offline_knowledge.py",
             ROOT / "knowledge/llm_wiki/manifest.json",
             ROOT / "frontend/chat.py", ROOT / "frontend/simulation.py", ROOT / "frontend/styles.css",
             ROOT / "frontend/ui.py", Path(__file__), SITE / "package-lock.json"]
    paths += sorted((SITE / "assets").glob("*"))
    paths += sorted((ROOT / "knowledge/llm_wiki").glob("*.md"))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def build(check=False):
    output = SITE / "dist"
    data = (json.dumps(make_catalog(), ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    hashes = source_hashes()
    if check:
        old = json.loads((output / "build-info.json").read_text(encoding="utf-8"))
        if old["source_hashes"] != hashes or (output / "catalog.json").read_bytes() != data:
            raise SystemExit("Static demo is stale. Run scripts/build_static_demo.py, then republish.")
        for name in ("index.html", "app.js", "style.css"):
            if (output / name).read_bytes() != (SITE / "assets" / name).read_bytes():
                raise SystemExit("Static UI output is stale: " + name)
        print("Static demo matches shared UI presentation, 10 featured questions and 35 articles.")
        return
    output.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "app.js", "style.css"):
        shutil.copyfile(SITE / "assets" / name, output / name)
    vendor = output / "vendor"
    vendor.mkdir(exist_ok=True)
    packages = SITE / "node_modules"
    for source, name in [("marked/lib/marked.umd.js", "marked.js"),
                         ("dompurify/dist/purify.min.js", "purify.js"),
                         ("katex/dist/katex.min.js", "katex.js"),
                         ("katex/dist/contrib/auto-render.min.js", "auto-render.js"),
                         ("katex/dist/katex.min.css", "katex.css")]:
        shutil.copyfile(packages / source, vendor / name)
    shutil.copytree(packages / "katex/dist/fonts", vendor / "fonts", dirs_exist_ok=True)
    for package in ("marked", "katex", "dompurify"):
        license_file = next((p for p in (packages / package).iterdir() if p.name.lower().startswith("license")), None)
        if license_file:
            shutil.copyfile(license_file, vendor / (package + "-LICENSE.txt"))
    (output / "catalog.json").write_bytes(data)
    (output / ".nojekyll").write_text("", encoding="utf-8")
    info = {"source_hashes": hashes, "catalog_sha256": hashlib.sha256(data).hexdigest(),
            "recordings": 16, "questions": 10, "articles": 35, "provider_calls": 0}
    (output / "build-info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    # Netlify/Cloudflare Pages honor this; other static hosts may ignore it.
    (output / "_headers").write_text(
        "/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: no-referrer\n"
        "  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "font-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'\n",
        encoding="utf-8")
    print("Built static demo: 10 questions / 16 recordings / 35 articles. Provider calls: 0.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    build(parser.parse_args().check)
