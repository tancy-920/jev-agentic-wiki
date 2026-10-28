"""Static/local presentation parity and public export boundary. No network."""

import json
import re

from frontend.demo_catalog import featured_recordings
from frontend.math_display import answer_markdown
from frontend.offline_replay import load_recordings
from scripts.build_static_demo import SITE, build, make_catalog


def test_every_record_uses_same_clean_prose_as_local_ui():
    raw = featured_recordings(load_recordings())
    catalog = make_catalog()
    assert len(catalog["recordings"]) == 16
    assert len({r["question"] for r in catalog["recordings"]}) == 10
    assert len(catalog["articles"]) == 36
    for original, exported in zip(raw, catalog["recordings"], strict=True):
        assert exported["answer"] == answer_markdown(original["result"]["answer"])
        assert exported["evidence"]
        assert "captured public-source excerpt" not in exported["answer"]
        assert "snapshot L" not in exported["answer"]
        assert "PDF file p." not in exported["answer"]
        assert "AI整理词条" not in exported["answer"]
        assert not re.search(r"\[[^\]\n]*\.md\]", exported["answer"])
    bert = next(r for r in catalog["recordings"] if r["id"].startswith("H02"))
    assert "[MASK]" in bert["answer"]


def test_catalog_contains_only_allowlisted_presentation_fields():
    data = json.dumps(make_catalog(), ensure_ascii=False)
    for forbidden in ('"token"', '"api_key"', '"Authorization"', '"raw_output"',
                      '"request_body"', '"source_path"', ".env.local", "C:\\\\Users"):
        assert forbidden not in data
    assert "generation_failure" in data


def test_static_export_is_current_and_vendors_are_local():
    build(check=True)
    html = (SITE / "dist/index.html").read_text(encoding="utf-8")
    assert 'src="https://' not in html and 'href="https://' not in html
    assert 'id="query"' not in html
    assert 'id="composer"' not in html
    assert 'id="knowledge-query"' in html
    assert 'id="wiki-graph"' in html
    assert (SITE / "dist/vendor/katex.js").is_file()
    assert (SITE / "dist/vendor/marked.js").is_file()
    assert (SITE / "dist/vendor/purify.js").is_file()
    js = (SITE / "assets/app.js").read_text(encoding="utf-8")
    assert js.count("fetch(") == 1
    assert 'fetch("./catalog.json"' in js
    assert "api.deepseek.com" not in js


def test_source_change_fails_freshness_check(monkeypatch, tmp_path):
    import pytest

    import scripts.build_static_demo as builder

    monkeypatch.setattr(builder, "source_hashes", lambda: {"changed-ui": "new"})
    with pytest.raises(SystemExit, match="stale"):
        build(check=True)


def test_real_manifest_graph_and_reading_entries_are_exported():
    from frontend.demo_catalog import CASE_TOPICS, search_articles, wiki_articles

    catalog = make_catalog()
    articles = wiki_articles()
    by_id = {a["id"]: a for a in articles}
    assert len(by_id) == 36
    assert search_articles(articles, "GQA")[0]["id"] == "gqa"
    assert search_articles(articles, "MOCK-no-such-concept-123") == []
    assert "mqa" in by_id["gqa"]["related"]
    assert all(identifier in by_id for ids in CASE_TOPICS.values() for identifier in ids)
    assert all(r["wiki_ids"] for r in catalog["recordings"])
    assert all(a["aliases"] and isinstance(a["related"], list) for a in catalog["articles"])


def test_wiki_identity_and_dedicated_full_question_collection():
    html = (SITE / "assets/index.html").read_text(encoding="utf-8")
    js = (SITE / "assets/app.js").read_text(encoding="utf-8")
    assert '<b>Wiki</b>' in html and '<b>Jev</b>' not in html
    assert '<select id="question"' not in html
    assert 'id="examples-view"' in html and 'id="browse-examples"' in html
    assert 'id="answer-view"' in html
    assert 'el("p", "example-question", record.question)' in js
    assert 'switchView("examples")' in js
