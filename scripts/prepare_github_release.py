"""Prepare an allowlisted source release, without history, secrets or API calls."""
import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["build_static_demo.py", "benchmark_live_app.py", "answer_text_coverage_candidate.py",
           "prepare_local_workspace.py", "prepare_github_release.py"]
TESTS = ["conftest.py", "test_static_demo.py", "test_math_display.py", "test_offline_simulation.py",
         "test_session_recovery.py", "test_answer_gaps.py", "test_article_repairs.py", "test_dependency_scope.py", "test_semantic_scope.py", "test_quantity_scope.py", "test_chat_agent.py"]


def publication_files():
    files = {ROOT / name for name in ["README.md", "LICENSE", "PUBLICATION.md", "pyproject.toml",
        ".env.example", ".gitignore", ".gitattributes", ".streamlit/config.toml",
        "configs/base.yaml", "configs/transformer.yaml", "experiments/__init__.py",
        "experiments/ablation.py", ".github/workflows/wiki-graph-experiments.yml",
        ".github/workflows/static-demo.yml", "hosted-demo/package.json",
        "hosted-demo/package-lock.json", "hosted-demo/.gitignore", "hosted-demo/README.md"]}
    files.update(ROOT / "scripts" / name for name in SCRIPTS)
    files.update(ROOT / "tests" / name for name in TESTS)
    for directory, suffixes in [("src", {".py"}), ("frontend", {".py", ".css"}),
                                ("frontend/fixtures", {".json", ".sha256"}),
                                ("knowledge/llm_wiki", {".md", ".json"})]:
        files.update(p for p in (ROOT / directory).rglob("*") if p.is_file() and p.suffix in suffixes)
    files.update(p for p in (ROOT / "hosted-demo/assets").rglob("*") if p.is_file())
    artifact = ROOT / "experiments/wiki_graph_value"
    manifest = json.loads((artifact / "MANIFEST.json").read_text("utf8"))
    files.update(artifact / rel for rel in [*manifest["files"], "MANIFEST.json"])
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release/github-ready")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = publication_files()
    sys.path.insert(0, str(ROOT / "experiments/wiki_graph_value"))
    from reproduce import verify

    verify()
    blocked = re.compile(rb"(?:sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
    hashes = {}
    for path in files:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Missing or unsafe public input: " + str(path))
        if path.suffix.lower() in {".pdf", ".pt", ".safetensors", ".sqlite", ".log"}:
            raise ValueError("Private/heavy artifact blocked")
        if path.name.startswith(".env") and path.name != ".env.example":
            raise ValueError("Credential file blocked")
        data = path.read_bytes()
        if blocked.search(data):
            raise ValueError("Credential-shaped data blocked: " + path.relative_to(ROOT).as_posix())
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
    destination = args.output.resolve()
    if destination == ROOT or not destination.is_relative_to(ROOT / "release"):
        raise ValueError("Publish only into a dedicated release subdirectory")
    if destination.exists():
        unexpected = [p.relative_to(destination).as_posix() for p in destination.rglob("*")
                      if p.is_file() and p.relative_to(destination).as_posix() not in hashes
                      and p.name != "PUBLICATION_MANIFEST.json"]
        if unexpected:
            raise ValueError("Use a fresh release directory; unexpected files: " + str(unexpected[:5]))
    if args.check:
        for rel, expected in hashes.items():
            path = destination / rel
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Stale or missing release file: " + rel)
        print(json.dumps({"checked_files": len(hashes), "provider_calls": 0}))
        return
    for path in files:
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    manifest = dict(schema=1, purpose="Curated GitHub source release", provider_calls=0,
                    algorithm="sha256-exact-bytes", files=hashes,
                    exclusions=["credentials", "private grants", "local logs", "intermediate reports",
                        "unrelated historical runners", "downloaded PDFs", "model weights", "vector indexes",
                        "node_modules", "virtual environments", "Git history"])
    (destination / "PUBLICATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf8")
    archive = ROOT / "release/jev-agentic-wiki-github-ready.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in [*hashes, "PUBLICATION_MANIFEST.json"]:
            z.write(destination / rel, "jev-agentic-wiki/" + rel)
    print(json.dumps({"directory": str(destination), "files": len(hashes) + 1,
                      "archive": str(archive), "bytes": archive.stat().st_size,
                      "provider_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
