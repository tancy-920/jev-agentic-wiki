"""Restore verified original snapshots locally; no downloads or provider calls."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "experiments/wiki_graph_value"
sys.path.insert(0, str(ARTIFACT))
from benchmark_retrieval import validate_corpus  # noqa: E402
from reproduce import verify  # noqa: E402


def sources():
    return [json.loads(s) for s in (ARTIFACT / "data/corpus/sources.jsonl").read_text("utf8").splitlines()]


def snapshot_files(directory):
    directory = directory.resolve()
    files = []
    for source in sources():
        relative = Path(source["snapshot"].replace("\\", "/"))
        if relative.parts[0] != "snapshots" or len(relative.parts) != 2:
            raise ValueError("Unexpected snapshot path")
        path = (directory / relative.name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Missing snapshot: " + relative.name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["snapshot_sha256"]:
            raise ValueError("Snapshot hash differs: " + relative.name)
        files.append((path, relative))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-snapshots", action="store_true")
    parser.add_argument("--snapshots", type=Path, help="directory with original snapshot filenames")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--build-source-index", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    verify()
    validate_corpus()
    if args.list_snapshots:
        print(json.dumps([{k: s[k] for k in ("source_id", "origin", "snapshot", "snapshot_sha256")}
                          for s in sources()], ensure_ascii=False, indent=2))
        return
    target = ROOT / "data/processed/transformer_wiki_v13"
    if args.snapshots:
        files = snapshot_files(args.snapshots)
        if args.check_only:
            print(json.dumps({"verified_snapshots": len(files), "provider_calls": 0}))
            return
        if target.exists():
            raise ValueError("Local corpus already exists; refusing to overwrite it")
        shutil.copytree(ARTIFACT / "data/corpus", target)
        shutil.copyfile(ARTIFACT / "data/source_index_metadata.json", target / "index.json")
        for source, relative in files:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    elif not args.build_source_index:
        parser.error("Use --list-snapshots, --snapshots DIR or --build-source-index")
    sys.path.insert(0, str(ROOT / "src"))
    from jev_rag.knowledge import KnowledgeStore
    from jev_rag.llm_wiki import ArticleStore

    store = KnowledgeStore(target)  # original bytes and normalized spans checked
    if args.build_source_index:
        import yaml

        from jev_rag.config import load_config
        from jev_rag.retrieval import build_index

        config_path = ROOT / "configs/transformer.yaml"
        config = load_config(config_path)
        config.embedding.device = args.device
        config.paths.processed = target
        config.paths.manifest = target / "sources.jsonl"
        config.qdrant.path = target / "qdrant"
        metadata = build_index(config)
        # Bind this user's fresh index metadata, never edit frozen experiment data.
        manifest_path = ROOT / "knowledge/llm_wiki/manifest.json"
        manifest = json.loads(manifest_path.read_text("utf8"))
        manifest["source_binding"] = store.binding()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf8")
        portable = load_config(config_path).model_dump(mode="json")
        portable["embedding"]["device"] = args.device
        config_path.write_text(yaml.safe_dump(portable, allow_unicode=True, sort_keys=False), "utf8")
        receipt = ROOT / ".local/source-index-restoration.json"
        receipt.parent.mkdir(exist_ok=True)
        receipt.write_text(json.dumps({"kind": "NEW_LOCAL_INDEX_NOT_HISTORICAL_BENCHMARK",
            "provider_calls": 0, "metadata": metadata}, ensure_ascii=False, indent=2), "utf8")
    ArticleStore(ROOT / "knowledge/llm_wiki", source_store=store)
    print(json.dumps({"sources": len(store.sources), "chunks": len(store.chunks),
                      "provider_calls": 0, "source_index_built": args.build_source_index}))


if __name__ == "__main__":
    main()
