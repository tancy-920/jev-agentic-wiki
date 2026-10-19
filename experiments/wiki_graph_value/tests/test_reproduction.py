"""Portable network-denied artifact checks, usable without pytest or GPU libraries."""
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce  # noqa: E402
from benchmark_retrieval import validate_corpus  # noqa: E402
from navigation import ArticleStore, recommend, shortest_paths  # noqa: E402


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = reproduce.ROOT / "outputs/test-temp"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.no_network = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden in tests"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)

    def test_complete_offline_reproduction_and_exact_published_results(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            output = Path(temp) / "results"
            reproduce.offline(output)
            self.assertTrue(json.loads((output / "verification.json").read_text(encoding="utf8"))["results_match"])
            for name in ["bench_wiki_vs_rag.json", "bench_graph_navigation.json"]:
                self.assertEqual(json.loads((output / name).read_text(encoding="utf8")), reproduce.read("results/" + name))

    def test_corpus_inputs_have_exact_normalized_source_spans(self):
        proof = validate_corpus()
        self.assertEqual(proof["sources"], 56)
        self.assertGreater(proof["chunks"], 408)

    def test_directed_paths_and_empty_graph_fallback(self):
        self.assertEqual(shortest_paths({"a": {"b"}, "b": set()}, "b"), {"b": ["b"]})
        store = ArticleStore(reproduce.ROOT / "data/wiki")
        direct, graph, _ = recommend(store, {i: set() for i in store.pages}, "gqa", "GQA")
        self.assertEqual(direct, graph)
        self.assertEqual(len(set(graph)), 6)

    def test_manifest_rejects_tampered_input(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp:
            root = Path(temp)
            (root / "input.txt").write_text("changed", encoding="utf8")
            (root / "MANIFEST.json").write_text(json.dumps({"files": {"input.txt": "0" * 64}}), encoding="utf8")
            with patch.object(reproduce, "ROOT", root), self.assertRaisesRegex(ValueError, "Artifact changed"):
                reproduce.verify()

    def test_live_requires_finite_budget_before_provider_import(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as temp, self.assertRaisesRegex(ValueError, "max-calls"):
            reproduce.live(0, Path(temp) / "new")


if __name__ == "__main__":
    unittest.main()
