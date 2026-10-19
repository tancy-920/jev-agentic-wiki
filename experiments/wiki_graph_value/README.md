# Wiki and Lightweight Graph: Reproducible Experiments

A standalone, inspectable artifact for the two comparisons in the [project README](../../README.md). The layout follows mygpt: headline comparisons, per-case JSON, explicit setup, raw measurements and reproduction commands. No mygpt performance numbers are reused as this project's results.

## Results at a glance

| Comparison | Recorded result | JSON |
|---|---|---|
| Original-document RAG → complete-article Wiki | Mean per-question rubric coverage 66.67% → 76.67%; integration subset 75% → 90% | `results/bench_wiki_vs_rag.json` |
| Same comparison, first-answer input | Mean paired token reduction 59.89% | Same file, paired usage |
| Same comparison, warm local retrieval | Per-question medians 539–689 ms → 1.45–1.84 ms | `data/retrieval/`, 30 samples per arm/question |
| Keyword-only → graph-assisted concept exploration | Target coverage 6/25 (24%) → 13/25 (52%) | `results/bench_graph_navigation.json` |
| Full-question search, boundary condition | 24/25 (96%) → 21/25 (84%) | Same file |

Coverage is archived source-grounded, non-blind AI review on exposed development cases, not human-attested ground truth or general accuracy. N06 Wiki contains an unsupported memory inference; N04 is 0/4 in both arms. Graph QA gains were not stable. See `data/wiki_review.json` and `data/history/` for errors and adverse evidence. JEV does not participate.

## Setup and controls

Wiki: six questions, the same neutral plans, source provenance, DeepSeek Flash/none, temperature 0, 2,048 output tokens and 24,000 evidence characters. The source arm uses local dense/BM25 fusion and cross-encoder reranking. Wiki selects complete Markdown articles, bypassing that pipeline. Evidence organization is intentionally different. The archive has 14 generation slots, 13 distinct requests; the full original study used 18 physical calls including planning and product checks. Exact replays are not independent samples. Unknown cost is `null`.

Navigation: 35 articles, 137 directed related-topic edges, eight tasks/six distinct entry queries, six slots. Preserve the top three keyword results, then prioritize neighbors within three hops. Remove the opened starting concept from target scoring. Twenty-five target occurrences are not 25 independent users. Targets never enter ranking. The fixed permutation uses seed 19; the empty graph must reproduce keyword-only results exactly. Earlier graph-QA reviews remain in `data/history/`; different study scores must not be pooled.

## Quick start: no API, no third-party packages

Python >=3.11; run inside this directory:

```bash
python -S reproduce.py
python -S benchmark_retrieval.py --check-inputs
python -S -m unittest discover -s tests -v
```

`reproduce.py` verifies every file in `MANIFEST.json`, checks review quotations against archived answers and token summaries against physical requests, recomputes rubric coverage and the medians of the original timing samples, and reruns navigation rankings, paths, structure, permutation and empty-graph controls. The two resulting JSON files must exactly match `results/bench_*.json`. Generated files go to ignored `outputs/offline/`.

This reproduces metric calculation and navigation, not independent factual review. Full prompts, answers, item-level judgments and source excerpts are available for external review; `human_attested=false` remains unchanged.

## Rebuild retrieval without an LLM API

```bash
python -m venv .venv
# Activate the environment; choose a CPU/CUDA PyTorch wheel for your platform.
python -m pip install -r requirements-retrieval.txt
```

`environment.json` records original installed versions. Requirements pin portable versions; the original CUDA build suffix is recorded separately. Explicitly fetch the pinned open-source models once into the Hugging Face cache (weight downloads, not LLM API requests):

```python
from huggingface_hub import snapshot_download
snapshot_download("BAAI/bge-small-en-v1.5", revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
snapshot_download("cross-encoder/ms-marco-MiniLM-L-6-v2", revision="233902d25c440f23af6f7d6e94d2946bac0bee0a")
```

```bash
python benchmark_retrieval.py --device cpu --warmups 3 --repetitions 30 --output outputs/retrieval-new
```

Model loading requires local cache and disables remote code. The script builds a fresh local Qdrant index from published chunks and invokes the original retrieval implementations with frozen shared plans. It saves timing samples, evidence IDs and matches against historical IDs. Hardware, versions and floating-point differences can affect time and ranking; report changes rather than assigning old answer scores to new contexts.

The package preserves 56 normalized source texts, attribution URLs, hashes and 1,145 exact chunk spans. Original PDFs are not distributed. This run validates normalized input and rebuilds retrieval; it does not redownload or reparse PDFs. Old timing records are never overwritten.

## Optional paid generation rerun

Install `requirements-live.txt`. Set **your own** `DEEPSEEK_API_KEY` in the process environment; never put a key in a command-line argument or repository. Inspect the exact technical content in `data/wiki_records.json` before sending it.

```bash
python reproduce.py benchmark-live --max-calls 14 --output outputs/live-new
```

Only this explicit entrypoint can call the provider. It preserves original complete request arguments, Flash/none, temperature 0, 2,048 output tokens and strict-function transport. Fourteen slots produce at most 13 distinct requests; exact requests are reused. Every network attempt consumes budget, including failures. Consumption is persisted before dispatch; failure stops the run, there are no transport retries or balance probes, and unused allowance is closed. The inherited weekday Asia/Shanghai 09:00–12:00 and 14:00–18:00 restrictions apply. A fresh output directory is mandatory. No historical grant is reopened, and returned tools are never executed.

This reruns **fixed-context generation**, not the entire original 18-request product trajectory. New answers never inherit old review labels. Provider revisions or nondeterminism may change answers even at temperature 0. Invalid responses and failures must remain in the record. Default reproduction and tests neither read credentials nor call providers.

## Data layout

```text
results/bench_*.json       Setup, per-case rows and computed summaries
data/wiki_records.json    Original prompts/answers/usage and exact request contracts
data/wiki_review.json     Item-level source-grounded AI judgments and errors
data/source_first_review.json  Source-first draft review and evidence
data/retrieval/           Archived contexts and raw timing samples
data/shared_plans.json    Fixed shared planning, no new planning API calls
data/wiki/                35 complete articles and original graph manifest
data/corpus/              Source catalog, normalized text, chunks and legacy catalog
data/history/             Adverse graph-QA reviews and navigation sensitivity evidence
vendor/src/jev_rag/        Frozen implementation for optional retrieval/live reproduction
PROVENANCE.json           Original run, export scope and accounting conventions
VALIDATION.json           Executed packaging checks and their limitations
MANIFEST.json             Per-file SHA256 for the immutable artifact
```

Intermediate narrative reports and one-off historical runners are omitted. The raw judgments and zero/negative findings needed to interpret the claims are retained. Original answers retain citations for auditing; demo prose cleanup does not rewrite scientific records.

## Validation and publication

Independent Windows extraction with Python `-S` passed metric reproduction, corpus checks and five network-denied tests. A fresh CUDA index rebuild matched historical evidence IDs in all 14 slots (zero warmups/one repetition: execution validation, not a new latency benchmark). See `VALIDATION.json`. Windows/Linux GitHub Actions workflows are provided, but were not executed remotely during packaging. The paid entrypoint was not run during export.

Publish this entire directory, including hidden `.gitattributes` and `.github` files. Do not publish `outputs/`, environments, keys, weights or indexes. Exact-byte `.gitattributes` prevents Git line-ending conversion from invalidating hashes. This directory is standalone and can also be published as its own repository; the project-level README link is optional there.

The corpus includes attributed third-party technical text/code, which retains its applicable rights; this artifact does not grant a new license over third-party materials. Wiki authoring cost, end-to-end latency, actual user navigation time and broad generalization remain unmeasured.
