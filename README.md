# Jev Agentic Wiki

A Transformer knowledge assistant built around **complete Markdown Wiki articles** and a lightweight topic graph. This project focuses on measured comparisons: does organizing evidence into complete knowledge units improve answers, and when do graph links help readers discover related topics? Every headline result below is backed by committed JSON, archived answers and explicit evaluation rules.

[Explore the offline demo](https://tancy-920.github.io/jev-agentic-wiki-demo/) · [Reproduction guide](experiments/wiki_graph_value/README.md) · [Publication contents](PUBLICATION.md)

## Results at a glance

| Experiment | Baseline → modified system | Measured effect | Data |
|---|---|---|---|
| A · Answer coverage | Original-document RAG → complete-article Wiki | Mean per-question coverage **66.67% → 76.67%** (+10 percentage points), six development questions | [JSON](experiments/wiki_graph_value/results/bench_wiki_vs_rag.json) |
| A · Integration questions | Same comparison, four-question subset | **75% → 90%** coverage (+15 pp) | Same JSON |
| A · Generation input | Same six paired questions | Mean paired input-token reduction **59.89%** | Same JSON, per-question usage |
| A · Warm local retrieval | Dense + BM25 + reranker → article matching | Per-question median **539–689 ms → 1.45–1.84 ms** | [30 samples per arm/question](experiments/wiki_graph_value/data/retrieval/) |
| B · Concept exploration | Wiki keyword recommendations → Wiki + graph | Target-topic coverage in six recommendations **24% → 52%** (+28 pp) | [JSON](experiments/wiki_graph_value/results/bench_graph_navigation.json) |
| B · Full-question search | Same graph intervention, full intent available | Coverage **96% → 84%** (−12 pp) | Same JSON, boundary condition |

These are small, exposed development experiments, with source-grounded, non-blind AI judgments rather than human-attested ground truth. Coverage is **rubric coverage, not general factual accuracy**. Retrieval timing excludes generation and cold loading. The Wiki experiment also contains an unsupported inference, so its complete quality/efficiency acceptance condition did not pass. Graph navigation gains do not establish better generated answers.

## Purpose

The primary contribution is an inspectable evidence trail for two different roles:

- **Wiki as answer context:** retrieve coherent, complete knowledge units instead of assembling isolated original-document chunks.
- **Graph as knowledge navigation:** suggest related articles from a starting concept, especially when a reader has not yet formulated a detailed query.

JEV is an optional product feature for automatically choosing answer depth. Manual depth selection remains available. JEV is excluded from these experiments; no JEV quality, speed or cost advantage is claimed here.

## System and setup

```text
Original sources ── offline curation ── 35 complete Markdown Wiki articles
                                         │ titles, aliases, article content
                                         ▼
Question ── lightweight article matching ── complete articles ── LLM answer
                                         │
                                         └── 137 directed related-topic links
                                             └── reading/navigation suggestions

Separate source baseline:
Question ── dense + BM25 ── fusion + cross-encoder ── original chunks ── LLM
```

The ordinary Wiki path bypasses dense chunk retrieval, BM25 and cross-encoder reranking. The graph connects articles through `related` links; it is not a learned relation extractor, a causal knowledge graph or a full GraphRAG pipeline.

The frozen corpus contains 56 source records and 1,145 chunks, plus 35 complete Wiki articles. Experiment A shares source provenance, neutral plans, DeepSeek Flash/none, temperature 0, a 2,048-token output cap and a 24,000-character evidence cap. Evidence organization and retrieval are the tested changes. The two arms do not differ by only a prompt flag.

Experiment B runs entirely offline on eight tasks with six distinct entry queries. Both arms have six recommendation slots; the graph arm preserves the first three keyword recommendations, then prioritizes related nodes within three hops. Reference targets are used for scoring, never ranking. After excluding the opened starting concept, there are 25 target occurrences, not 25 independent users.

## A — Complete Wiki articles versus original-document RAG

| Question | Original RAG supported items | Wiki supported items | Input tokens: RAG → Wiki |
|---|---:|---:|---:|
| N01 | 4/4 | 4/4 | 7,306 → 1,838 |
| N02 | 4/4 | 4/4 | 7,232 → 3,260 |
| N03 | 3/5 | 4/5 | 6,691 → 3,679 |
| N04 | 0/4 | 0/4 | 8,071 → 3,467 |
| N05 | 4/4 | 4/4 | 7,046 → 1,998 |
| N06 | 2/5 | 4/5 | 8,259 → 3,647 |

Coverage is averaged over questions, rather than pooling all items. Token reduction is the mean of `(RAG input − Wiki input) / RAG input` for each pair. N04 failed to locate the requested actual function body in both arms. N06 Wiki made an unsupported converse inference about memory reduction; higher rubric coverage does not erase this error. Exact questions, responses, quotes, source evidence and judgments are included in the [experiment data](experiments/wiki_graph_value/data/).

The frozen study used 18 physical requests across planning, generation and product checks. The published fixed-context generation subset has 14 slots and 13 distinct request contracts. Identical requests were reused and are not independent observations. Token usage follows the corresponding physical request, including replayed slots. Unknown monetary cost is `null`.

## B — Lightweight graph for topic navigation

| Query condition | Keyword-only hits | Graph-assisted hits | Interpretation |
|---|---:|---:|---|
| Starting title and aliases | 6/25 (24%) | 13/25 (52%) | Graph links expand discovery from a sparse entry point |
| Complete original question | 24/25 (96%) | 21/25 (84%) | Forced neighbors can displace useful direct matches |
| Starting title, permuted graph | — | 6/25 (24%) | One fixed topology-preserving semantic permutation; diagnostic only |

For entry exploration, five tasks improve and three tie. Two tasks share the same entry/targets; the eight rows are not eight independent user trials. With no edges, the graph implementation falls back exactly to keyword-only recommendations. Three-hop graph reachability is 766/1,190 ordered distinct-node pairs (64.37%); this is a structural property, not measured click time.

Earlier graph question-answering studies did **not** establish stable overall superiority. Their [zero/negative results and AI reviews](experiments/wiki_graph_value/data/history/) remain available. The supported use is related-topic exploration, not replacing strong explicit search or claiming a general answer-quality gain.

## Reproduce the experiments

Python 3.11 or newer. No API key or third-party Python package is needed for the default reproduction:

```bash
python -S experiments/wiki_graph_value/reproduce.py
python -S experiments/wiki_graph_value/benchmark_retrieval.py --check-inputs
python -S -m unittest discover -s experiments/wiki_graph_value/tests -v
```

This checks file hashes, recomputes archived coverage/token/timing metrics, and reruns navigation rankings, paths, empty-graph and permutation controls. It does not independently regenerate factual labels. Outputs are written to an ignored `outputs/` directory.

To rebuild local embeddings and reranking from the published text, follow the [retrieval reproduction instructions](experiments/wiki_graph_value/README.md#rebuild-retrieval-without-an-llm-api). To rerun fixed-context generation, follow the explicit [paid benchmark-live instructions](experiments/wiki_graph_value/README.md#optional-paid-generation-rerun). You supply your own DeepSeek key and finite request cap; new answers require new review. Provider updates can change outputs even at temperature 0.

## Try the application

The [public offline demo](https://tancy-920.github.io/jev-agentic-wiki-demo/) offers Wiki search, article navigation, formulas and ten example questions. It has no chat input or model API calls. The interface and knowledge content are currently in Chinese; this README documents the experiments in English.

For the local Streamlit demo, create and activate a virtual environment, then run:

```bash
python -m pip install -e ".[agent]"
python -m streamlit run frontend/app.py
```

Open `http://localhost:8501/?mode=simulate` for provider-free browsing and recorded examples. No source snapshots or neural weights are needed for this mode.

For **new live questions**, restore the original source snapshots listed in the published corpus catalog. They are not bundled as downloaded PDFs. Preserve the listed filenames and SHA256 values in a local snapshot directory, then run:

```bash
python scripts/prepare_local_workspace.py --list-snapshots
python scripts/prepare_local_workspace.py --snapshots /path/to/verified/snapshots
```

The helper validates every original snapshot and the published normalized spans before installing the local corpus. An unavailable or changed original is an explicit setup failure; it is not silently accepted. Local mygpt source origins are historical provenance, not required absolute paths: restoration uses snapshot filenames.

Copy `.env.example` to `.env.local` and set your own `DEEPSEEK_API_KEY`. In the live UI, explicitly authorize a bounded session (20 or 40 physical requests) before sending a question. Selecting the interface or setting a key alone does not enable model calls. Optional automatic answer depth additionally requires your own JEV credentials and explicit transmission permission. Failures consume the request budget; transport retries are disabled. The inherited weekday Beijing-time 09:00–12:00 and 14:00–18:00 request restrictions remain in effect.

The Wiki path does not need neural retrieval weights. The source-RAG option additionally needs the pinned embedding/reranking models. Install the experiment's `requirements-retrieval.txt` and cache its two pinned models, then run `python scripts/prepare_local_workspace.py --build-source-index --device cpu` (or `cuda`). This builds a new index and rebinds the current local article manifest to its descriptor; it never rewrites the frozen experimental artifact. New local indexes/timings are not historical benchmark runs. Never commit `.env.local`, session authorization files, generated indexes or weights.

## Build the static demo

Node.js 22 and Python 3.11+:

```bash
npm --prefix hosted-demo ci --ignore-scripts
python scripts/build_static_demo.py
python scripts/build_static_demo.py --check
python -m http.server 8510 --directory hosted-demo/dist
```

The static export shares example selection and answer formatting with the local UI. Source changes require a rebuild and redeployment. Editing local files does not automatically change the published site.

## Repository layout

```text
experiments/wiki_graph_value/  Frozen JSON, inputs, judgments, standalone reproduction
src/jev_rag/                   Local orchestration, Wiki, retrieval and provider adapters
frontend/                     Streamlit application and recorded demo fixtures
knowledge/llm_wiki/            Current complete articles and topic links
hosted-demo/assets/            Static UI source, local fonts and dependency licenses
scripts/                      Curated build, restore and live-session entrypoints
configs/                      Current application and baseline configuration
PUBLICATION.md                Included files, exclusions and publication boundaries
```

## Limitations

- Six QA development questions and eight navigation tasks cannot establish broad or statistically significant superiority. Labels are non-blind AI judgments, with no released human-attested ground truth or independent held-out benchmark.
- Warm retrieval timing is hardware-dependent and excludes cold loading and model generation. Wiki authoring/maintenance cost, full end-to-end latency and monetary savings were not measured.
- The graph has a useful navigation niche, but may reduce explicit-search coverage. No actual-user completion time, click efficiency or exhaustive semantic precision was measured.
- Default offline results are reproducible calculations over frozen records. New retrieval runs may change rankings across devices; new provider outputs and factual scores are not guaranteed to match.
- Attributed third-party text/code retains its original provenance and applicable rights; no new license is granted over third-party material. Downloaded PDFs and model weights are excluded.
