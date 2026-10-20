# Jev Agentic Wiki

A Transformer knowledge assistant built around **complete Markdown Wiki articles** and **lightweight Graph RAG**, organizing papers, official documentation and implementation references into a searchable knowledge base for reading and exploring related topics.

The collection currently contains 56 source records, organized into 35 complete articles and 137 directed topic links. It covers:

- **Model architecture and training:** Transformer, BERT, T5, encoders and decoders, self-attention and cross-attention, causal masking, MLM and the forward computation flow.
- **Core components:** residual connections, LayerNorm, RMSNorm, BatchNorm, GELU, GLU/SwiGLU, embeddings, and BPE, SentencePiece and WordPiece tokenization.
- **Positional encoding:** RoPE/RoFormer, ALiBi and position representations.
- **Efficient inference and serving:** FlashAttention/FlashAttention-2, MQA/GQA, KV Cache, PagedAttention, continuous batching and Prefill/Decode.
- **Implementation and measurement:** official PyTorch, Hugging Face Transformers and Tensor2Tensor implementations, plus mygpt code and performance experiments.

Representative papers include *Attention Is All You Need*, *BERT*, *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*, *RoFormer*, *FlashAttention*, *GQA* and *Efficient Memory Management for Large Language Model Serving with PagedAttention*. Click the static demo below to search and browse Wiki articles, inspect source information, follow related topics and explore ten recorded example questions. No installation or local deployment is required.

[Try it — no installation required](https://tancy-920.github.io/jev-agentic-wiki-demo/) · [Reproduction guide](experiments/wiki_graph_value/README.md) · [MIT License](LICENSE) · [Publication contents](PUBLICATION.md)

## Results at a glance

The table highlights measured benefits from two experiments. Scope and other comparison results are discussed in Limitations below.

| Experiment | Baseline → modified system | Measured effect | Data |
|---|---|---|---|
| A · Answer coverage | Original-document RAG → complete-article Wiki | Mean per-question coverage **66.67% → 76.67%** (+10 percentage points), six development questions | [JSON](experiments/wiki_graph_value/results/bench_wiki_vs_rag.json) |
| A · Integration questions | Same comparison, four-question subset | **75% → 90%** coverage (+15 pp) | Same JSON |
| A · Generation input | Same six paired questions | Mean paired input-token reduction **59.89%** | Same JSON, per-question usage |
| A · Warm local retrieval | Dense + BM25 + reranker → article matching | Per-question median **539–689 ms → 1.45–1.84 ms** | [30 samples per arm/question](experiments/wiki_graph_value/data/retrieval/) |
| B · Concept exploration | Wiki keyword recommendations → Wiki + Graph RAG | Target-topic coverage in six recommendations **24% → 52%** (+28 pp) | [JSON](experiments/wiki_graph_value/results/bench_graph_navigation.json) |

Coverage is calculated from evaluation items for each question, then averaged. Input-token reduction is calculated for paired questions. Warm retrieval timing measures the local retrieval stage. Headline results are traceable to committed JSON, archived answers and scoring rules. See Limitations for experimental scope and applicable conditions.

## Purpose

The primary contribution is an inspectable evidence trail for two different roles:

- **Wiki as answer context:** retrieve coherent, complete knowledge units instead of assembling isolated original-document chunks.
- **Lightweight Graph RAG as knowledge navigation:** suggest related articles from a starting concept, especially when a reader has not yet formulated a detailed query.

JEV is an optional product feature for choosing answer depth, alongside manual selection. The experiments above focus on Wiki context organization and related-topic navigation.

## System and setup

```text
Original sources ── offline curation ── 35 complete Markdown Wiki articles
                                         │ titles, aliases, article content
                                         ▼
Question ── lightweight article matching ── complete articles ── LLM answer
                                         │
                                         └── 137 directed related-topic links
                                             └── reading/navigation suggestions

Separate RAG baseline:
Question ── dense + BM25 ── fusion + cross-encoder ── original chunks ── LLM
```

The ordinary Wiki path bypasses dense chunk retrieval, BM25 and cross-encoder reranking. This project's lightweight Graph RAG organizes complete articles through `related` links, uses article matching to provide answer context, and provides reading navigation through directed links and multi-hop paths.

The frozen corpus contains 56 source records and 1,145 chunks, plus 35 complete Wiki articles. Experiment A shares source provenance, neutral plans, DeepSeek Flash/none, temperature 0, a 2,048-token output cap and a 24,000-character evidence cap. Evidence organization and retrieval are the tested changes.

Experiment B runs entirely offline on eight tasks with six distinct entry queries. Both arms have six recommendation slots; the graph arm preserves the first three keyword recommendations, then prioritizes related nodes within three hops. Reference targets are used for scoring, never ranking. After excluding the opened starting concept, there are 25 target occurrences; the counting unit is a target occurrence.

## A — Complete Wiki articles versus original-document RAG

| Question | Original RAG supported items | Wiki supported items | Input tokens: RAG → Wiki |
|---|---:|---:|---:|
| N01 | 4/4 | 4/4 | 7,306 → 1,838 |
| N02 | 4/4 | 4/4 | 7,232 → 3,260 |
| N03 | 3/5 | 4/5 | 6,691 → 3,679 |
| N04 | 0/4 | 0/4 | 8,071 → 3,467 |
| N05 | 4/4 | 4/4 | 7,046 → 1,998 |
| N06 | 2/5 | 4/5 | 8,259 → 3,647 |

Coverage is averaged over questions, rather than pooling all items. Token reduction is the mean of `(RAG input − Wiki input) / RAG input` for each pair. Exact questions, responses, quotes, source evidence and scoring records are included in the [experiment data](experiments/wiki_graph_value/data/).

The frozen study used 18 physical requests across planning, generation and product checks. The published fixed-context generation subset has 14 slots and 13 distinct request contracts. Identical requests were reused and are not independent observations. Token usage follows the corresponding physical request, including replayed slots. Unknown monetary cost is `null`.

## B — Lightweight Graph RAG for topic navigation

| Query condition | Keyword-only hits | Graph-assisted hits | Interpretation |
|---|---:|---:|---|
| Starting title and aliases | 6/25 (24%) | 13/25 (52%) | Graph links expand discovery from a sparse entry point |
| Starting title, permuted graph | — | 6/25 (24%) | One fixed topology-preserving semantic permutation; diagnostic only |

For entry exploration, five tasks improve and three tie. With no edges, the graph implementation falls back exactly to keyword-only recommendations. Three-hop graph reachability is 766/1,190 ordered distinct-node pairs (64.37%), describing the structural coverage of the topic network.

Lightweight Graph RAG supports further reading from a starting concept: readers can follow related links to discover relevant knowledge and combine this with keyword search to build an understanding of a topic.

## Reproduce the experiments

Python 3.11 or newer. No API key or third-party Python package is needed for the default reproduction:

```bash
python -S experiments/wiki_graph_value/reproduce.py
python -S experiments/wiki_graph_value/benchmark_retrieval.py --check-inputs
python -S -m unittest discover -s experiments/wiki_graph_value/tests -v
```

This checks file hashes, recomputes archived coverage/token/timing metrics, and reruns navigation rankings, paths, empty-graph and permutation controls. Reproduction uses frozen inputs, scoring records and experimental rules. Outputs are written to an ignored `outputs/` directory.

To rebuild local embeddings and reranking from the published text, follow the [retrieval reproduction instructions](experiments/wiki_graph_value/README.md#rebuild-retrieval-without-an-llm-api). To rerun fixed-context generation, follow the explicit [paid benchmark-live instructions](experiments/wiki_graph_value/README.md#optional-paid-generation-rerun). You supply your own DeepSeek key and finite request cap; new answers require fresh source and scoring checks. Provider updates can change outputs even at temperature 0.

## Try the application

[Open the static demo](https://tancy-920.github.io/jev-agentic-wiki-demo/) to use Wiki search, article navigation, formulas and ten example questions directly in your browser. No installation, local deployment or API key is required. Accessing the webpage requires an internet connection; browsing makes no model API calls. The static demo does not generate answers to new questions. The interface and knowledge content are currently in Chinese; this README documents the experiments in English.

**Local setup (optional):** to host the application yourself, modify the project or enable live questions, run the local Streamlit application. Create and activate a virtual environment, then run:

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
experiments/wiki_graph_value/  Frozen JSON, inputs, scoring records, standalone reproduction
src/jev_rag/                   Local orchestration, Wiki, retrieval and provider adapters
frontend/                     Streamlit application and recorded demo fixtures
knowledge/llm_wiki/            Current complete articles and topic links
hosted-demo/assets/            Static UI source, local fonts and dependency licenses
scripts/                      Curated build, restore and live-session entrypoints
configs/                      Current application and baseline configuration
PUBLICATION.md                Included files, exclusions and publication boundaries
```

## Limitations

- Current results come from six QA development questions and eight navigation tasks. The navigation tasks contain six distinct entry queries, and two tasks share the same entry and targets. The sample is small; the effects need further testing on a broader range of independent tasks.
- Answer quality still has room for improvement: both arms failed to locate the required actual function body in N04, and the N06 Wiki answer made an unsupported converse inference about memory reduction. The complete quality/efficiency acceptance condition has not yet been met. Per-question data and failure records are preserved in the [experiment data](experiments/wiki_graph_value/data/).
- The current benefit of lightweight Graph RAG is concentrated in concept-entry exploration. For complete-question search, target coverage fell from 96% to 84%, showing that related-topic recommendations need to be coordinated with direct matching. Historical QA comparisons have not established stable overall gains; the [historical experiment records](experiments/wiki_graph_value/data/history/) remain available. The current implementation centers on navigation through article links.
- Warm retrieval timing is hardware-dependent and excludes cold loading and model generation. Wiki authoring/maintenance cost, full end-to-end latency and monetary savings were not measured. Actual-user completion time and click efficiency also remain to be tested.
- Default offline results are recalculations over frozen records. New retrieval runs may change rankings across devices; provider updates may change generated outputs and scores, even at temperature 0.

## License

The project's own code and documentation are available under the [MIT License](LICENSE), allowing use, modification and distribution subject to the terms in the license file.

Third-party papers, documentation excerpts, code, fonts and dependencies retain their respective copyrights and licenses. The MIT License does not change ownership or rights in those materials. Downloaded original PDFs and model weights are excluded from the release.
