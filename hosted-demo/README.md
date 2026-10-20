# Offline Wiki Demo

[Public demo](https://tancy-920.github.io/jev-agentic-wiki/)

A static, provider-free knowledge browser: 35 complete Wiki articles, topic links,
10 selected questions and 16 archived answer configurations. There is no live chat,
login or API-key input. The local Streamlit app retains the live workflow.

## Build and preview

From the repository root (Python 3.11+, Node.js 22):

```bash
npm --prefix hosted-demo ci --ignore-scripts
python scripts/build_static_demo.py
python scripts/build_static_demo.py --check
python -m http.server 8510 --directory hosted-demo/dist
```

Open `http://localhost:8510/`. Build output is ignored; publish `dist/` to a static
host, or configure the supplied GitHub Pages workflow. Hosted updates require a
new build and deployment.

## Shared presentation

`frontend/fixtures/chat_replays.json` is the immutable replay source. Example
selection, clean answer prose, citation separation and math formatting are shared
with the local UI through `demo_catalog.py`, `citation_display.py` and
`math_display.py`. The catalog is generated, not hand-edited. Layout code remains
separate between Streamlit and HTML, so layout changes need checks in both.

The builder exports an allowlist of presentation fields, never credentials,
provider request bodies or local source paths. Fonts and formula libraries are
served locally. Dependency license notices are preserved in the build.

Run `python -m pytest tests/test_static_demo.py -q` after building to verify the
catalog, source freshness and presentation boundaries. Tests forbid network access.
