"""One shared selection for the local and hosted offline demo."""

FEATURED_CASES = {
    "D01": ("Transformer 编码器", "基础架构"),
    "D03": ("FFN 与激活函数", "公式与机制"),
    "D07": ("KV Cache 显存优化", "推理优化"),
    "H01": ("梯度消失与爆炸", "训练机制"),
    "H02": ("BERT 与掩码", "预训练"),
    "H03": ("SDPA 与 FlashAttention", "代码实现"),
    "H04": ("MHA / MQA / GQA", "注意力结构"),
    "H05": ("位置编码", "位置表示"),
    "H11": ("RMSNorm 与 LayerNorm", "归一化"),
    "H17": ("FlashAttention 的用途", "高效注意力"),
}


def featured_recordings(recordings):
    return [record for case in FEATURED_CASES for record in recordings if record["case_id"] == case]


# Reading suggestions, independent of the historical answer's cited evidence.
CASE_TOPICS = {
    "D01": ["transformer", "self-attention", "residual", "layernorm"],
    "D03": ["mlp", "activations"],
    "D07": ["kv-cache", "paged-kv", "gqa", "mqa"],
    "H01": ["training", "residual", "normalization"],
    "H02": ["bert-mlm", "padding-mask", "causal-mask"],
    "H03": ["sdpa", "flash-attention"],
    "H04": ["mha", "mqa", "gqa"],
    "H05": ["position", "rope"],
    "H11": ["layernorm", "normalization"],
    "H17": ["flash-attention", "online-softmax", "sdpa"],
}


def wiki_articles():
    import hashlib
    import json
    from pathlib import Path

    from frontend.citation_display import article_text

    root = Path(__file__).resolve().parents[1] / "knowledge/llm_wiki"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    articles = []
    for entry in manifest["articles"]:
        identifier = entry["article_id"]
        path = root / (identifier + ".md")
        if path.parent != root or path.name != entry["markdown_file"]:
            raise ValueError("Invalid article path")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Article checksum changed")
        articles.append(dict(id=identifier, title=entry["title"], aliases=entry["aliases"],
                             related=entry["related"], text=article_text(raw.decode("utf-8"))))
    valid = {a["id"] for a in articles}
    if any(r not in valid for a in articles for r in a["related"]):
        raise ValueError("Unknown graph destination")
    return articles


def search_articles(articles, query):
    terms = query.lower().strip().split()
    if not terms:
        return articles
    scored = []
    for article in articles:
        title = " ".join([article["id"], article["title"], *article["aliases"]]).lower()
        body = article["text"].lower()
        if all(term in title or term in body for term in terms):
            score = sum(10 if term in title else 1 for term in terms)
            scored.append((score, article))
    return [a for _, a in sorted(scored, key=lambda pair: -pair[0])]
