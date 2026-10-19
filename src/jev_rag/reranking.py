import numpy as np


class LocalReranker:
    """Maximum relevance over full-passage windows; pair truncation is avoided."""

    def __init__(self, config, device):
        from sentence_transformers import CrossEncoder

        self.config = config
        self.model = CrossEncoder(
            config.reranker_model,
            device=device,
            revision=config.reranker_revision,
            local_files_only=config.reranker_local_files_only,
            trust_remote_code=False,
            max_length=512,
        )

    def query_budget(self, query):
        tokenizer = self.model.tokenizer
        query_tokens = tokenizer.encode(query, add_special_tokens=False, verbose=False)
        return 512 - len(query_tokens) - tokenizer.num_special_tokens_to_add(pair=True)

    def supports_query(self, query):
        return self.query_budget(query) > self.config.reranker_window_overlap

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        tokenizer = self.model.tokenizer
        budget = self.query_budget(query)
        if budget <= self.config.reranker_window_overlap:
            raise ValueError("query exceeds reranker budget; shorten it explicitly")
        pairs, owners = [], []
        for owner, text in enumerate(passages):
            tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            if not tokens:
                pairs.append((query, ""))
                owners.append(owner)
            for start in range(0, len(tokens), budget - self.config.reranker_window_overlap):
                pairs.append((query, tokenizer.decode(tokens[start : start + budget])))
                owners.append(owner)
                if start + budget >= len(tokens):
                    break
        predictions = self.model.predict(pairs, batch_size=16, show_progress_bar=False)
        output = np.full(len(passages), -np.inf)
        for owner, prediction in zip(owners, predictions, strict=True):
            value = float(prediction)
            if not np.isfinite(value):
                raise ValueError("reranker returned a non-finite score")
            output[owner] = max(output[owner], value)
        return output.tolist()
