"""Deterministic component substitution; provider predictions are never authorization."""

VARIANTS = {
    "A0": (),
    "A1": ("relevance",),
    "A2": ("relevance", "evidence"),
    "A3": ("relevance", "evidence", "loop"),
    "A4": ("intent", "routing", "relevance", "evidence", "loop", "citation"),
}
TASKS = set(VARIANTS["A4"])


class MixedDecisions:
    """The selected variant is application configuration, not a model output."""

    def __init__(self, llm, jev, variant):
        if (
            variant not in VARIANTS
            or llm.source != jev.source
            or llm.source not in {"LIVE", "MOCK"}
        ):
            raise ValueError("known variant and same-mode providers required")
        self.llm, self.jev, self.variant = llm, jev, variant
        self.source = llm.source
        self._gate = getattr(llm, "_gate", None)
        if self.source == "LIVE" and (
            self._gate is None or self._gate is not getattr(jev, "_gate", None)
        ):
            raise ValueError("both providers must share the same application gate")
        self.dispatches = []

    def _provider(self, task, count):
        if task not in TASKS:
            raise ValueError("unknown bounded decision task")
        selected = "jev" if task in VARIANTS[self.variant] else "llm"
        self.dispatches.append({"task": task, "provider": selected, "count": count})
        return self.jev if selected == "jev" else self.llm

    def decide(self, request):
        return self._provider(request.task, 1).decide(request)

    def decide_many(self, requests):
        if not requests or any(r.task != "relevance" for r in requests):
            raise ValueError("only independent relevance requests can be batched")
        provider = self._provider("relevance", len(requests))
        if hasattr(provider, "decide_many"):
            return provider.decide_many(requests)
        return [provider.decide(r) for r in requests]
