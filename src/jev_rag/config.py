from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, Field, model_validator

from jev_rag.agent.intent import IntentPolicy
from jev_rag.agent.language import TranslationPolicy
from jev_rag.agent.verification import VerificationPolicy
from jev_rag.llm.answering import AnswerPolicy
from jev_rag.llm.structured import StructuredOutputPolicy


class Paths(BaseModel):
    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    manifest: Path = Path("data/papers.jsonl")


class ChunkConfig(BaseModel):
    tokenizer: str = "cl100k_base"
    size: int = Field(default=600, ge=1)
    overlap: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.overlap >= self.size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    revision: str | None = None
    local_files_only: bool = True
    device: str = "cpu"
    batch_size: int = Field(default=16, ge=1)
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    window_tokens: int = Field(default=450, ge=1)
    window_overlap: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.window_overlap >= self.window_tokens:
            raise ValueError("embedding overlap must be smaller than window size")
        return self


class QdrantConfig(BaseModel):
    mode: str = "local"
    path: Path = Path("data/processed/qdrant")
    url: str = "http://localhost:6333"


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=20, ge=1)
    dense_k: int = Field(default=40, ge=1)
    sparse_k: int = Field(default=40, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_revision: str | None = None
    reranker_local_files_only: bool = True
    reranker_window_overlap: int = Field(default=50, ge=0)
    reranker_enabled: bool = False


class AgentConfig(BaseModel):
    max_iterations: int = Field(default=5, ge=1, le=5)
    jev_mode: str = "mock"
    web_search_enabled: bool = False
    relevance_uncertainty: Literal["exclude", "stop"] = "exclude"
    decision_threshold: float = Field(default=0.85, ge=0, le=1)


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    strong_model: str = "deepseek-v4-pro"
    api_format: Literal["responses", "chat_completions"] = "responses"
    timeout_seconds: float = Field(default=60.0, gt=0, le=120)
    max_output_tokens: int = Field(default=4096, ge=1)
    fast_reasoning_effort: Literal["none", "low", "high", "max"] | None = None
    strong_reasoning_effort: Literal["none", "low", "high", "max"] | None = None
    max_claims: int = Field(default=30, ge=1, le=30)

    @model_validator(mode="after")
    def validate_endpoint(self):
        url = urlsplit(self.base_url)
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ValueError("LLM base URL must be a configured HTTPS endpoint without credentials")
        if url.query or url.fragment:
            raise ValueError("LLM base URL cannot contain query or fragment")
        if not self.model.strip() or not self.strong_model.strip():
            raise ValueError("both model tiers must be configured")
        return self

    def with_environment(self, environment):
        """Explicit overrides; never read keys into serializable configuration."""
        fields = {
            "LLM_PROVIDER": "provider",
            "LLM_BASE_URL": "base_url",
            "LLM_MODEL": "model",
            "LLM_STRONG_MODEL": "strong_model",
            "LLM_API_FORMAT": "api_format",
        }
        values = self.model_dump()
        values.update(
            {field: environment[key] for key, field in fields.items() if environment.get(key)}
        )
        return type(self).model_validate(values)


class JevConfig(BaseModel):
    endpoint: Literal["https://api.typesafe.ai/v1/systemone"] = (
        "https://api.typesafe.ai/v1/systemone"
    )
    model: str = Field(default="jev-latest", min_length=1)
    timeout_seconds: float = Field(default=60.0, gt=0, le=120)
    relevance_batch_size: int = Field(default=20, ge=1, le=20)


class Config(BaseModel):
    seed: int = 42
    paths: Paths = Field(default_factory=Paths)
    chunking: ChunkConfig = Field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    jev: JevConfig = Field(default_factory=JevConfig)
    answer: AnswerPolicy = Field(default_factory=AnswerPolicy)
    intent: IntentPolicy = Field(default_factory=IntentPolicy)
    translation: TranslationPolicy = Field(default_factory=TranslationPolicy)
    verification: VerificationPolicy = Field(default_factory=VerificationPolicy)
    structured_output: StructuredOutputPolicy = Field(default_factory=StructuredOutputPolicy)


def load_config(path: str | Path) -> Config:
    return Config.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
