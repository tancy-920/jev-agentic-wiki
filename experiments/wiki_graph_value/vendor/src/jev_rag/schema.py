import re

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator


def source_location(*, text, source_kind=None, locator=None, page=None):
    """Distinguish retained excerpt labels from normalized snapshot line numbers.

    This only formats provenance. It never changes stored locators, source text,
    chunk identities or historical serialized payloads.
    """
    location = locator or f"p. {page}"
    if source_kind != "web_excerpt":
        return location
    lines = [line for line in text.splitlines() if line.strip()]
    labels = [re.match(r"^L(\d+):", line) for line in lines]
    if lines and all(labels):
        numbers = [int(match[1]) for match in labels]
        if all(a < b for a, b in zip(numbers, numbers[1:])):
            return f"captured excerpt L{numbers[0]}–L{numbers[-1]}; snapshot {location}"
    return f"snapshot {location}"


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    year: int
    source_url: str
    pdf_url: str
    abstract: str = ""
    selection: str = "seed"
    review_status: str = "pending"
    license: str | None = None

    @field_validator("paper_id")
    @classmethod
    def safe_id(cls, value):
        if not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", value):
            raise ValueError("only modern arXiv IDs are accepted")
        return value


class Chunk(BaseModel):
    paper_id: str | None = None
    title: str
    authors: list[str]
    year: int
    section: str = "unknown"
    section_source: str = "unknown"
    page: int | None = Field(default=None, ge=1)
    chunk_id: str
    source_url: str
    text: str
    token_count: int = Field(ge=1)
    source_id: str | None = None
    source_kind: str | None = None
    source_version: str | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    locator: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def source_identity(self):
        if self.paper_id is None and not self.source_id:
            raise ValueError("paper or typed source identity required")
        if self.source_id and not all((self.source_kind, self.locator, self.source_sha256)):
            raise ValueError("typed sources require kind, locator and snapshot hash")
        if self.paper_id is None and self.source_kind == "pdf" and self.page is None:
            raise ValueError("PDF sources require the actual file page")
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("both normalized snapshot offsets required")
        if self.end_char is not None and self.end_char <= self.start_char:
            raise ValueError("invalid source span")
        return self

    @model_serializer(mode="wrap")
    def compatible_dump(self, handler):
        data = handler(self)
        # Preserve legacy paper payloads/hashes: new metadata appears only when set.
        for name in (
            "source_id",
            "source_kind",
            "source_version",
            "source_sha256",
            "locator",
            "start_char",
            "end_char",
        ):
            if data.get(name) is None:
                data.pop(name, None)
        return data

    @property
    def identity(self) -> str:
        return self.source_id or self.paper_id

    @property
    def citation(self) -> str:
        location = source_location(text=self.text, source_kind=self.source_kind,
                                   locator=self.locator, page=self.page)
        return f"[{self.title}, {self.section}, {location}]"
