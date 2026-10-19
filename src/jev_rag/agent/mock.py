"""Offline fixtures only; their judgments are not evidence or ground truth."""

from collections import defaultdict, deque

from jev_rag.llm.answering import AnswerPolicy
from jev_rag.retrieval import lexical_tokens

from .contracts import Claim, Decision, DecisionRequest, Draft, ProviderError
from .intent import RetrievalPlan
from .language import QueryTranslation, TranslationDraft, digest, has_chinese
from .verification import VerificationPolicy, validate_draft


class MockDecisions:
    source = "MOCK"

    def __init__(self, scripted: dict[str, list] | None = None):
        self.scripted = defaultdict(deque)
        for task, values in (scripted or {}).items():
            self.scripted[task].extend(values)
        self.requests = []

    def decide(self, request: DecisionRequest):
        self.requests.append(request)
        if self.scripted[request.task]:
            result = self.scripted[request.task].popleft()
            if isinstance(result, Exception):
                raise result
            if isinstance(result, str):
                return Decision(choice=result, confidence=0.99, probability=0.99, source="MOCK")
            return result
        choices = {
            "intent": "CONCEPT_EXPLANATION",
            "routing": "LOCAL_RAG",
            "relevance": "RELEVANT",
            "evidence": "SUFFICIENT" if request.evidence else "INSUFFICIENT",
            "loop": "STOP" if request.evidence_sufficient else "CONTINUE",
            "citation": "SUPPORTED",
        }
        if request.task == "relevance":
            terms = set(lexical_tokens(request.query))
            choices["relevance"] = (
                "RELEVANT"
                if any(terms.intersection(lexical_tokens(c.text)) for c in request.evidence)
                else "IRRELEVANT"
            )
        return Decision(
            choice=choices[request.task], confidence=0.99, probability=0.99, source="MOCK"
        )


class MockGenerator:
    source = "MOCK"

    def __init__(
        self,
        drafts: list | None = None,
        rewrites: list[str] | None = None,
        plans: list | None = None,
        answer_policy: AnswerPolicy | None = None,
        translations: list | None = None,
        verification_policy: VerificationPolicy | None = None,
    ):
        self.drafts = deque(drafts or [])
        self.rewrites = deque(rewrites or [])
        self.plans = deque(plans or [])
        self.plan_requests = []
        self.answer_policy = answer_policy or AnswerPolicy()
        self.translations = deque(translations or [])
        self.verification_policy = verification_policy or VerificationPolicy()

    def translate_query(self, query, policy):
        if not has_chinese(query):
            return QueryTranslation(
                english_query=query,
                original_sha256=digest(query),
                policy_sha256=policy.snapshot()["hash"],
                source="IDENTITY",
            )
        if not self.translations:
            raise ProviderError("MOCK translation fixture required; no fabricated translation")
        value = self.translations.popleft()
        if isinstance(value, Exception):
            raise value
        draft = TranslationDraft.model_validate(value)
        return QueryTranslation(
            **draft.model_dump(),
            original_sha256=digest(query),
            policy_sha256=policy.snapshot()["hash"],
            source="MOCK",
        ).validate_for(query, policy)

    def plan(self, query, intent, policy, **kwargs):
        self.plan_requests.append({"query": query, "intent": intent, **kwargs})
        if self.plans:
            result = self.plans.popleft()
            if isinstance(result, Exception):
                raise result
            return result
        return RetrievalPlan(
            intent=intent,
            goal="MOCK interpretation only",
            evidence_needs=["MOCK evidence need"],
            queries=[kwargs.get("translated_query") or query],
            user_constraints=[],
        )

    def generate_focused(self, query, evidence, plan):
        return self.generate(query, evidence)

    def generate_repair(self, query, evidence, feedback):
        return self.generate(query, evidence)

    def rewrite(self, query, evidence, iteration):
        if self.rewrites:
            return self.rewrites.popleft()
        return f"{query} evidence iteration {iteration}"

    def generate(self, query, evidence):
        if self.drafts:
            result = self.drafts.popleft()
            if isinstance(result, Exception):
                raise result
            return result
        if not evidence:
            return Draft(claims=[Claim(text=self.answer_policy.withheld("MOCK", None))])
        draft = Draft(
            claims=[
                Claim(
                    text="MOCK 占位内容：仅检查流程，未生成或评估真实解释。"
                    if self.answer_policy.language == "zh-CN"
                    else "MOCK extract (not an evaluated answer): " + evidence[0].text[:500],
                    citations=[evidence[0].chunk_id],
                )
            ]
        )
        if self.verification_policy.mode == "paragraph":
            return draft
        value = draft.model_dump()
        for paragraph in value["claims"]:
            paragraph["spans"] = [{"text": paragraph["text"], "citations": paragraph["citations"]}]
        return validate_draft(value, self.verification_policy)
