"""M4 decision-provider substitution, using the same Jev task rubrics and batches."""

import json

from jev_rag.agent.contracts import ProviderError
from jev_rag.jev.provider import choice_payload, parse_answer, validate_request

from .structured import parse_json


class LLMDecisions:
    payload_builder = staticmethod(choice_payload)

    def __init__(self, adapter, policy, relevance_batch_size=20, format_version="json-envelope-v2"):
        if adapter.config != policy.config or not 1 <= relevance_batch_size <= 20:
            raise ValueError("shared configuration and valid batch size required")
        if format_version not in {"legacy-v1", "json-envelope-v2"}:
            raise ValueError("unknown decision format version")
        self.format_version = format_version
        self.adapter, self.policy = adapter, policy
        self.source, self._gate = adapter.source, adapter._gate
        self.relevance_batch_size = relevance_batch_size

    @property
    def records(self):
        return self.adapter.records

    def decide(self, request):
        return self._send([validate_request(request)])[0]

    def format_instruction(self, instruction, requests):
        return instruction

    def observe_output(self, output, record):
        """Historical providers retain their existing bounded logging policy."""
        return None

    def parse_output(self, output, record):
        return parse_json(output)

    def request_model(self, requests):
        return (
            self.policy.model("decision"),
            self.policy.tier("decision").value,
            self.policy.snapshot()["hash"],
        )

    def request_schema(self, schema, requests):
        return schema

    def request_task(self, requests):
        return "decision"

    def decide_many(self, requests):
        requests = [validate_request(r) for r in requests]
        if any(r.task != "relevance" for r in requests):
            raise ValueError("only independent passage relevance may be batched")
        decisions = []
        for offset in range(0, len(requests), self.relevance_batch_size):
            decisions.extend(self._send(requests[offset : offset + self.relevance_batch_size]))
        return decisions

    def _send(self, requests):
        payload = self.payload_builder(requests)
        questions, answers_schema = payload["questions"], {}
        for index, request in enumerate(requests):
            key = f"q{index}"
            number = {"type": "number", "minimum": 0, "maximum": 1}
            answers_schema[key] = {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "choice", "confidence", "probabilities"],
                "properties": {
                    "type": {"type": "string", "const": "choice"},
                    "choice": {"type": "string", "enum": request.candidates},
                    "confidence": number,
                    "probabilities": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {c: number for c in request.candidates},
                        "required": request.candidates,
                    },
                },
            }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["answers"],
            "properties": {
                "answers": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": answers_schema,
                    "required": list(questions),
                }
            },
        }
        instruction = (
            "Return JSON answers to each typed question. Confidence/probabilities are predictions, not ground truth. Evaluate each request independently; do not output reasoning."
            if self.format_version == "legacy-v1"
            else "Return only a JSON object with exactly one root key, answers, containing an object keyed by the supplied question IDs (q0, q1, etc.). Each answer must contain type, choice, confidence, probabilities. No Markdown fences, prose, comments or reasoning. Include every candidate in probabilities; values must sum to 1. Confidence/probabilities are predictions, not ground truth. Evaluate each request independently."
        )
        instruction += " For each question, choice MUST be a candidate with the maximum value in that question's probabilities. If uncertain, report that uncertainty honestly; never inflate confidence to pass a threshold."
        instruction = self.format_instruction(instruction, requests)
        model, tier, policy_hash = self.request_model(requests)
        output = self.adapter.complete(
            model=model,
            task=self.request_task(requests),
            tier=tier,
            policy_hash=policy_hash,
            output_schema=self.request_schema(schema, requests),
            instructions=instruction,
            prompt=json.dumps(payload, ensure_ascii=False),
        )
        record = self.adapter.records[-1]
        self.observe_output(output, record)
        record.update(
            decision_task=requests[0].task,
            decision_count=len(requests),
            decision_format_version=self.format_version,
            payload_bytes=len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
        )
        try:
            data = self.parse_output(output, record)
            record["response_structure"] = {
                "root_type": type(data).__name__,
                "root_field_count": len(data) if isinstance(data, dict) else None,
                "answers_type": type(data.get("answers")).__name__
                if isinstance(data, dict)
                else None,
                "answer_field_count": len(data["answers"])
                if isinstance(data, dict) and isinstance(data.get("answers"), dict)
                else None,
            }
            if set(data) != {"answers"} or set(data["answers"]) != set(questions):
                raise ValueError("invalid question IDs")
            # Bounded predictions only, never provider prose, prompts or credentials.
            record["prediction_snapshot"] = [
                {
                    "question_id": f"q{i}",
                    "choice": answer.get("choice")
                    if answer.get("choice") in r.candidates
                    else None,
                    "confidence": answer.get("confidence")
                    if type(answer.get("confidence")) in {int, float}
                    and 0 <= answer["confidence"] <= 1
                    else None,
                    "probabilities": {
                        candidate: value
                        for candidate in r.candidates
                        if type(value := answer.get("probabilities", {}).get(candidate))
                        in {int, float}
                        and 0 <= value <= 1
                    }
                    if isinstance(answer.get("probabilities"), dict)
                    else {},
                }
                for i, r in enumerate(requests)
                if isinstance(answer := data["answers"][f"q{i}"], dict)
            ]
            decisions = [
                parse_answer(data["answers"][f"q{i}"], r.candidates, self.source)
                for i, r in enumerate(requests)
            ]
            record["decisions"] = [d.model_dump() for d in decisions]
            return decisions
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            record["status"] = "invalid_decision"
            record["error"] = "invalid_llm_decision"
            record["validation_error_type"] = type(error).__name__
            record["output_has_markdown_fence"] = output.strip().startswith("```")
            # Only fixed local validator messages may be logged; never raw output.
            reasons = {
                "invalid question IDs",
                "invalid Choice answer",
                "distribution candidate mismatch",
                "invalid probabilities",
                "distribution must sum to one",
                "choice must be a maximum-probability candidate",
                "confidence missing or invalid",
            }
            record["validation_reason"] = (
                str(error) if str(error) in reasons else "invalid_response_structure"
            )
            raise ProviderError("invalid_llm_decision") from None
