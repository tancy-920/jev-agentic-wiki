"""Opt-in local literal veto plus unchanged model support and shared repair budget."""

from .anchored import AnchoredAgent, validate_binding
from .arithmetic import PROTOCOL, mismatches
from .contracts import Draft


class ArithmeticAgent(AnchoredAgent):
    protocol_version = "anchored-arithmetic-v1"
    guard_protocol = PROTOCOL
    guard_node = "literal_operation_guard"
    guard_instruction = "Fallible bounded extraction found a literal operation mismatch. Re-read full cited source and repair the complete answer. Do not treat extracted metadata as a universal mathematical proof. Preserve all user requirements. All units will be rechecked."

    def _guard_issues(self, draft, evidence):
        return mismatches(draft, evidence)

    def verify(self, state):
        draft = Draft.model_validate(state["draft_answer"])
        try:
            validate_binding(
                getattr(self.generator, "last_binding", None),
                state["query"],
                draft,
                self._evidence(state),
            )
        except ValueError:
            return super().verify(state)
        issues = self._guard_issues(draft, self._evidence(state))
        self._event(state, self.guard_node, [], executed=False)
        state["events"][-1].update(
            protocol=self.guard_protocol, issues=issues, semantic_support_certified=False
        )
        if not issues:
            return super().verify(state)
        state["citation_results"] += [
            dict(
                claim=draft.claims[i["claim_index"]].text,
                status="APPLICATION_LITERAL_MISMATCH",
                source="application",
            )
            for i in issues
        ]
        if state["citation_repair_count"] < 1 and callable(
            getattr(self.generator, "generate_repair", None)
        ):
            state["citation_repair_count"] += 1
            state["retry_count"] += 1
            state["loop_count"] += 1
            state["verification_feedback"] = [
                dict(
                    claim=draft.claims[i["claim_index"]].text,
                    status="APPLICATION_LITERAL_MISMATCH",
                    literal_operations=i,
                    instruction=self.guard_instruction,
                )
                for i in issues
            ]
            state["termination_reason"], state["next_node"] = None, "generate"
            self._event(state, "citation_repair", ["GENERATE"], executed=True, action="GENERATE")
        else:
            state["termination_reason"], state["next_node"] = "citation_failure", "finish"
        return state
