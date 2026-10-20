"""Read-only experiment replay. Never load keys, run tools or generate unchecked answers."""

import json
import math
from pathlib import Path

from jev_rag.io import read_jsonl, sha256_file
from jev_rag.schema import source_location

ROOT = Path(__file__).resolve().parents[1]


def failure_diagnostic(row):
    """Explain known stages without exposing prompts, provider bodies or credentials."""
    status = row.get("status")
    messages = {
        "planning_failure": "检索规划未通过格式或内容校验；尚不能据此判断知识库缺少资料。",
        "generation_failure": "答案生成或修复未通过结构校验；不等于知识库没有相关内容。",
        "provider_failure": "模型服务或返回格式失败；不等于知识库没有答案。",
        "citation_failure": "已有候选证据，但答案的来源核验未通过。",
        "answer_incomplete": "答案未通过原问题完整性核验。",
        "low_confidence": "已有候选证据，但模型核验判断不够确定，未交付答案；不等于知识库没有相关内容。",
        "conflicting_evidence": "模型判断答案与引用证据存在未解决的矛盾，当前未交付；不代表已证实来源文档互相冲突。",
        "insufficient_evidence": "本次检索没有找到足够证据；不代表整个知识库没有相关内容。",
        "budget_exhausted": "本次请求额度已耗尽，未继续生成或核验。",
        "time_window_closed": "已进入高峰时段，暂停新请求。",
        "rewrite_failure": "补检索或修复规划未通过校验。",
    }
    if status not in messages:
        return None
    failures = [r for r in [*row.get("llm_calls", []), *row.get("jev_calls", [])]
                if r.get("status") == "failed"]
    stage = (failures[-1].get("failure_details") or {}).get("stage") if failures else None
    if stage == "transport_timeout":
        phase = {"generation_failure": "答案生成", "planning_failure": "检索规划",
                 "rewrite_failure": "补检索规划"}.get(status, "模型请求")
        message = f"{phase}请求超时，尚未得到可交付答案；未自动重试，不代表知识库缺少资料。"
        transport_phase = failures[-1].get("response_phase")
        if transport_phase == "reading_response_body":
            message += " 已收到服务响应头，但等待完整正文超时。"
        elif transport_phase == "waiting_for_headers":
            message += " 超时前尚未收到服务响应头，服务或网络原因仍待排查。"
    else:
        message = messages[status]
    malformed_decision = status == "provider_failure" and any(
        r.get("status") == "invalid_decision" for r in row.get("llm_calls", [])
    )
    if malformed_decision:
        message = "模型已返回核验结果，但其结构或字段未通过校验，未交付答案；不是连接失败或证据不足。"
    if status == "low_confidence":
        checks = [e for e in row.get("agent", {}).get("events", [])
                  if e.get("node") in {"citation", "original_query_completion"}]
        if checks:
            stage = "来源支持" if checks[-1]["node"] == "citation" else "原问题完整性"
            message = f"已有候选证据，但{stage}核验不够确定，未交付答案；不等于知识库没有相关内容。"
    if status == "generation_failure" and any(
        error.get("type") == "unit_count_exceeded"
        for record in row.get("llm_calls", [])
        for error in record.get("application_validation_errors", [])
    ):
        message = "答案生成或修复超过本轮回答单元限制，未进入最终核验。"
    if status == "generation_failure" and any(
        error.get("type") == "citation_not_retrieved"
        for record in row.get("llm_calls", [])
        for error in record.get("application_validation_errors", [])
    ):
        message = "已检索到资料，但生成答案使用了无效引用标识，未进入最终核验；不等于资料不足。"
    next_step = {
        "time_window_closed": "非高峰时段再继续提问。",
        "budget_exhausted": "结束当前授权后，可通过页面重新明确授权；不会自动追加额度。",
        "provider_failure": "请先检查服务连接；恢复后可继续提问。",
    }.get(status, "可以继续输入新的问题，失败记录会保留。")
    if malformed_decision:
        next_step = "可以继续提问；已记录核验格式错误供修复，无需将其当成网络故障或资料不足。"
    if malformed_decision:
        next_step = "可以继续提问；已保留具体格式错误供排查，不自动重试或放行答案。"
    elif stage == "transport_timeout":
        next_step = "请等待模型服务或网络恢复后再试；不会自动追加调用。"
    result = {"reason": status, "message": message, "next_step": next_step}
    if status in {"answer_incomplete", "low_confidence", "citation_failure", "conflicting_evidence"}:
        from frontend.answer_gaps import gap_diagnostic

        detail = gap_diagnostic(row)
        if detail:
            result.update(detail)
            if detail["uncertain_judgment"]:
                result["message"] = "核验倾向于支持候选回答，但确定性未达到交付条件；不能据此判断答案错误或资料缺失。"
            result["next_step"] = (
                "已针对缺口尝试补检索，仍未得到可交付结论；未展示未经完整核验的草稿。"
                if detail["supplement_attempted"] else
                "未确认的结论暂不展示；可补充具体使用场景，或查看引用词条进一步核对。"
            )
    return result


def provider_timing(row):
    def total(records):
        if not records:
            return None
        values = [r.get("latency_ms") for r in records]
        if any(not isinstance(v, (int, float)) or isinstance(v, bool)
               or not math.isfinite(v) or v < 0 for v in values):
            return None
        result = sum(values)
        return result if math.isfinite(result) else None
    llm = row.get("llm_calls", [])
    jev = row.get("jev_calls", [])
    verification = [r for r in llm if r.get("decision_task") == "citation"]
    return {"llm_ms": total(llm), "jev_ms": total(jev),
            "verification_ms": total(verification), "verification_requests": len(verification),
            "note": "物理请求累计耗时；并行请求的等待时间会重叠，累计值不等于处理耗时。核验是LLM耗时的子集，不能重复相加。未记录项显示未知。"}


def budget_routing(row):
    trace = next((event.get("answer_budget_policy") for event in row.get("agent", {}).get("events", [])
                  if event.get("node") == "answer_budget"), None)
    records = row.get("jev_calls", [])
    calls = len(records)
    budget_calls = sum(c.get("task") == "answer_budget" for c in records)
    events = row.get("agent", {}).get("events", [])
    light_checks = [e for e in events if e.get("node") == "light_complete_verification"]
    check_failed = any(e.get("error") for e in light_checks)
    fallback = check_failed or any(
        e.get("node") == "light_verification_fallback" for e in events)
    selected = trace.get("selected") if trace else "STANDARD"
    effective = "STANDARD" if fallback else selected
    message = (f"本题实际调用JEV {calls}次，其中回答预算判断{budget_calls}次。" if calls else "本题未调用JEV。")
    if check_failed:
        message += " LIGHT核验未通过，已转入STANDARD修复；不会跳过来源核验。"
    elif fallback:
        message += " 答案不满足LIGHT结构与引用条件，改用STANDARD来源及整题核验。"
    elif trace and trace.get("fallback") == "insufficient_budget_for_optional_classifier":
        message += " 剩余额度优先保留给生成及最终核验，跳过可选分类。"
    return {"jev_requests": calls, "answer_budget_requests": budget_calls,
            "selected": selected, "effective": effective, "light_fallback": fallback,
            "reason": trace.get("reason") if trace else "router_off",
            "message": message}


def reports(workspace=ROOT):
    base = (workspace / "results").resolve()
    return sorted(
        [
            p
            for p in (base / "live/ablation").glob("*/experiment.json")
            if p.resolve().is_relative_to(base) and p.is_file()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_report(path, workspace=ROOT):
    path = path.resolve()
    if not path.is_relative_to((workspace / "results").resolve()) or path.name != "experiment.json":
        raise ValueError("only local experiment replay artifacts can be opened")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("mode") != "LIVE" or data.get("type") != "LIVE_M6_DEVELOPMENT_ABLATION":
        raise ValueError("LIVE M6 receipt required; cannot pass MOCK as live")
    return data


def questions(report, workspace=ROOT):
    version = report["plan"]["dataset_version"]
    if Path(version).name != version or version in {".", ".."}:
        raise ValueError("invalid local dataset version")
    path = (workspace / "data/evaluation/drafts" / version / "questions.jsonl").resolve()
    if not path.is_relative_to((workspace / "data/evaluation/drafts").resolve()):
        raise ValueError("dataset escapes local review directory")
    if sha256_file(path) != report["plan"]["binding"]["dataset_sha256"]:
        raise ValueError("dataset is no longer the source of this replay")
    return {q["id"]: q["question"] for q in read_jsonl(path)}


def replay(report, question_map, query, variant, seed):
    # Exact matching only. An unseen question is not answered with an unrelated cached example.
    identifiers = {q for q, text in question_map.items() if text.strip() == query.strip()}
    rows = [
        r
        for r in report["records"]
        if r["case_id"] in identifiers and r["variant"] == variant and r["seed"] == seed
    ]
    if len(rows) > 1:
        raise ValueError("duplicate replay record")
    return rows[0] if rows else None


def presentation(row):
    agent = row.get("agent", {})
    papers = {}
    for chunk in agent.get("retrieved_documents", []):
        key = chunk.get("source_id") or chunk["paper_id"], chunk["section"], chunk["page"]
        papers[key] = {"paper": chunk["title"], "section": chunk["section"], "page": chunk["page"]}
        if chunk.get("locator"):
            papers[key].update(location=source_location(text=chunk.get("text", ""),
                source_kind=chunk.get("source_kind"), locator=chunk["locator"], page=chunk["page"]),
                kind=chunk.get("source_kind"))
    return {
        "status": row["status"],
        "verification_scope": row.get("verification_scope", "original_source_support"),
        "answer": row.get("answer"),
        "route": agent.get("route"),
        "intent": agent.get("intent"),
        "iterations": row.get("iterations"),
        "papers": list(papers.values()),
        "timeline": [
            {
                k: event.get(k)
                for k in (
                    "step_id",
                    "node",
                    "selected_action",
                    "probability",
                    "confidence",
                    "executed",
                    "error",
                )
            }
            for event in agent.get("events", [])
        ],
        "evidence": agent.get("relevant_documents", []),
        "citations": agent.get("final_citation_results", []),
        "failure_diagnostic": failure_diagnostic(row),
        "provider_timing": provider_timing(row),
        "budget_routing": budget_routing(row),
    }
