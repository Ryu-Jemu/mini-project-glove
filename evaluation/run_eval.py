#!/usr/bin/env python
"""골든셋 평가. 검색 지표는 비용 0, --with-llm 을 붙이면 실제 답변까지 평가한다."""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from baseball.chain import RagService  # noqa: E402
from baseball.config import get_settings  # noqa: E402
from baseball.prompts import NOT_IN_CONTEXT_REFUSAL, OFF_TOPIC_REFUSAL, PROMPT_SHA  # noqa: E402
from baseball.retriever import build_retriever  # noqa: E402
from baseball.router import route as route_question  # noqa: E402

sys.path.insert(0, str(ROOT / "evaluation"))
import judge as judge_mod  # noqa: E402

# 절대 개수가 아니라 허용 오차로 적는다. 골든셋에 케이스를 더해도 게이트가 깨지지 않는다.
GATES = {
    "recall_at_6": 0.8,
    "route_mismatch_allowed": 0,
    "status_mismatch_allowed": 2,
    "answer_kind_mismatch_allowed": 1,
    "sources_nonempty_ratio": 1.0,
    "judge_grounded_avg": 4.0,
    "judge_grounded_min": 3,
    "judge_schema_format_avg": 4.0,
}
EXACT_MAP = {"OFF_TOPIC_REFUSAL": OFF_TOPIC_REFUSAL, "NOT_IN_CONTEXT_REFUSAL": NOT_IN_CONTEXT_REFUSAL}


@dataclass
class CaseResult:
    id: int
    question: str
    route: str = ""
    route_ok: bool = False
    recall: float = 1.0
    recall_scored: bool = False
    top1_ok: bool | None = None
    status: str = ""
    status_ok: bool | None = None
    must_contain_ok: bool | None = None
    refusal_exact_ok: bool | None = None
    sources: int = 0
    freshness: str = ""
    llm_called: bool | None = None
    retrieved: list[str] = field(default_factory=list)
    answer_kind: str | None = None
    answer_kind_ok: bool | None = None
    format_ok: bool | None = None
    lint_codes: list[str] = field(default_factory=list)
    judge: dict[str, Any] | None = None


def _covered(expected: str, retrieved: list[str]) -> bool:
    return any(r == expected or r.startswith(expected) for r in retrieved)


def evaluate(with_llm: bool, baseline: str, *, judge_on: bool = False,
             judge_model: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    if baseline == "dense":
        settings = settings.model_copy(update={"rrf_weight_bm25": 0.0})
    elif baseline == "bm25":
        settings = settings.model_copy(update={"rrf_weight_dense": 0.0})

    retriever = build_retriever(settings)
    service = RagService(settings, retriever) if with_llm else None
    judge_runner = judge_mod.build_judge(settings, judge_model) if (with_llm and judge_on) else None
    cases = yaml.safe_load((ROOT / "evaluation" / "golden_v1.yaml").read_text(encoding="utf-8"))["cases"]

    results: list[CaseResult] = []
    for case in cases:
        r = CaseResult(id=case["id"], question=case["question"])
        routed = route_question(case["question"], settings=settings)
        r.route = routed.kind
        r.route_ok = routed.kind == case["expect_route"]

        if routed.kind != "off_topic":
            res = retriever.retrieve(case["question"], k=6)
            r.retrieved = [d["rule_id"] for d in res.docs]
            expected = case.get("expected_rule_refs") or []
            if expected:
                hits = sum(1 for e in expected if _covered(e, r.retrieved))
                r.recall = hits / len(expected)
                r.recall_scored = True
            if case.get("expect_top1"):
                r.top1_ok = bool(r.retrieved) and _covered(case["expect_top1"], r.retrieved[:1])

        if with_llm and service is not None:
            turn = service.answer(case["question"])
            r.status, r.sources, r.freshness = turn.status, len(turn.sources), turn.freshness
            r.llm_called = turn.llm_called
            expect_status = case.get("expect_status")
            allowed = case.get("expect_status_in") or ([expect_status] if expect_status else [])
            r.status_ok = turn.status in allowed if allowed else None
            if case.get("must_contain"):
                r.must_contain_ok = all(
                    re.search(pat, turn.answer) for pat in case["must_contain"]
                )
            if case.get("expect_answer_exact"):
                r.refusal_exact_ok = turn.answer == EXACT_MAP[case["expect_answer_exact"]]
            if case.get("expect_freshness_in"):
                r.status_ok = bool(r.status_ok) and turn.freshness in case["expect_freshness_in"]
            r.answer_kind, r.format_ok = turn.answer_kind, turn.format_ok
            r.lint_codes = list(turn.format_issues)
            if case.get("expect_answer_kind") and turn.llm_called:
                r.answer_kind_ok = turn.answer_kind == case["expect_answer_kind"]
            if judge_runner is not None and turn.status == "answered":
                r.judge = judge_mod.judge(
                    judge_runner, question=case["question"],
                    context=turn.context, answer=turn.answer,
                )
        results.append(r)

    # expected_rule_refs 가 빈 케이스(최신정보·구단정보·오프토픽)는 분모에서 뺀다.
    # 예전에는 이들이 recall=1.0 으로 들어가 평균을 부풀렸다.
    scored = [r for r in results if r.recall_scored]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_sha": PROMPT_SHA,
        "baseline": baseline,
        "with_llm": with_llm,
        "index_version": retriever.document_id,
        "cases": len(results),
        "route_match": sum(1 for r in results if r.route_ok),
        "recall_at_6": round(sum(r.recall for r in scored) / max(len(scored), 1), 3),
        "recall_scored_cases": len(scored),
        "top1_ok": all(r.top1_ok for r in results if r.top1_ok is not None),
    }
    if with_llm:
        answered = [r for r in results if r.status == "answered"]
        kind_cases = [r for r in results if r.answer_kind_ok is not None]
        summary.update({
            "schema_enabled": settings.answer_schema_enabled,
            "status_match": sum(1 for r in results if r.status_ok),
            "status_total": sum(1 for r in results if r.status_ok is not None),
            "must_contain_match": sum(1 for r in results if r.must_contain_ok),
            "must_contain_total": sum(1 for r in results if r.must_contain_ok is not None),
            "refusal_exact": sum(1 for r in results if r.refusal_exact_ok),
            "refusal_exact_total": sum(1 for r in results if r.refusal_exact_ok is not None),
            "sources_nonempty_ratio": round(
                sum(1 for r in answered if r.sources > 0) / max(len(answered), 1), 3),
            "answer_kind_match": sum(1 for r in kind_cases if r.answer_kind_ok),
            "answer_kind_total": len(kind_cases),
            "schema_fallback": sum(1 for r in results if "SCHEMA_FALLBACK" in r.lint_codes),
            "lint_warn": sorted({c for r in results for c in r.lint_codes}),
        })
        if judge_on:
            summary.update(judge_mod.averages([r.judge for r in results if r.judge is not None]))
    return {"summary": summary, "cases": [vars(r) for r in results]}


def check_gates(summary: dict[str, Any]) -> list[str]:
    failures = []
    total = summary["cases"]
    if summary["recall_at_6"] < GATES["recall_at_6"]:
        failures.append(f"recall@6 {summary['recall_at_6']} < {GATES['recall_at_6']}")
    if total - summary["route_match"] > GATES["route_mismatch_allowed"]:
        failures.append(f"route {summary['route_match']}/{total}")
    if not summary["top1_ok"]:
        failures.append("규칙번호 질의 top-1 불일치")
    if not summary.get("with_llm"):
        return failures

    if summary["status_total"] - summary["status_match"] > GATES["status_mismatch_allowed"]:
        failures.append(f"status {summary['status_match']}/{summary['status_total']}")
    if summary["refusal_exact"] < summary["refusal_exact_total"]:
        failures.append(f"refusal_exact {summary['refusal_exact']}/{summary['refusal_exact_total']}")
    if summary["sources_nonempty_ratio"] < GATES["sources_nonempty_ratio"]:
        failures.append(f"sources_nonempty {summary['sources_nonempty_ratio']}")
    # 포맷 회귀의 최저비용 탐지기 — 여태 출력만 하고 게이트가 아니었다
    if summary["must_contain_match"] < summary["must_contain_total"]:
        failures.append(f"must_contain {summary['must_contain_match']}/{summary['must_contain_total']}")

    if summary.get("schema_enabled"):
        if summary["schema_fallback"]:
            failures.append(f"schema_fallback {summary['schema_fallback']}건 — 스키마 파싱 실패")
        blocking = [c for c in summary["lint_warn"]
                    if c in {"REFUSAL_INSIDE_ANSWER", "EMPTY_HEADLINE"}]
        if blocking:
            failures.append(f"lint blocking: {', '.join(blocking)}")
        gap = summary["answer_kind_total"] - summary["answer_kind_match"]
        if gap > GATES["answer_kind_mismatch_allowed"]:
            failures.append(f"answer_kind {summary['answer_kind_match']}/{summary['answer_kind_total']}")

    for key in ("judge_grounded_avg", "judge_schema_format_avg"):
        value = summary.get(key)
        if value is not None and value < GATES[key]:
            failures.append(f"{key} {value} < {GATES[key]}")
    low = summary.get("judge_grounded_min")
    if low is not None and low < GATES["judge_grounded_min"]:
        failures.append(f"judge_grounded 최저 {low} < {GATES['judge_grounded_min']}")
    # 가독성·간결성은 주관적이고 심판 모델에 의존한다. 리포트만 하고 게이트로 두지 않는다.
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python evaluation/run_eval.py")
    ap.add_argument("--baseline", choices=["rrf", "dense", "bm25"], default="rrf")
    ap.add_argument("--all-baselines", action="store_true")
    ap.add_argument("--with-llm", action="store_true", help="실제 답변까지 평가(비용 발생)")
    ap.add_argument("--judge", action="store_true", help="LLM 심판 채점 추가(비용 발생)")
    ap.add_argument("--judge-model", default=None, help="심판 모델(기본: OPENAI_CHAT_MODEL)")
    args = ap.parse_args(argv)
    if args.judge and not args.with_llm:
        ap.error("--judge 는 --with-llm 과 함께 써야 한다")

    baselines = ["rrf", "dense", "bm25"] if args.all_baselines else [args.baseline]
    exit_code = 0
    out_dir = ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for baseline in baselines:
        report = evaluate(args.with_llm, baseline, judge_on=args.judge,
                          judge_model=args.judge_model)
        s = report["summary"]
        # PROMPT_SHA 는 이번 작업으로 바뀌지 않는다. 접미사가 없으면 A/B 두 판이 서로를 덮어쓴다.
        variant = ""
        if args.with_llm:
            variant = "_schema" if s.get("schema_enabled") else "_legacy"
        path = out_dir / f"{PROMPT_SHA}_{baseline}{'_llm' if args.with_llm else ''}{variant}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        total = s["cases"]
        line = (f"[{baseline}] recall@6={s['recall_at_6']} route={s['route_match']}/{total}"
                f" top1={s['top1_ok']}")
        if args.with_llm:
            line += (f" status={s['status_match']}/{s['status_total']}"
                     f" must_contain={s['must_contain_match']}/{s['must_contain_total']}"
                     f" refusal_exact={s['refusal_exact']}/{s['refusal_exact_total']}"
                     f" sources_nonempty={s['sources_nonempty_ratio']}")
            if s.get("schema_enabled"):
                line += (f" kind={s['answer_kind_match']}/{s['answer_kind_total']}"
                         f" fallback={s['schema_fallback']}")
                if s["lint_warn"]:
                    line += f" lint={','.join(s['lint_warn'])}"
        if args.judge:
            line += (f" judge(g/f/r/c)={s.get('judge_grounded_avg')}/"
                     f"{s.get('judge_schema_format_avg')}/{s.get('judge_readability_avg')}/"
                     f"{s.get('judge_conciseness_avg')}"
                     f" scored={s.get('judge_scored')} failed={s.get('judge_failed')}")
        print(line)
        failures = check_gates(s)
        if failures:
            exit_code = 1
            for f in failures:
                print(f"  GATE FAIL: {f}")
        print(f"  → {path.relative_to(ROOT)}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
