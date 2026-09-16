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

GATES = {
    "recall_at_6": 0.8,
    "status_match": 13,
    "route_match": 15,
    "refusal_exact": 2,
    "sources_nonempty_ratio": 1.0,
}
EXACT_MAP = {"OFF_TOPIC_REFUSAL": OFF_TOPIC_REFUSAL, "NOT_IN_CONTEXT_REFUSAL": NOT_IN_CONTEXT_REFUSAL}


@dataclass
class CaseResult:
    id: int
    question: str
    route: str = ""
    route_ok: bool = False
    recall: float = 1.0
    top1_ok: bool | None = None
    status: str = ""
    status_ok: bool | None = None
    must_contain_ok: bool | None = None
    refusal_exact_ok: bool | None = None
    sources: int = 0
    freshness: str = ""
    llm_called: bool | None = None
    retrieved: list[str] = field(default_factory=list)


def _covered(expected: str, retrieved: list[str]) -> bool:
    return any(r == expected or r.startswith(expected) for r in retrieved)


def evaluate(with_llm: bool, baseline: str) -> dict[str, Any]:
    settings = get_settings()
    if baseline == "dense":
        settings = settings.model_copy(update={"rrf_weight_bm25": 0.0})
    elif baseline == "bm25":
        settings = settings.model_copy(update={"rrf_weight_dense": 0.0})

    retriever = build_retriever(settings)
    service = RagService(settings, retriever) if with_llm else None
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
        results.append(r)

    scored = [r for r in results if r.retrieved or r.recall < 1.0]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_sha": PROMPT_SHA,
        "baseline": baseline,
        "with_llm": with_llm,
        "index_version": retriever.document_id,
        "cases": len(results),
        "route_match": sum(1 for r in results if r.route_ok),
        "recall_at_6": round(sum(r.recall for r in scored) / max(len(scored), 1), 3),
        "top1_ok": all(r.top1_ok for r in results if r.top1_ok is not None),
    }
    if with_llm:
        answered = [r for r in results if r.status == "answered"]
        summary.update({
            "status_match": sum(1 for r in results if r.status_ok),
            "must_contain_match": sum(1 for r in results if r.must_contain_ok),
            "must_contain_total": sum(1 for r in results if r.must_contain_ok is not None),
            "refusal_exact": sum(1 for r in results if r.refusal_exact_ok),
            "sources_nonempty_ratio": round(
                sum(1 for r in answered if r.sources > 0) / max(len(answered), 1), 3),
        })
    return {"summary": summary, "cases": [vars(r) for r in results]}


def check_gates(summary: dict[str, Any]) -> list[str]:
    failures = []
    if summary["recall_at_6"] < GATES["recall_at_6"]:
        failures.append(f"recall@6 {summary['recall_at_6']} < {GATES['recall_at_6']}")
    if summary["route_match"] < GATES["route_match"]:
        failures.append(f"route {summary['route_match']}/15 < {GATES['route_match']}")
    if not summary["top1_ok"]:
        failures.append("규칙번호 질의 top-1 불일치")
    if summary.get("with_llm"):
        if summary["status_match"] < GATES["status_match"]:
            failures.append(f"status {summary['status_match']}/15 < {GATES['status_match']}")
        if summary["refusal_exact"] < GATES["refusal_exact"]:
            failures.append(f"refusal_exact {summary['refusal_exact']}/2")
        if summary["sources_nonempty_ratio"] < GATES["sources_nonempty_ratio"]:
            failures.append(f"sources_nonempty {summary['sources_nonempty_ratio']}")
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python evaluation/run_eval.py")
    ap.add_argument("--baseline", choices=["rrf", "dense", "bm25"], default="rrf")
    ap.add_argument("--all-baselines", action="store_true")
    ap.add_argument("--with-llm", action="store_true", help="실제 답변까지 평가(비용 발생)")
    args = ap.parse_args(argv)

    baselines = ["rrf", "dense", "bm25"] if args.all_baselines else [args.baseline]
    exit_code = 0
    out_dir = ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for baseline in baselines:
        report = evaluate(args.with_llm, baseline)
        s = report["summary"]
        path = out_dir / f"{PROMPT_SHA}_{baseline}{'_llm' if args.with_llm else ''}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        line = f"[{baseline}] recall@6={s['recall_at_6']} route={s['route_match']}/15 top1={s['top1_ok']}"
        if args.with_llm:
            line += (f" status={s['status_match']}/15 must_contain={s['must_contain_match']}"
                     f"/{s['must_contain_total']} refusal_exact={s['refusal_exact']}/2"
                     f" sources_nonempty={s['sources_nonempty_ratio']}")
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
