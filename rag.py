#!/usr/bin/env python
"""하위 호환 CLI 래퍼 — 실제 로직은 baseball.chain 에 있다.

사용: python rag.py [--show-context] [--model MODEL] "질문"
"""
from __future__ import annotations

import argparse
import sys

from baseball.chain import RagService


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python rag.py")
    ap.add_argument("question")
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--session", default=None)
    args = ap.parse_args(argv)

    service = RagService.create()
    result = service.answer(args.question, session_id=args.session, model=args.model)

    if args.show_context:
        print(result.context)
        print("--- context_end ---")
    print(result.answer)
    print(
        f"--- status={result.status} partial={result.partial_refusal} "
        f"freshness={result.freshness} sources={len(result.sources)} "
        f"citations={len(result.citations)} llm_called={result.llm_called}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
