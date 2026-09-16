"""Streamlit 프런트엔드 — 백엔드(FastAPI)의 SSE 를 턴마다 한 번씩 소비한다."""
from __future__ import annotations

import uuid

import streamlit as st

import clubs
import gate
import theme
from backend import MODE, document_url, readyz, reset_session, stream_answer

APP_TITLE = "KBO 야구 규칙 도우미"
PROMPT_SHA = "13a28529-718e8ea1"
EXAMPLES = [
    "인필드 플라이가 뭐야?", "보크가 뭐야?", "타점이 뭐야?",
    "도루가 뭐야?", "5.09 알려줘", "피치클락 몇 초야?",
]
STATUS_BADGE = {
    "answered": ("규칙집 근거 {n}건", "green", ":material/verified:"),
    "not_in_rulebook": ("자료에서 확인 불가", "gray", ":material/help:"),
    "out_of_scope": ("야구 외 질문", "gray", ":material/block:"),
    "phase2_pending": ("최신 정보 도구는 Phase 2", "orange", ":material/schedule:"),
}
FRESHNESS_LABEL = {"static": "규칙집", "snapshot": "리그 규정 스냅샷", "web": "웹 검색",
                   "live": "실시간", "model": "모델 지식(출처 없음)"}

st.set_page_config(
    page_title=APP_TITLE, page_icon=":material/sports_baseball:",
    layout="wide", initial_sidebar_state="expanded",
)
theme.inject_theme()

# 접근 코드가 설정되어 있으면 통과 전까지 아래를 아무것도 그리지 않는다.
if not gate.ensure_access():
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "pending" not in st.session_state:
    st.session_state.pending = None


def render_final(final: dict) -> None:
    """답변 밑에 붙는 상태 줄. 기록 다시 그리기와 방금 받은 답변이 같은 모양이어야 한다.

    두 자리에 같은 코드가 따로 있던 탓에 새 필드를 넣을 때마다 한쪽만 고쳐졌다.
    """
    label, color, icon = STATUS_BADGE.get(final["status"], ("처리됨", "gray", None))
    st.badge(label.format(n=len(final.get("sources", []))), color=color, icon=icon)

    notice = _grounding_notice(final)
    if notice:
        with st.container(key=f"partial-notice-{uuid.uuid4().hex[:6]}"):
            st.html(f'<div class="st-key-partial-notice">{notice}</div>')

    render_sources(final.get("sources", []))

    usage = final.get("usage", {})
    st.caption(
        f"{FRESHNESS_LABEL.get(final.get('freshness'), final.get('freshness'))} · "
        f"{usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)} tok · "
        f"${usage.get('cost_usd', 0):.4f} · {final.get('latency_ms', 0)}ms"
    )
    # 형식 검사가 어긋난 턴은 눈으로 구분되지 않으면 없는 것과 같다.
    issues = final.get("format_issues") or []
    if issues:
        st.caption(f"형식 점검: {', '.join(issues)}")


def _grounding_notice(final: dict) -> str | None:
    """근거 상태가 평소와 다르면 알린다.

    partial_refusal 은 답변 스키마를 켠 뒤로는 뜰 수 없다. chain 이 거부 문장이 섞인 답변을
    통째로 순수 거부로 강등하기 때문이다. 배너와 CSS 를 버리는 대신 여기로 용도를 옮긴다.
    """
    if final.get("freshness") == "model":
        return "⚠ 규칙집과 웹 검색에서 근거를 찾지 못해 일반적인 야구 상식으로 답했습니다."
    if final.get("partial_refusal"):
        return "⚠ 답변 일부에 근거가 없는 부분이 있습니다."
    if final.get("format_ok") is False:
        return "⚠ 답변 형식 검사를 통과하지 못했습니다. 내용이 평소와 다를 수 있습니다."
    return None


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"근거 자료 {len(sources)}건", expanded=False):
        with st.container(key=f"sources-panel-{uuid.uuid4().hex[:6]}"):
            chips = "".join(theme.chip(s.get("kind", "static"), s.get("label", "")) for s in sources)
            st.html(f'<div class="st-key-sources-panel">{chips}</div>')
            for s in sources:
                line = s.get("label", "")
                meta = " · ".join(x for x in [s.get("as_of"), s.get("confidence")] if x)
                st.markdown(f"**{line}**")
                if meta:
                    st.caption(meta)
                if s.get("url"):
                    st.link_button("출처 열기", s["url"])


with st.sidebar:
    st.markdown(f"**{APP_TITLE}**")
    ready = readyz()
    if ready:
        st.markdown(f"● 준비됨 · chunks {ready['document']['chunks']}")
        st.caption(f"{ready['chat_model']} · {ready['prompt_sha']} · web {ready['web_search']}")
    else:
        hint = ("백엔드(127.0.0.1:8000)를 실행하세요" if MODE == "api"
                else st.session_state.get("service_hint", "색인·데이터베이스 설정을 확인하세요"))
        st.markdown(f"○ 준비 안 됨 — {hint}")
        detail = st.session_state.get("service_error")
        if detail:
            with st.expander("오류 내용 보기"):
                st.code(detail[:600], language="text")
        st.caption(f"prompt {PROMPT_SHA}")

    url = document_url()
    if url:
        st.link_button("원문 PDF 열기", url, width="stretch")

    st.caption("예시 질문")
    for index, example in enumerate(EXAMPLES):
        if st.button(example, key=f"example-{index}", width="stretch"):
            st.session_state.pending = example
            st.rerun()

    st.caption("신선도: 규칙집(고정) · 스냅샷(30일) · 웹(6시간)")
    if st.button("세션 초기화", width="stretch"):
        reset_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    if gate.gate_enabled() and st.button("잠그기", width="stretch"):
        gate.lock()
        st.rerun()

    st.caption(
        "비공식 개인 학습용 프로젝트입니다. KBO 및 소속 10개 구단과 무관하며 "
        "구단 로고·워드마크·마스코트·유니폼 디자인·구단 서체를 사용하지 않습니다. "
        "구단 상징색은 각 구단 공식 웹사이트에서 확인한 색상값만 사용했습니다. "
        "순위·일정·선수 기록: 네이버 스포츠, 구단 기본 정보: 위키백과(CC BY-SA 4.0)."
    )

tab_chat, tab_clubs = st.tabs(["대화", "구단"])

with tab_clubs:
    clubs.render()

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=":material/sports_baseball:" if message["role"] == "assistant" else None):
        st.markdown(message["content"])
        if message.get("final"):
            final = message["final"]
            render_final(final)

typed = st.chat_input("야구 규칙을 물어보세요", submit_mode="disable")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/sports_baseball:"):
        captured: dict[str, dict] = {}
        with st.status("규칙집 검색 중…", expanded=False) as status:
            stream = stream_answer(question, st.session_state.session_id)

            def tokens():
                for event, data in stream:
                    if event == "route":
                        status.update(label=f"의도 분류: {data.get('kind')} ({data.get('by')})")
                    elif event == "status":
                        # 구조화 출력은 완성될 때까지 화면이 비므로 이 라벨이 유일한 피드백이다.
                        status.update(label="답변을 구성하는 중…"
                                      if data.get("status") == "generating"
                                      else "근거 수집 중…")
                    elif event == "sources":
                        status.update(label=f"근거 {len(data.get('sources', []))}건 확보 · 답변 생성 중…")
                    elif event == "token":
                        yield data.get("text", "")
                    elif event == "final":
                        captured["final"] = data
                    elif event == "error":
                        captured["error"] = data
                        yield f"\n\n{data.get('detail', '알 수 없는 오류')}"

            answer = st.write_stream(tokens())
            status.update(label="완료", state="complete")

        error = captured.get("error")
        if error:
            if error.get("kind") == "quota":
                st.error(
                    "OpenAI 사용 한도를 초과했습니다. platform.openai.com → 해당 프로젝트 → "
                    "Limits 에서 한도를 올리거나 결제 수단을 확인한 뒤 다시 시도해 주세요.",
                    icon=":material/credit_card_off:",
                )
            else:
                st.error(error.get("detail", "요청을 처리하지 못했습니다."), icon=":material/error:")

        final_payload = captured.get("final", {})
        if final_payload:
            render_final(final_payload)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer or final_payload.get("answer", ""), "final": final_payload}
    )
