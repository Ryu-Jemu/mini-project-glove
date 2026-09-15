# 경로에 공백이 있으므로 상대 경로만 사용한다.
SHELL := /bin/bash
PY := ./bin/br python

.PHONY: fix-pth sync up down ingest api ui test test-live eval doctor setup
fix-pth:            ## macOS UF_HIDDEN 으로 .pth 가 무시되는 문제 해제
	@chflags nohidden .venv/lib/python3.12/site-packages/*.pth 2>/dev/null || true
sync: ; uv sync && $(MAKE) fix-pth
setup:              ## 팀원용 1회 셋업: 의존성 → 컨테이너 → 스키마 → PDF 업로드 → 색인
	@test -f .env || { echo '먼저 .env 를 만드세요:  cp .env.example .env  (OPENAI_API_KEY 필수)'; exit 1; }
	$(MAKE) sync
	$(MAKE) up
	@echo '컨테이너 health 대기...'
	@ok=0; for i in $$(seq 1 60); do [ "$$(docker compose ps --format '{{.Health}}' | grep -cx healthy)" = '2' ] && { ok=1; break; } || sleep 2; done; \
	  [ $$ok = 1 ] || { echo '컨테이너가 준비되지 않았습니다. docker compose logs --tail 50 으로 확인하세요.'; exit 1; }
	$(PY) -m baseball.db ensure
	$(PY) -m baseball.storage sync
	$(PY) -m baseball.ingest
	$(PY) -m baseball.doctor
	@echo ''
	@echo '준비 완료. 터미널 두 개에서:  make api   /   make ui'
up: ; docker compose up -d
down: ; docker compose down
doctor: fix-pth ; $(PY) -m baseball.doctor
ingest: fix-pth ; $(PY) -m baseball.ingest
api: fix-pth ; ./bin/br uvicorn baseball.api:app --host 127.0.0.1 --port 8000
ui: fix-pth ; API_BASE_URL=http://127.0.0.1:8000 ./bin/br streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501
test: fix-pth ; ./bin/br pytest -q -m "not live and not db and not llm_judge and not youtube_api"
test-live: fix-pth ; ./bin/br pytest -q -m live
eval: fix-pth ; $(PY) evaluation/run_eval.py --all-baselines
