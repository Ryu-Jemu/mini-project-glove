# KBO 야구 규칙 도우미

야구 규칙을 질문하면 규칙집을 근거로 답해 주는 웹 애플리케이션이다.
아래 순서를 위에서부터 그대로 따르면 로컬에서 실행할 수 있다. 최초 1회 소요 시간은 5분에서 10분이다.

이 문서는 **macOS 기준**이다. Linux 도 같은 순서로 동작한다.
Windows 에서는 절차를 그대로 쓸 수 없으므로 WSL2 환경에서 진행한다.

---

## 1. 준비물

다음 네 가지를 먼저 갖춘다.

1. **Xcode Command Line Tools** (`git` 과 `make` 를 제공한다)
   - 확인 명령은 `git --version` 과 `make --version` 이다.
   - 설치 창이 뜨거나 오류가 나면 다음을 실행하고, 설치가 끝난 뒤 다시 확인한다.
     ```bash
     xcode-select --install
     ```
2. **Docker Desktop**
   - 설치한 뒤 반드시 실행해 둔다.
   - 확인 명령은 `docker compose version` 이다. Compose v2 이상이어야 한다.
3. **uv** (파이썬 패키지 관리자)
   - 확인 명령은 `uv --version` 이다.
   - 설치되어 있지 않으면 다음을 실행한다.
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     source $HOME/.local/bin/env
     ```
   - 설치 후 `uv --version` 이 버전을 출력하는지 반드시 다시 확인한다.
     `command not found` 가 나오면 터미널을 새로 열고 한 번 더 확인한다.
4. **OpenAI API 키**
   - platform.openai.com 의 API keys 메뉴에서 발급한다.
   - 팀원마다 각자 발급하여 각자 사용한다.

참고 사항은 다음과 같다.

- 파이썬은 따로 설치하지 않는다. uv 가 3.12 버전을 자동으로 내려받는다.
- 디스크 여유 공간이 약 1.5 GB 필요하다.

---

## 2. 설치

터미널에서 다음을 순서대로 실행한다.

1. 저장소를 내려받고 해당 디렉터리로 이동한다.
   ```bash
   git clone https://github.com/Ryu-Jemu/mini-project-glove.git
   cd mini-project-glove
   ```
2. 환경 파일을 복사한다.
   ```bash
   cp .env.example .env
   ```
3. `.env` 파일을 편집기로 열어 **`OPENAI_API_KEY` 에만** 발급받은 키를 붙여 넣는다.
   ```
   OPENAI_API_KEY=sk-...
   ```
   - 값에 따옴표를 붙이지 않는다. 앞뒤 공백이 들어가지 않게 한다.
   - 나머지 항목은 수정하지 않는다. 기본값이 로컬 환경을 가리킨다.
   - `TAVILY_API_KEY` 와 `LANGCHAIN_API_KEY` 는 비워 두어도 무방하다. 해당 기능만 꺼진다.
4. 설치 명령을 실행한다.
   ```bash
   make setup
   ```
5. 마지막 줄에 `준비 완료` 가 출력되면 성공이다.

참고 사항은 다음과 같다.

- `make setup` 은 여러 번 실행하여도 안전하다.
- 자료 준비 비용은 최초 1회 약 $0.016 이다. 같은 규칙집으로 다시 실행하면 건너뛰므로 추가 비용이 없다.
- **설치 이후에는 질문 한 번마다 비용이 따로 발생한다.** 질문 1건당 약 $0.0005 이며, 화면 하단에 실제 금액이 표시된다.

---

## 3. 실행

터미널 두 개를 사용한다.

1. 컨테이너를 기동한다. `make setup` 직후라면 이미 떠 있으므로 건너뛴다.
   ```bash
   make up
   ```
2. 첫 번째 터미널에서 백엔드를 실행한다.
   ```bash
   cd mini-project-glove
   make api
   ```
3. 두 번째 터미널을 새로 열고 같은 디렉터리로 이동한 뒤 화면을 실행한다.
   ```bash
   cd mini-project-glove
   make ui
   ```
4. 브라우저에서 다음 주소를 연다.
   ```
   http://127.0.0.1:8501
   ```
5. 왼쪽 사이드바에 `● 준비됨` 이 표시되면 정상이다.

종료와 재실행 방법은 다음과 같다.

- 종료할 때는 각 터미널에서 `Ctrl + C` 를 누른다.
- 컨테이너까지 내리려면 `make down` 을 실행한다. 데이터는 보존된다.
- **다시 시작할 때는 `make up` 을 먼저 실행한다.** `make down` 을 했거나 컴퓨터를 재부팅한 뒤에는 반드시 필요하다.

---

## 4. 동작 확인

다음 두 가지 방법으로 확인한다.

1. 화면에서 확인한다.
   - 사이드바의 예시 질문 버튼을 누른다.
   - 답변과 함께 하단에 근거 배지가 표시된다.
2. 명령으로 확인한다.
   ```bash
   make test
   make doctor
   ```
   - `make test` 는 `95 passed` 가 출력되면 정상이다. API 키가 없어도 실행된다.
   - `make doctor` 는 10줄을 출력한다. `OPENAI`·`PG`·`MinIO`·`Kiwi`·`index`·`KBO teams`·`prompts` 일곱 줄이 `ok` 이면 정상이다.
   - `Tavily`·`LangSmith`·`YouTube` 는 선택 항목이다. 키를 넣지 않았다면 `absent` 로 표시되며 문제가 아니다.
   - `index` 줄에 `ok` 가 없으면 `make setup` 이 끝까지 진행되지 않은 것이다. `make setup` 을 다시 실행한다.

---

## 5. 명령어 목록

| 명령 | 용도 |
|---|---|
| `make setup` | 최초 1회 전체 설치 |
| `make up` | 컨테이너 기동 |
| `make api` | 백엔드 실행 |
| `make ui` | 화면 실행 |
| `make test` | 테스트 실행 (비용 없음) |
| `make test-net` | 외부 공개 API 형태 점검 (비용 없음, 네트워크 필요) |
| `make doctor` | 환경 점검 (비용 없음) |
| `make down` | 컨테이너 정지 |
| `make ingest` | 규칙집 자료 준비 (이미 준비되어 있으면 건너뛰며 비용 없음) |
| `make test-live` | 실제 API 호출 테스트 (비용 발생) |
| `make eval` | 품질 평가 (비용 발생) |

- 명령줄에서 직접 질문하려면 다음을 실행한다. 질문 1건의 비용이 발생한다.
  ```bash
  ./bin/br python rag.py "인필드 플라이가 뭐야?"
  ```

---

## 6. 문제 해결

증상별로 다음을 확인한다.

1. **`ModuleNotFoundError: No module named 'baseball'`**
   - `uv run` 을 직접 사용하지 말고 `make` 명령 또는 `./bin/br` 를 사용한다.
   - 그래도 발생하면 `make fix-pth` 를 실행한다.
2. **사이드바에 `준비 안 됨` 이 표시된다**
   - 첫 번째 터미널에서 `make api` 가 실행 중인지 확인한다.
   - 다음 명령의 응답에 `"status":"ready"` 가 있으면 백엔드는 정상이다.
     ```bash
     curl http://127.0.0.1:8000/readyz
     ```
   - `{"detail"` 로 시작하는 응답이 돌아오면 컨테이너나 자료가 준비되지 않은 것이다. `make up` 을 실행한 뒤 `make api` 를 다시 시작한다.
3. **`address already in use` 가 출력된다**
   - 이전 프로세스가 남아 있다. 다음으로 확인한 뒤 종료한다.
     ```bash
     lsof -iTCP:8000 -sTCP:LISTEN
     lsof -iTCP:8501 -sTCP:LISTEN
     ```
   - `make setup` 중에 `port is already allocated` 가 나오면 컨테이너 포트가 겹친 것이다. 이미 실행 중인 로컬 Postgres 나 MinIO 를 종료하고 다시 실행한다.
4. **질문하여도 답변이 나오지 않는다**
   - 사이드바가 `준비 안 됨` 이면 2번 항목을 먼저 확인한다.
   - `죄송하지만` 으로 시작하는 문장이 나오면 오류가 아니라 정상적인 거부 응답이다.
   - `OpenAI 사용 한도를 초과` 문구가 뜨면 platform.openai.com 의 해당 프로젝트 Limits 에서 한도를 상향한다.
   - 그 밖의 경우에는 `.env` 의 `OPENAI_API_KEY` 값에 오타나 따옴표가 없는지 확인한다.
5. **`Cannot connect to the Docker daemon` 이 출력된다**
   - Docker Desktop 이 실행 중인지 확인한다.
6. **컨테이너가 준비되지 않는다**
   - 다음으로 원인을 확인한다.
     ```bash
     docker compose logs --tail 50
     ```

---

## 7. 준수 사항

- `.env` 파일은 커밋하지 않는다.
- API 키를 채팅, 이슈, 커밋에 붙여 넣지 않는다.
- `prompts/` 디렉터리의 세 파일은 수정하지 않는다.
  `system_prompt.txt` 와 `human_prompt.txt` 는 수정하면 테스트가 실패하고 앱이 기동하지 않는다.
  `router_v1.md` 는 자동 검증이 없으므로 더욱 주의한다.
- 모든 포트는 외부에 공개하지 않는다.

---

## 8. 추가 문서

- [docs/팀_시작하기.md](docs/팀_시작하기.md) — 설치 절차의 상세 설명과 오류 대처
- [docs/배포.md](docs/배포.md) — 외부 배포 가능 범위와 절차
- [docs/설계서v1.md](docs/설계서v1.md) — 설계 내용과 측정 수치

---

## 9. 고지

비공식 개인 학습용 프로젝트입니다. LG 트윈스·(주)LG스포츠·KBO와 무관하며 구단 로고·워드마크·마스코트·유니폼 디자인을 사용하지 않습니다. 순위·일정 데이터: 네이버 스포츠, 영상: YouTube.
