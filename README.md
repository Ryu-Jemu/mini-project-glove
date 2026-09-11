# 야구 첫걸음 챗봇

야구 입문자가 규칙과 기본 용어를 물어볼 수 있는 최소 프로젝트입니다. 현재는 외부 API 없이 예시 답변을 돌려주므로 설치할 패키지가 없습니다.

## 실행

Node.js 18 이상에서 아래 명령을 실행합니다.

```bash
npm run dev
```

브라우저에서 `http://localhost:3000`을 열면 됩니다.

## 구성

- `public/index.html`: 대화 화면
- `server.js`: 채팅 API (`POST /api/chat`)와 입문자용 예시 응답
- `.env.example`: 실제 LLM 연동용 환경 변수 자리

실제 모델을 연결할 때는 `server.js`의 `reply()`를 원하는 LLM API 호출로 교체하고, `.env`에 API 키를 설정하면 됩니다.
