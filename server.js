import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = dirname(fileURLToPath(import.meta.url));
const port = process.env.PORT || 3000;

const answers = [
  {
    keywords: ["야구", "규칙", "시작", "처음"],
    text: "야구는 공격팀이 공을 쳐서 베이스를 돌아 홈으로 들어오면 점수를 얻는 경기예요. 9회까지 더 많은 점수를 낸 팀이 이깁니다. 궁금한 단어를 하나씩 물어보세요!"
  },
  {
    keywords: ["스트라이크", "볼", "삼진"],
    text: "타자가 친 공이 스트라이크존을 통과하면 스트라이크, 벗어나면 볼이에요. 스트라이크 3개면 삼진 아웃, 볼 4개면 1루로 출루합니다."
  },
  {
    keywords: ["아웃", "안타", "홈런"],
    text: "안타는 친 공이 수비에 잡히기 전에 페어 지역에 떨어져 출루한 것이고, 홈런은 공을 친 뒤 모든 베이스를 돌아 홈까지 들어온 타격이에요. 공격팀은 아웃 3개가 되면 교대합니다."
  }
];

function reply(message = "") {
  const normalized = message.toLowerCase();
  return answers.find(({ keywords }) => keywords.some((word) => normalized.includes(word)))?.text
    ?? "좋은 질문이에요. ‘스트라이크와 볼’, ‘안타와 홈런’, ‘포지션’처럼 궁금한 야구 용어를 물어보면 쉽게 설명해 드릴게요.";
}

createServer(async (req, res) => {
  if (req.method === "POST" && req.url === "/api/chat") {
    let body = "";
    for await (const chunk of req) body += chunk;
    try {
      const { message } = JSON.parse(body || "{}");
      res.writeHead(200, { "Content-Type": "application/json; charset=utf-8" });
      return res.end(JSON.stringify({ reply: reply(message) }));
    } catch {
      res.writeHead(400, { "Content-Type": "application/json; charset=utf-8" });
      return res.end(JSON.stringify({ error: "올바른 요청 형식이 아닙니다." }));
    }
  }

  if (req.method === "GET" && (req.url === "/" || req.url === "/index.html")) {
    const page = await readFile(join(root, "public", "index.html"));
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
    return res.end(page);
  }

  res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
  res.end("Not Found");
}).listen(port, () => console.log(`Baseball chatbot: http://localhost:${port}`));
