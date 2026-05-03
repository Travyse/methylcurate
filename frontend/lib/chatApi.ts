// lib/chatApi.ts
import type { LangChainMessage, LangGraphCommand } from "@assistant-ui/react-langgraph";

function apiBase() {
  // same-origin proxy: /api/proxy -> Next route -> FastAPI
  if (typeof window !== "undefined") return new URL("/api/proxy", window.location.href).toString().replace(/\/$/, "");
  return "/api/proxy";
}

async function* parseSse(res: Response): AsyncGenerator<readonly [string, any], void, unknown> {
  if (!res.body) throw new Error("SSE response has no body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const sep = buffer.indexOf("\n\n");
      if (sep === -1) break;

      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      let event = "message";
      const dataLines: string[] = [];

      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }

      const dataStr = dataLines.join("\n");
      if (!dataStr) continue;

      let data: any;
      try { data = JSON.parse(dataStr); } catch { data = dataStr; }

      yield [event, data] as const;   // ✅ tuple, iterable
    }
  }
}

export async function createThread(): Promise<{ thread_id: string }> {
  const res = await fetch(`${apiBase()}/threads`, { method: "POST" });
  if (!res.ok) throw new Error(`createThread failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getThreadState(threadId: string) {
  const res = await fetch(`${apiBase()}/threads/${encodeURIComponent(threadId)}/state`);
  if (!res.ok) throw new Error(`getThreadState failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function sendMessage(params: {
  threadId: string;
  messages?: LangChainMessage[];
  command?: LangGraphCommand | undefined;
}) {
  const res = await fetch(`${apiBase()}/threads/${encodeURIComponent(params.threadId)}/runs/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      input: params.messages?.length ? { messages: params.messages } : null,
      command: params.command ?? null,
      streamMode: ["messages", "updates"],
    }),
  });

  if (!res.ok) throw new Error(`sendMessage failed: ${res.status} ${await res.text()}`);

  // IMPORTANT: return the AsyncIterable that assistant-ui expects
  return parseSse(res);
}