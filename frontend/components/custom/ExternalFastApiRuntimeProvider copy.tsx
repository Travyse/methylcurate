"use client";

import { ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
  type AppendMessage,
  type ThreadMessageLike,
} from "@assistant-ui/react";

type LCMsg =
  | { type: "human"; content: any }
  | { type: "ai"; content: any }
  | { type: "tool"; content: any; artifact?: any; name?: string; tool_call_id?: string };

function apiBase() {
  return new URL("/api/proxy", window.location.href).toString().replace(/\/$/, "");
}

/** Parse SSE into [event, data] tuples */
async function* parseSse(res: Response): AsyncGenerator<readonly [string, any], void, unknown> {
  if (!res.body) throw new Error("SSE response has no body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) return;

    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const sep = buffer.indexOf("\n\n");
      if (sep === -1) break;

      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      // skip keep-alive/comment-only frames
      if (!raw.trim() || raw.startsWith(":")) continue;

      let event = "message";
      const dataLines: string[] = [];

      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length === 0) continue;

      const dataStr = dataLines.join("\n");
      let data: any;
      try { data = JSON.parse(dataStr); } catch { data = dataStr; }

      yield [event, data] as const;

      if (event === "end") return;
    }
  }
}

async function createThread(): Promise<string> {
  const res = await fetch(`${apiBase()}/threads`, { method: "POST" });
  if (!res.ok) throw new Error(`createThread failed: ${res.status} ${await res.text()}`);
  const j = await res.json();
  return j.thread_id;
}

async function loadState(threadId: string): Promise<LCMsg[]> {
  const res = await fetch(`${apiBase()}/threads/${encodeURIComponent(threadId)}/state`);
  if (!res.ok) throw new Error(`getThreadState failed: ${res.status} ${await res.text()}`);
  const j = await res.json();
  // Expect { values: { messages: [...] } }
  return (j?.values?.messages ?? []) as LCMsg[];
}

async function streamRun(params: { threadId: string; messages: LCMsg[] }) {
  const res = await fetch(`${apiBase()}/threads/${encodeURIComponent(params.threadId)}/runs/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      input: { messages: params.messages },
      command: null,
      streamMode: ["messages", "updates"],
    }),
  });
  if (!res.ok) throw new Error(`stream failed: ${res.status} ${await res.text()}`);
  return parseSse(res);
}

/** Convert stored messages to assistant-ui ThreadMessageLike */
function toThreadMessageLike(m: LCMsg): ThreadMessageLike {
  const role = m.type === "human" ? "user" : "assistant";

  // Your backend sometimes sends content as string or as [{type:"text",text:"..."}]
  let text = "";
  if (typeof m.content === "string") text = m.content;
  else if (Array.isArray(m.content)) {
    text = m.content
      .map((p: any) => (p?.type === "text" ? (p.text ?? "") : ""))
      .join("");
  } else text = JSON.stringify(m.content);

  return { role, content: [{ type: "text", text }] };
}

export function ExternalFastApiRuntimeProvider({ children }: { children: ReactNode }) {
  const [isRunning, setIsRunning] = useState(false);

  // This is the “external store”: messages + selected thread id
  const [threadId, setThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<LCMsg[]>([]);

  // Ensure we have a thread selected on first load
  useEffect(() => {
    const existing = localStorage.getItem("thread_id");
    if (existing) {
      setThreadId(existing);
      loadState(existing).then(setMessages).catch(() => setMessages([]));
      return;
    }

    // create a new one
    createThread().then((id) => {
      localStorage.setItem("thread_id", id);
      setThreadId(id);
      setMessages([]);
    });
  }, []);

  const onNew = useCallback(async (msg: AppendMessage) => {
    if (!threadId) throw new Error("No thread selected yet");

    // Only accept plain text composer for now
    const first = msg.content?.[0];
    if (!first || first.type !== "text") throw new Error("Only text messages are supported");

    const userText = first.text ?? "";

    // 1) append user message locally immediately
    const next = [...messages, { type: "human", content: userText } as LCMsg];
    setMessages(next);

    // 2) stream assistant response from backend
    setIsRunning(true);
    try {
      const sse = await streamRun({ threadId, messages: next });

      for await (const [event, data] of sse) {
        if (event === "messages") {
          const incoming = (data?.messages ?? []) as LCMsg[];

          // append whatever backend sends (ai messages)
          setMessages((cur) => [...cur, ...incoming]);
        }

        // ignore updates for now unless you want interrupts/tools
        if (event === "end") break;
      }
    } finally {
      setIsRunning(false);
    }
  }, [threadId, messages]);

  const runtime = useExternalStoreRuntime({
    isRunning,
    messages,
    convertMessage: toThreadMessageLike,
    onNew,
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      {children}
    </AssistantRuntimeProvider>
  );
}