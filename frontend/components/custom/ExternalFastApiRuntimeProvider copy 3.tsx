"use client";

import { ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
  Tools,
  useAui,
  type Toolkit,
  type AppendMessage,
  type ThreadMessageLike,
} from "@assistant-ui/react";
import { DataTable } from "@/components/tool-ui/data-table";
import { safeParseSerializableDataTable } from "@/components/tool-ui/data-table/schema";

/* ----------------------------
 * Toolkit
 * ---------------------------- */
export const toolkit: Toolkit = {
  geoDatasetSummary: {
    type: "backend",
    render: ({ result }) => {
      const parsed = safeParseSerializableDataTable(result);
      if (!parsed) return null;
      return <DataTable rowIdKey="accession_code" {...parsed} />;
    },
  },
};

type LCMsg =
  | { type: "human"; content: any }
  | { type: "ai"; content: any }
  | { type: "tool"; content: any; artifact?: any; name?: string; tool_call_id?: string };

type ThreadInfo = {
  thread_id: string;
  title?: string | null;
  updated_at?: number | null;
  archived?: boolean | null;
};

function apiBase() {
  return new URL("/api/proxy", window.location.href).toString().replace(/\/$/, "");
}

/* ----------------------------
 * SSE parsing
 * ---------------------------- */
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
      try {
        data = JSON.parse(dataStr);
      } catch {
        data = dataStr;
      }

      yield [event, data] as const;

      // Your FastAPI emits `event: done`, not `end`
      if (event === "done") return;
    }
  }
}

/* ----------------------------
 * API helpers
 * ---------------------------- */
async function createThread(): Promise<string> {
  const res = await fetch(`${apiBase()}/threads`, { method: "POST" });
  if (!res.ok) throw new Error(`createThread failed: ${res.status} ${await res.text()}`);
  const j = await res.json();
  return j.thread_id;
}

async function listThreads(): Promise<ThreadInfo[]> {
  const res = await fetch(`${apiBase()}/threads`);
  if (!res.ok) throw new Error(`listThreads failed: ${res.status} ${await res.text()}`);
  const j = await res.json();
  return (j?.threads ?? []) as ThreadInfo[];
}

async function deleteThread(threadId: string): Promise<void> {
  const res = await fetch(`${apiBase()}/threads/${encodeURIComponent(threadId)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`deleteThread failed: ${res.status} ${await res.text()}`);
}

async function loadState(threadId: string): Promise<LCMsg[]> {
  const res = await fetch(`${apiBase()}/threads/${encodeURIComponent(threadId)}/state`);
  if (!res.ok) throw new Error(`getThreadState failed: ${res.status} ${await res.text()}`);
  const j = await res.json();
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

/* ----------------------------
 * Message conversion for assistant-ui
 * ---------------------------- */
function toThreadMessageLike(m: LCMsg): ThreadMessageLike {
  if (m.type === "tool" && m.artifact && m.artifact.columns && (m.artifact.data || m.artifact.rows)) {
    const toolName =
      m.name ?? (m.artifact.id === "geo-dataset-summary" ? "geoDatasetSummary" : "unknownTool");

    const result = {
      ...m.artifact,
      data: m.artifact.data ?? m.artifact.rows,
    };

    return {
      role: "assistant",
      content: [
        ...(m.content ? [{ type: "text", text: String(m.content) }] : []),
        {
          type: "tool-call",
          toolCallId: m.tool_call_id ?? `synthetic-${m.artifact.id ?? crypto.randomUUID()}`,
          toolName,
          args: {},
          result,
          status: { type: "complete" },
        },
      ],
    };
  }

  const role = m.type === "human" ? "user" : "assistant";
  const text =
    typeof m.content === "string"
      ? m.content
      : Array.isArray(m.content)
        ? m.content.map((p: any) => (p?.type === "text" ? p.text ?? "" : "")).join("")
        : JSON.stringify(m.content);

  return { role, content: [{ type: "text", text }] };
}

/* ----------------------------
 * Provider
 * ---------------------------- */
export function ExternalFastApiRuntimeProvider({ children }: { children: ReactNode }) {
  const [isRunning, setIsRunning] = useState(false);

  // Active thread + messages (external store)
  const [threadId, setThreadId] = useState<string | null>(null);
  const [messages, setMessages] = useState<LCMsg[]>([]);

  // Thread list state (THIS is what you were missing)
  const [threads, setThreads] = useState<ThreadInfo[]>([]);

  // Initial load: list threads, pick one, load its state, or create if none exist
  useEffect(() => {
    (async () => {
      const t = await listThreads().catch(() => [] as ThreadInfo[]);
      setThreads(t);

      const saved = localStorage.getItem("thread_id");
      const initial =
        (saved && t.some((x) => x.thread_id === saved)) ? saved :
        t[0]?.thread_id ?? null;

      if (initial) {
        setThreadId(initial);
        localStorage.setItem("thread_id", initial);
        setMessages(await loadState(initial).catch(() => []));
        return;
      }

      const id = await createThread();
      const t2 = await listThreads().catch(() => [] as ThreadInfo[]);
      setThreads(t2);
      setThreadId(id);
      localStorage.setItem("thread_id", id);
      setMessages([]);
    })().catch((e) => {
      console.error(e);
      setThreads([]);
      setThreadId(null);
      setMessages([]);
    });
  }, []);

  const onNew = useCallback(async (msg: AppendMessage) => {
    if (!threadId) throw new Error("No thread selected yet");

    const first = msg.content?.[0];
    if (!first || first.type !== "text") throw new Error("Only text messages are supported");
    const userText = first.text ?? "";

    const next = [...messages, { type: "human", content: userText } as LCMsg];
    setMessages(next);

    setIsRunning(true);
    try {
      const sse = await streamRun({ threadId, messages: next });

      for await (const [event, data] of sse) {
        if (event === "messages") {
          const incoming = (data?.messages ?? []) as LCMsg[];
          setMessages((cur) => [...cur, ...incoming]);
        }
        if (event === "done") break;
      }

      // refresh thread list metadata after a run (optional but useful)
      setThreads(await listThreads().catch(() => threads));
    } finally {
      setIsRunning(false);
    }
  }, [threadId, messages, threads]);

  const threadEntries = threads.map(t => ({
    threadId: t.thread_id,          // must match what assistant-ui expects
    title: t.title ?? undefined,
    updatedAt: t.updated_at ?? undefined,
  }));

  const activeThreadId =
    threadId && threadEntries.some(e => e.threadId === threadId)
      ? threadId
      : undefined;

  const runtime = useExternalStoreRuntime({
    isRunning,
    messages,
    convertMessage: toThreadMessageLike,
    onNew,

    adapters: {
      threadList: {
        threadId: activeThreadId,
        threads: threadEntries,

        onSwitchToNewThread: async () => {
          const id = await createThread();
          const t = await listThreads().catch(() => [] as ThreadInfo[]);
          setThreads(t);
          setThreadId(id);
          localStorage.setItem("thread_id", id);
          setMessages(await loadState(id).catch(() => []));
        },

        onSwitchToThread: async (id: string) => {
          setThreadId(id);
          localStorage.setItem("thread_id", id);
          setMessages(await loadState(id).catch(() => []));
        },

        onDelete: async (id: string) => {
          await deleteThread(id);
          const t = await listThreads().catch(() => [] as ThreadInfo[]);
          setThreads(t);

          if (id === threadId) {
            const next = t[0]?.thread_id ?? null;
            if (next) {
              setThreadId(next);
              localStorage.setItem("thread_id", next);
              setMessages(await loadState(next).catch(() => []));
            } else {
              setThreadId(null);
              localStorage.removeItem("thread_id");
              setMessages([]);
            }
          }
        },
      },
    },
  });

  const aui = useAui({ tools: Tools({ toolkit }) });

  return (
    <AssistantRuntimeProvider runtime={runtime} aui={aui}>
      {children}
    </AssistantRuntimeProvider>
  );
}