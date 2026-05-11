"use client";

import React, { ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import {
  AssistantRuntimeProvider,
  RuntimeAdapterProvider,
  unstable_useRemoteThreadListRuntime as useRemoteThreadListRuntime,
  type unstable_RemoteThreadListAdapter as RemoteThreadListAdapter,
  type ThreadHistoryAdapter,
  useExternalStoreRuntime,
  Tools,
  useAui,
  Suggestions,
  type Toolkit,
  type AppendMessage,
  type ThreadMessageLike,
} from "@assistant-ui/react";

import { DataTable } from "@/components/tool-ui/data-table";
import { safeParseSerializableDataTable } from "@/components/tool-ui/data-table/schema";

import { ToolUI, createResultToolRenderer } from "@/components/tool-ui/shared";

import { Plan } from "@/components/tool-ui/plan";
import { safeParseSerializablePlan } from "@/components/tool-ui/plan/schema";

import { OptionList } from "@/components/tool-ui/option-list";
import { safeParseSerializableOptionList } from "@/components/tool-ui/option-list/schema";

import { ProgressTracker } from "@/components/tool-ui/progress-tracker";
import { safeParseSerializableProgressTracker } from "@/components/tool-ui/progress-tracker/schema";

import { safeParseSerializableApprovalCard } from "@/components/tool-ui/approval-card/schema";
import { ApprovalCard } from "@/components/tool-ui/approval-card";

/* ----------------------------
 * Toolkit
 * ---------------------------- */
export const toolkit: Toolkit = {
  geoDatasetSummary: {
    type: "backend",
    render: ({ result }) => {
      const parsed = safeParseSerializableDataTable(result);
      if (!parsed) return null;
      return <div><br /><DataTable {...parsed} /><br /></div>;
    },
  },
  geoRetrievalProgress: {
    type: "backend",
    render: createResultToolRenderer({
      safeParse: safeParseSerializablePlan,
      render: (parsedResult) => (
        <div>
          <br />
        <ToolUI id={parsedResult.id}>
          <ToolUI.Surface>
            <Plan {...parsedResult} />
          </ToolUI.Surface>
        </ToolUI>
        </div>
      ),
    }),
  },
  geoSupplementaryFileSelection: {
    type: "backend",
    render: ({ args, toolCallId, result, addResult }) => {
      console.log("[geoSupplementaryFileSelection render]", {
        toolCallId,
        hasArgs: Boolean(args),
        hasResult: Boolean(result),
        hasAddResult: Boolean(addResult),
      })
      const rawArgs = args as Record<string, unknown> | undefined;
      const parsedArgs = safeParseSerializableOptionList({
        ...(rawArgs ?? {}), id: (rawArgs?.id as string) ?? `format-selection-${toolCallId}`
      });
      if (!parsedArgs) return null;
      return result ? (
        <div>
        <br />
        <OptionList
          {...parsedArgs}
          value={undefined}
          choice={result.action === "confirm" ? result.data?.selections : undefined} />
        </div>
      ) : (
        <div>
        <br />
        <OptionList {...parsedArgs} value={undefined} onAction={(actionId, selection) => {
          console.log("[OPtionList] onAction", { actionId, selection });
            if (actionId === "skip") {
              void addResult?.({ action: "skip", data: null });
            } else if (actionId === "confirm") {
              void addResult?.({ action: "confirm", data: { selections: selection } });
            }
          }}
        />
        </div>
    );
    },
  },
  requestApproval: {
    type: "backend",
    render: ({ args, toolCallId, result, addResult }) => {
      const rawArgs = args as Record<string, unknown> | undefined;
      const parsedArgs = safeParseSerializableApprovalCard({
        ...(rawArgs ?? {}),
        id: (rawArgs?.id as string) ?? `approval-card-${toolCallId}`,
      });
      if (!parsedArgs) return null;
      return (
        <div>
        <ApprovalCard {...parsedArgs} choice="approved" />
        <br />
        </div>
      )
    }
  },
  benchmarkProgress: {
    type: "backend",
    render: ({ result }) => {
      const parsed = safeParseSerializableProgressTracker(result);
      if (!parsed) return null;
      return (
        <div>
        <ProgressTracker {...parsed} />
        <br />
        </div>
      );
    },
  }
  
};

type LCMsg =
  | { type: "human"; id: string; content: any; additional_kwargs?: any}
  | { type: "ai"; id: string; content: any; additional_kwargs?: any }
  | { type: "tool"; id: string; content: any; artifact?: any; name?: string; tool_call_id?: string; additional_kwargs?: any };

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
 * SSE parsing (KEEP THIS)
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

      if (event === "done") return;
    }
  }
}

/* ----------------------------
 * API helpers
 * ---------------------------- */
async function createThread(): Promise<string> {
  const res = await fetch(`${apiBase()}/threads`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  const j = await res.json();
  return j.thread_id;
}

async function listThreads(): Promise<ThreadInfo[]> {
  const res = await fetch(`${apiBase()}/threads`);
  if (!res.ok) throw new Error(await res.text());
  const j = await res.json();
  return (j?.threads ?? []) as ThreadInfo[];
}

async function deleteThread(threadId: string): Promise<void> {
  const res = await fetch(`${apiBase()}/threads/${encodeURIComponent(threadId)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
}

async function loadState(threadId: string): Promise<LCMsg[]> {
  const res = await fetch(`${apiBase()}/threads/${encodeURIComponent(threadId)}/state`);
  if (!res.ok) throw new Error(await res.text());
  const j = await res.json();
  return (j?.values?.messages ?? []) as LCMsg[];
}

async function streamRun(params: { threadId: string; messages: LCMsg[]; files?: Array<{ name: string; content: string }> }) {
  const res = await fetch(`${apiBase()}/threads/${encodeURIComponent(params.threadId)}/runs/stream`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({
      input: { messages: params.messages, files: params.files ?? [] },
      command: null,
      streamMode: ["messages", "updates"],
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return parseSse(res);
}

async function resumeRun(params: {
  threadId: string;
  toolName: string;
  toolCallId: string;
  result: any;
}) {
  console.log("[resumeRun] called with", params);
  const res = await fetch(`${apiBase()}/threads/${encodeURIComponent(params.threadId)}/runs/stream`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({
      input: null,
      command: { resume: { tool: { name: params.toolName, tool_call_id: params.toolCallId }, data: params.result } },
      streamMode: ["messages", "updates"],
      }),
    },
  );
  if (!res.ok) throw new Error(await res.text());
  return parseSse(res);
}

/* ----------------------------
 * Convert backend msgs -> assistant-ui msgs
 * ---------------------------- */
function upsertById(prev: LCMsg[], incoming: LCMsg[]): LCMsg[] {
  if (incoming.length === 0) return prev;

  // index existing messages by id
  const index = new Map<string, number>();
  const next = prev.slice();

  for (let i = 0; i < next.length; i++) {
    const id = (next[i] as any).id;
    if (id) index.set(id, i);
  }

  for (const msg of incoming) {
    const id = (msg as any).id;
    if (!id) {
      // If you ever receive messages without ids, you *can't* upsert them safely.
      next.push(msg);
      continue;
    }

    const existingIdx = index.get(id);
    if (existingIdx == null) {
      index.set(id, next.length);
      next.push(msg);
    } else {
      // Replace semantics. If you want merge semantics, do it here.
      next[existingIdx] = msg;
    }
  }

  return next;
}

function toThreadMessageLike(m: LCMsg): ThreadMessageLike {
  if (m.type === "tool" && m.artifact) {
    const toolName = m.additional_kwargs?.name || m.name;
    let result;
    let args;

    if (toolName === "geoDatasetSummary") {
      result = { ...m.artifact, data: m.artifact.data ?? m.artifact.rows };
    } else if (toolName === "geoRetrievalProgress") {
      result = { ...m.artifact }
    } else {
      result = { ...m.artifact };
    }
    
    const isInteractiveTool = toolName === "geoSupplementaryFileSelection";

    if (toolName === "geoDatasetSummary") {
      args = [];
      result = { ...m.artifact, data: m.artifact.data ?? m.artifact.rows };
    } else if (toolName === "geoRetrievalProgress") {
      args = [];
      result = { ...m.artifact }
    } else if (toolName === "geoSupplementaryFileSelection") {
      args = { ...m.artifact };
      result = null
    } else if (toolName === "requestApproval") {
      args = { ...m.artifact };
      result = null;
    } else {
      args = [];
      result = { ...m.artifact };
    }

    return {
      id: m.id,
      role: "assistant",
      content: [
        ...(m.content ? [{ type: "text" as const, text: String(m.content) }] : []),
        {
          type: "tool-call",
          toolCallId: m.tool_call_id ?? `synthetic-${m.artifact.id ?? crypto.randomUUID()}`,
          toolName,
          args: args,
          result,
          status: { type: "complete" },
        },
      ] as any,
    };
  }

  const role = m.type === "human" ? "user" : "assistant";
  const text =
    typeof m.content === "string"
      ? m.content
      : Array.isArray(m.content)
        ? m.content.map((p: any) => (p?.type === "text" ? p.text ?? "" : "")).join("")
        : JSON.stringify(m.content);

  return { id: m.id, role, content: [{ type: "text", text }] };
}

/* ----------------------------
 * Remote thread list adapter
 * ---------------------------- */
const remoteAdapter: any = {
  async list() {
    const t = await listThreads();
    return {
      threads: t.map((x) => ({
        remoteId: x.thread_id,
        title: x.title ?? undefined,
        status: x.archived ? "archived" : "regular",
      })),
    };
  },
  async initialize(_threadId: string) {
    const id = await createThread();
    return { remoteId: id, externalId: id };
  },
  async delete(remoteId: string) {
    await deleteThread(remoteId);
  },
  async rename(_remoteId: string, _title: string) {},
  async archive(_remoteId: string) {},
  async unarchive(_remoteId: string) {},
  async generateTitle(_remoteId: string, _messageHistory: any) {
    return "";
  },
  async fetch(_remoteId: string) {
    return { messages: [] };
  },
};

export function ExternalFastApiRuntimeProvider({ children }: { children: ReactNode }) {
  const auiTop = useAui({
    tools: Tools({ toolkit }),
    suggestions: Suggestions([
      "What can you help me with?",
      "Download GSE38608",
    ])
  });

  const runtime = useRemoteThreadListRuntime({
    adapter: {
      ...remoteAdapter,

      unstable_Provider: ({ children }) => {
        // optional; ok to keep
        const history = useMemo<ThreadHistoryAdapter>(() => ({ load: async () => ({ messages: [] }), append: async () => {} }), []);
        return <RuntimeAdapterProvider adapters={{ history }}>{children}</RuntimeAdapterProvider>;
      },
    },

    runtimeHook: () => {
      const aui = useAui();

      const item: any = aui.threadListItem();

      // ✅ THIS is the key: use the reactive hook if your version has it.
      const state = typeof item.useState === "function" ? item.useState() : item.getState?.() ?? {};
      const remoteId: string | null = state?.remoteId ?? null;

      const [messages, setMessages] = useState<LCMsg[]>([]);
      const [isRunning, setIsRunning] = useState(false);

      // ✅ reload when user switches threads in sidebar
      useEffect(() => {
        (async () => {
          if (!remoteId) {
            setMessages([]);
            return;
          }
          setMessages(await loadState(remoteId).catch(() => []));
        })();
      }, [remoteId]);

      useEffect(() => {
        let cancelled = false;

        (async () => {
          if (!remoteId) return;

          const threads = await listThreads().catch(() => [] as ThreadInfo[]);
          const exists = threads.some((t) => t.thread_id === remoteId);

          if (!exists && !cancelled) {
            // selects/creates a valid thread via adapter.initialize()
            await aui.threadListItem().initialize().catch(console.error);
            // messages will reload on remoteId change
          }
        })();

        return () => {
          cancelled = true;
        };
      }, [remoteId, aui]);
      const onNew = useCallback(
        async (msg: AppendMessage) => {
          const init = await aui.threadListItem().initialize();
          const tid = init?.remoteId as string | undefined;
          if (!tid) throw new Error("No active thread");

          const first = msg.content?.[0];
          if (!first || first.type !== "text") throw new Error("Only text supported");
          const userText = first.text ?? "";

          const fileEntries: Array<{ name: string; content: string }> = [];
          const attachments = (msg as any).attachments as Array<{ file?: File; name?: string }> | undefined;
          if (attachments?.length) {
            const results = await Promise.all(
              attachments.map(async (att) => {
                if (!att.file) return null;
                const buf = await att.file.arrayBuffer();
                const bytes = new Uint8Array(buf);
                let binary = "";
                for (let i = 0; i < bytes.length; i++) {
                  binary += String.fromCharCode(bytes[i]);
                }
                return { name: att.name ?? att.file.name, content: btoa(binary) };
              }),
            );
            for (const r of results) {
              if (r) fileEntries.push(r);
            }
          }

          setIsRunning(true);
          try {
            const cur = await loadState(tid).catch(() => []);
            const payload = [...cur, { id: crypto.randomUUID(), type: "human", content: userText } as LCMsg];

            const sse = await streamRun({ threadId: tid, messages: payload, files: fileEntries });
            for await (const [event, data] of sse) {
              if (event === "messages") {
                const incoming = (data?.messages ?? []) as LCMsg[];
                  for (const m of incoming) {
                    if (m.type === "tool") {
                      console.log("RAW TOOL MSG", m);
                      console.log("CONVERTED", toThreadMessageLike(m));
                    }
                  }
                if (incoming.length) setMessages((cur2) => upsertById(cur2, incoming));
              }
              if (event === "done") break;
            }
          } finally {
            setIsRunning(false);
          }
        },
        [aui],
      );
      const onAddToolResult = useCallback(
        async (options: { messageId: string, toolCallId: string, result: any }) => {

        const init = await aui.threadListItem().initialize();
        const tid = init?.remoteId as string | undefined;
        if (!tid) throw new Error("No active thread");

          let set_approval = {
            artifact: {
              id: options.toolCallId,
              title: "Downloading supplementary files...",
              description: "Your selections are being processed. This may take a few moments.",
              choice: "approved"
            },
            content: null,
            id: options.toolCallId,
            tool_call_id: options.toolCallId,
            type: "tool",
            additional_kwargs: {
              name: "requestApproval",
              created_at: new Date().toISOString(),
            },
          }

          const langgraphThreadId = remoteId!;

          const toolName = "geoSupplementaryFileSelection";

          setIsRunning(true);
          try {
            const cur = await loadState(tid).catch(() => []);
            setMessages((cur) => upsertById(cur, [set_approval as LCMsg]));

          const sse = await resumeRun({
            threadId: langgraphThreadId,
            toolName,
            toolCallId: options.toolCallId,
            result: options.result,
          });
          for await (const [event, data] of sse) {
            if (event === "messages") {
              const incoming = (data?.messages ?? []) as LCMsg[];
                for (const m of incoming) {
                  if (m.type === "tool") {
                    console.log("RAW TOOL MSG", m);
                    console.log("CONVERTED", toThreadMessageLike(m));
                  }
                }
              // if (incoming.length) setMessages((cur2) => [...cur2, ...incoming]); TODO: Check
              if (incoming.length) setMessages((cur2) => upsertById(cur2, incoming));
            }
            if (event === "done") break;
          }
          } finally {
            setIsRunning(false);
          }
        },
        [remoteId]
      )
      const cleanedMessages = useMemo(
        () =>
          messages.filter((m) => {
            if (m.type !== "human") return true;
            const text =
              typeof m.content === "string"
                ? m.content
                : Array.isArray(m.content)
                  ? m.content.map((p: any) => (p?.type === "text" ? p.text ?? "" : "")).join("")
                  : String(m.content ?? "");
            return text.trim().length > 0;
          }),
        [messages]
      );
      return useExternalStoreRuntime({
        isRunning,
        messages: cleanedMessages,
        convertMessage: toThreadMessageLike,
        onNew,
        onAddToolResult
      });
    },
  });
  
  return (
    <AssistantRuntimeProvider runtime={runtime} aui={auiTop}>
      {children}
    </AssistantRuntimeProvider>
  );
}