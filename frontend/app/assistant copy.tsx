"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useLangGraphRuntime } from "@assistant-ui/react-langgraph";

import { createThread, getThreadState, sendMessage } from "@/lib/chatApi";
import { Thread } from "@/components/assistant-ui/thread";
import { ThreadListSidebar } from "@/components/assistant-ui/threadlist-sidebar";
import {
  SidebarProvider,
  SidebarInset,
  SidebarTrigger
} from "@/components/ui/sidebar";

export function Assistant() {
  const runtime = useLangGraphRuntime({
    stream: async function* (messages, { initialize, command }) {
      let { externalId } = await initialize();
      if (!externalId) {
        const { thread_id } = await createThread(); // calls your backend
        externalId = thread_id;
      }

      const generator = await sendMessage({
        threadId: externalId,
        messages,
        command,
      });

      yield* generator;
    },
    create: async () => {
      console.log("[create] called") 
      const { thread_id } = await createThread();
      return { externalId: thread_id };
    },
    load: async (externalId) => {
      const state = await getThreadState(externalId);
      return {
        messages: state.values.messages,
      };
    },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
    <SidebarProvider>
      <div className="flex h-dvh w-full">
        <ThreadListSidebar />
        <SidebarInset>
          {/* Add sidebar trigger, location can be customized */}
          <SidebarTrigger className="absolute top-4 left-4" />
          <Thread />
        </SidebarInset>
      </div>
    </SidebarProvider>
    </AssistantRuntimeProvider>
  );
}
