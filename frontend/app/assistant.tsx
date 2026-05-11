"use client";

import { Thread } from "@/components/assistant-ui/thread";
import { ThreadListSidebar } from "@/components/assistant-ui/threadlist-sidebar";
import {
  SidebarProvider,
  SidebarInset,
  SidebarTrigger
} from "@/components/ui/sidebar";
import { ExternalFastApiRuntimeProvider } from "@/components/custom/ExternalFastApiRuntimeProvider"; // wherever you put it

export function Assistant() {
  return (
    <ExternalFastApiRuntimeProvider>
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
    </ExternalFastApiRuntimeProvider>
  );
}