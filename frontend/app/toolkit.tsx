// src/assistant/toolkit.tsx
import { type Toolkit } from "@assistant-ui/react";
import { DataTable } from "@/components/tool-ui/data-table";
import { safeParseSerializableDataTable } from "@/components/tool-ui/data-table/schema";
import { createResultToolRenderer } from "@/components/tool-ui/shared";

export const toolkit: Toolkit = {
  geoDatasetSummary: {
    type: "backend",
    render: createResultToolRenderer({
      safeParse: safeParseSerializableDataTable,
      render: (parsed) => <DataTable rowIdKey="accession_code" {...parsed} />,
    }),
  },
};