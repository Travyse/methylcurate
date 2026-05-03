# AGENTS.md

## Project Overview

MethylCurate is a Python framework for curating DNA methylation datasets from GEO and evaluating epigenetic aging clocks.

The codebase is organized around four primary workflow modules:

- GEO data retrieval, metadata extraction, and supplementary data extraction
- Metadata harmonization
- Quality control
- Benchmarking

The agent layer coordinates these modules through the graph, node, runtime, registry, prompt, and state components under `backend/methylcurate/agent`.

Agents should orchestrate workflows and call module-level tools. They should not duplicate business logic already implemented in the tools, schemas already defined in `contracts`, or workflow state requirements already defined in `agent/state`.

## Repository Organization

Main backend package:

    backend/methylcurate

Agent-facing code:

    backend/methylcurate/agent

Agent components:

    backend/methylcurate/agent/graphs
    backend/methylcurate/agent/llm
    backend/methylcurate/agent/nodes
    backend/methylcurate/agent/prompts
    backend/methylcurate/agent/registry
    backend/methylcurate/agent/runtime
    backend/methylcurate/agent/state

Agent contracts:

    backend/methylcurate/contracts

Agent-Specific Logic:

    backend/methylcurate/tools/geo
    backend/methylcurate/tools/harmonize
    backend/methylcurate/tools/qc
    backend/methylcurate/tools/clocks

Shared Functions:

    backend/methylcurate/utils

API:

    backend/methylcurate/api

## Core Design Rule

Agents coordinate the workflow. Tools perform the work. Contracts define structured inputs and outputs. State models define workflow state.

Agent nodes should:

- select the next module-level action
- prepare inputs for tools using the existing contracts
- call the appropriate tool or workflow function
- interpret returned structured results
- update state only through the existing state model expectations
- surface recoverable errors or required user decisions

Agent nodes should not:

- redefine contract schemas
- redefine state objects
- reimplement tool logic
- silently bypass module-level checks
- fabricate sample metadata, methylation values, harmonized labels, ontology identifiers, clock predictions, or benchmark metrics

## Agent Nodes

Agent nodes are located in:

    backend/methylcurate/agent/nodes

Current top-level nodes include:

    backend/methylcurate/agent/nodes/router.py
    backend/methylcurate/agent/nodes/harmonize.py
    backend/methylcurate/agent/nodes/qc.py
    backend/methylcurate/agent/nodes/benchmarking.py
    backend/methylcurate/agent/nodes/geo

## Router Node

Location:

    backend/methylcurate/agent/nodes/router.py

The router determines the next workflow step from the current user request, conversation state, available artifacts, and module outputs.

The router may route to:

- GEO retrieval and extraction
- metadata harmonization
- quality control
- benchmarking
- summarization
- user clarification or file selection

The router should rely on the state models to determine what artifacts and prior results are available. It should not maintain a separate informal representation of workflow state.

The router should not skip required module transitions. For example, benchmarking should be routed only after the relevant upstream modules have produced the artifacts required by the benchmarking node and tools.

## GEO Module

Agent node location:

    backend/methylcurate/agent/nodes/geo

Tool locations:

    backend/methylcurate/tools/geo/download_softfile.py
    backend/methylcurate/tools/geo/extract_sample_level_metadata.py
    backend/methylcurate/tools/geo/extract_supplementary_data.py
    backend/methylcurate/tools/geo/metadata_column_extraction.py

The GEO module handles dataset access and extraction from GEO records and supplementary files.

Responsibilities include:

- downloading or locating GEO SOFT files
- extracting sample-level metadata
- identifying metadata columns relevant to requested fields
- extracting supplementary methylation data
- supporting mapping between GEO sample identifiers and methylation data columns
- producing artifacts used by harmonization, QC, and benchmarking

The GEO agent node should coordinate these tools and update the workflow state with returned artifacts according to the existing state models.

The GEO agent node may use LLM-assisted reasoning for ambiguous metadata fields or heterogeneous supplementary file layouts. LLM-assisted outputs must flow through the existing contracts and tool interfaces.

## Harmonization Module

Agent node location:

    backend/methylcurate/agent/nodes/harmonize.py

Tool location:

    backend/methylcurate/tools/harmonize/harmonize_field.py

The harmonization module standardizes heterogeneous metadata fields.

Responsibilities include:

- harmonizing raw metadata values into standardized labels
- preserving the relationship between raw values and harmonized values
- supporting fields such as sex, tissue, disease status, diagnosis, phenotype, or other metadata fields requested by the workflow
- supporting ontology-aware harmonization when the implementation provides candidate terms or mapping context
- returning harmonization results using the existing contracts

The harmonization agent node should call the harmonization tool and rely on the tool output rather than constructing harmonized artifacts independently.

## Quality Control Module

Agent node location:

    backend/methylcurate/agent/nodes/qc.py

Tool locations:

    backend/methylcurate/tools/qc/data_type_conversion.py
    backend/methylcurate/tools/qc/feature_selection.py
    backend/methylcurate/tools/qc/impute.py
    backend/methylcurate/tools/qc/qc.py
    backend/methylcurate/tools/qc/workflow.py

The QC module prepares methylation matrices for downstream clock evaluation.

Responsibilities include:

- detecting or handling methylation data representation
- converting data types when supported by the implementation
- applying QC operations
- selecting features required for downstream workflows
- imputing missing values when requested or configured
- producing QC artifacts consumed by benchmarking

The QC agent node should coordinate the QC workflow and rely on outputs from the QC tools. It should not directly mutate methylation matrices outside the QC module.

## Benchmarking Module

Agent node location:

    backend/methylcurate/agent/nodes/benchmarking.py

Tool locations:

    backend/methylcurate/tools/clocks/clock_models.py
    backend/methylcurate/tools/clocks/inference.py

The benchmarking module evaluates epigenetic aging clocks on curated methylation data.

Responsibilities include:

- loading or selecting supported clock models
- preparing inputs for clock inference
- running clock inference
- returning prediction artifacts and benchmark outputs
- summarizing results for the user or downstream workflow

The benchmarking agent node should call the clock tools and use their returned outputs. It should not fabricate predictions, infer unavailable chronological ages, or calculate metrics outside the benchmarking implementation unless that logic is explicitly part of the module.

## Contracts

Structured inputs and outputs are defined in:

    backend/methylcurate/contracts

Agents should use these contracts as the source of truth for module I/O.

When adding or modifying an agent-facing behavior:

- update or reuse the appropriate contract
- keep schemas out of `AGENTS.md`
- ensure prompts, nodes, and tools agree on the same contract objects
- avoid duplicating schema definitions in multiple places

## State Models

Workflow state requirements are defined in:

    backend/methylcurate/agent/state

Agents should use the state models as the source of truth for available artifacts, workflow progress, prior outputs, errors, and user decisions.

When modifying state behavior:

- update the state model directly
- keep state field definitions out of `AGENTS.md`
- avoid creating parallel ad hoc state dictionaries inside nodes
- make node transitions explicit through the graph and state interfaces

## Prompts

Prompt templates are located in:

    backend/methylcurate/agent/prompts

Prompts should be specific to the module or node they support.

Prompt guidance should:

- instruct the LLM to use existing contracts
- require outputs compatible with the relevant contract
- discourage unsupported inference
- ask for clarification when module inputs are ambiguous
- avoid embedding state schemas that are already defined in `agent/state`

## Registry and Runtime

Registry and runtime components are located in:

    backend/methylcurate/agent/registry
    backend/methylcurate/agent/runtime

The registry should provide the mapping between available actions, tools, nodes, or workflows as implemented by the project.

The runtime should execute graph steps, manage tool calls, and coordinate state transitions according to the existing graph and state design.

## Graphs

Graph definitions are located in:

    backend/methylcurate/agent/graphs

Graphs should encode valid workflow transitions among the module-level nodes.

Graph transitions should reflect the intended high-level workflow:

1. GEO retrieval and extraction
2. Metadata harmonization
3. Quality control
4. Benchmarking
5. Result summarization or user-facing response

Not every request needs to execute the full workflow. The router and graph should support partial workflows when the required upstream artifacts already exist in state.

## Module-Level Workflow

A typical full workflow is:

1. The user requests curation or benchmarking for a GEO dataset.
2. The router selects the next module-level action.
3. GEO tools retrieve the dataset and extract metadata or supplementary data.
4. Harmonization tools standardize requested metadata fields.
5. QC tools prepare methylation data for downstream use.
6. Clock tools run benchmarking or inference.
7. The agent summarizes generated artifacts and results.

The exact inputs, outputs, and state updates for each step are governed by `contracts` and `agent/state`.

## Development Guidelines

When adding or modifying agent behavior:

1. Keep orchestration in `agent/nodes`.
2. Keep deterministic implementation in `tools`.
3. Keep structured I/O in `contracts`.
4. Keep workflow state definitions in `agent/state`.
5. Keep graph transitions in `agent/graphs`.
6. Keep prompts in `agent/prompts`.
7. Avoid duplicating schema or state definitions in documentation.
8. Add tests at the module, node, and workflow levels where appropriate.

## Module Responsibilities Summary

### GEO

Owns GEO download, metadata extraction, supplementary data extraction, and GEO-specific sample mapping.

### Harmonization

Owns standardization of metadata values and preservation of raw-to-harmonized mappings.

### Quality Control

Owns methylation matrix preparation, data conversion, QC operations, feature selection, and imputation.

### Benchmarking

Owns clock model loading, inference, prediction outputs, and benchmark result generation.

### Agents

Own routing, orchestration, prompt-mediated reasoning, state-aware decision making, and user-facing workflow coordination.

### Contracts

Own structured input and output schemas.

### State Models

Own workflow state requirements and artifact tracking.

## Testing

This project uses an existing Mamba environment named `dnam-aging-agentic-ai`.

Always run tests with:

```bash
mamba run -n dnam-aging-agentic-ai pytest
```

Do not run plain `pytest`.

Do not create a new Conda environment.