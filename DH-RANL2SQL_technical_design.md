# DH-RANL2SQL — Technical Architecture & Implementation Design

**Full name:** Dynamic Hybrid Retrieval-Augmented NL2SQL  
**Document type:** Technical architecture and implementation blueprint  
**Status:** Proposed architecture for design review and phased implementation  
**Primary platform:** Google Cloud Platform (GCP)  
**Reference stack:** Python, FastAPI, Pydantic v2, Google Agent Development Kit (ADK), Gemini on Vertex AI, BigQuery, SQLGlot

---

## 1. Executive summary

DH-RANL2SQL is an agent-orchestrated architecture for answering analytical questions over datasets that combine structured columns with long, unstructured narrative fields.

A conventional NL2SQL system is often effective when the answer can be derived from known tables, columns, filters, joins, and aggregations. It becomes less reliable when the user asks for a concept expressed in free text—for example, *“cases where the notes describe delayed or missing approval”*—and then requests an exact count, average, sum, or comparison over that population.

DH-RANL2SQL addresses this by separating the problem into two related but distinct tasks:

1. **Discover and interpret evidence in narrative data.** Use exact matching, lexical retrieval, semantic retrieval, and evidence fusion to identify relevant records and formulate a testable semantic predicate.
2. **Perform governed analytical computation.** Establish whether the semantic predicate has been evaluated over the complete eligible population, then generate, validate, and execute SQL against approved BigQuery objects.

The architecture is controlled by two central roles:

- **Query Planner Agent:** understands the question and produces a typed execution plan identifying which specialist agents are needed and how their outputs depend on one another.
- **Orchestrator Agent:** executes that plan. It invokes the selected specialist agents, manages parallel work, shared state, budgets, retries, deadlines, and stopping conditions.

Nine specialist agents provide focused capabilities. They are callable by the orchestrator through an agent-as-tool interface. They are still agents with their own responsibilities and contracts; they are not interchangeable with the deterministic tools and cloud services that they may call internally.

The governing correctness principle is:

> **A retrieval result is evidence for finding records; it is not proof that every qualifying record has been found. Never present an aggregate over top-K search results as an exact aggregate over the full population.**

For semantic aggregation, DH-RANL2SQL must establish both:

- **Population coverage:** every record in the eligible population has been evaluated under the requested predicate, or a formally approved complete deterministic method has been used.
- **Classification quality:** the semantic classifier has met task-specific quality criteria on representative, labelled evaluation data.

If either condition is not met, the system must narrow the question, perform a bounded full-population evaluation, return a clearly labelled exploratory result, or abstain. It must not silently promote an incomplete sample into a population-level answer.

---

## 2. Problem definition and target workload

### 2.1 Data characteristics

The initial target environment is expected to contain:

- Approximately **5–30 BigQuery tables or views** in the governed analytical scope.
- Around **1–10 million rows** across the relevant tables, depending on the workload.
- A mixture of structured fields (dates, amounts, status, sponsor, provider, region, identifiers, and other dimensions) and narrative text fields.
- Approximately **2–5 paragraphs** of narrative content in some records, where important business context is not represented consistently in structured columns.
- Multiple possible table grains and relationships, with a risk of incorrect joins and aggregation inflation if those relationships are not understood.

These figures are planning assumptions, not hard system limits. They must be confirmed through data profiling and workload measurement.

### 2.2 Representative questions

DH-RANL2SQL should support multiple query families:

| Query family | Example | Main work required |
|---|---|---|
| Structured analytics | “What was the average claim amount by sponsor last quarter?” | Metadata resolution and SQL |
| Exact text lookup | “How many records contain the exact phrase ‘prior authorization’?” | Exact matching and SQL |
| Lexical lookup | “How many records mention approval delays?” | Lexical search and SQL |
| Semantic lookup | “How many cases describe patients waiting too long for approval?” | Semantic retrieval and semantic evaluation |
| Hybrid lookup | “Average claim amount for cases where notes describe delayed approval.” | Lexical + semantic retrieval, predicate evaluation, SQL |
| Multi-table analytics | “Compare those cases by sponsor, region, and eligibility month.” | Schema/relationship discovery, retrieval, predicate evaluation, safe joins and SQL |

The planner should not invoke every agent for every question. A simple numeric query should not trigger embedding generation or narrative classification. A broad semantic aggregation should not skip population-coverage checks.

### 2.3 Primary design objectives

1. Correctly interpret mixed structured and narrative questions.
2. Select only the capabilities needed for the current question.
3. Keep the runtime flow inspectable, deterministic where possible, and bounded.
4. Preserve provenance from every retrieved or classified record to its source table, row, and narrative field.
5. Prevent incomplete retrieval from being mistaken for complete population evaluation.
6. Generate read-only SQL against an approved schema and validate it before execution.
7. Make schema growth and additional source tables manageable through metadata discovery rather than hardcoded prompts.
8. Measure answer quality, retrieval coverage, cost, and latency independently.
9. Deploy and operate the system using managed GCP services wherever practical.

---

## 3. Core architecture and terminology

### 3.1 The execution principle

The system follows this rule:

> **The planner decides what work is required. The orchestrator decides when and how to run it. Specialist agents perform the assigned work. Deterministic validators and policy gates decide whether an output may proceed.**

The planner is responsible for reasoning about the question. The orchestrator is responsible for controlled execution. Specialist agents are narrow and task-specific. Tools are the concrete operations used inside those agents, such as a BigQuery metadata query, a `SEARCH` query, an embedding request, a `VECTOR_SEARCH` query, a SQLGlot AST check, or a BigQuery dry run.

### 3.2 Agent versus tool versus service

These terms must remain distinct in the implementation and documentation:

| Concept | Meaning in DH-RANL2SQL | Example |
|---|---|---|
| **Agent** | A bounded reasoning or task-execution component with a defined role, instructions, inputs, outputs, and failure behavior. | Semantic Evaluation Agent |
| **Agent-as-tool interface** | A way for another agent or the orchestrator to invoke a specialist agent as a callable capability. The specialist remains an agent; the interface makes it invocable. | An ADK `AgentTool`-style wrapper or equivalent explicit invocation adapter |
| **Tool** | A concrete function or API call used by an agent. It should do one narrow operation and return structured data. | `run_bigquery_search()`, `create_query_embedding()`, `validate_sql_ast()` |
| **Service** | A deployed or managed runtime component that hosts or supports the system. | FastAPI on Cloud Run, BigQuery, Vertex AI |
| **Workflow node** | A step in the execution graph. It may be a Python function, an agent, a tool, or a nested workflow. | Evidence assembly node |

Not every specialist agent needs an LLM. A deterministic specialist agent is still useful when it provides a cohesive capability, a stable contract, and traceable execution. For example, Evidence Fusion should normally use deterministic ranking logic rather than an LLM deciding which retrieval result ranks higher.

The recommended ADK pattern is to use a graph-based workflow for controlled execution and invoke specialist agents through explicit agent-as-tool calls or equivalent adapters. Current ADK guidance describes graph-based `Workflow` orchestration and explicit agent invocation patterns; older template workflow primitives have been superseded in newer ADK versions. Confirm the exact API against the pinned ADK release before implementation. See [ADK Workflow](https://github.com/google/adk-python/blob/main/docs/guides/workflow/workflow/index.md), [ADK multi-agent patterns](https://adk.dev/workflows/patterns/), and [ADK agent/tool patterns](https://github.com/google/adk-docs/blob/main/docs/workflows/patterns.md).

### 3.3 Runtime components

At a high level, the architecture has these stages:

1. User question enters through the API.
2. Query Planner Agent creates a typed `ExecutionPlan`.
3. Orchestrator Agent validates and executes that plan.
4. Selected specialist agents discover metadata, retrieve evidence, evaluate narrative predicates, generate SQL, verify results, and synthesize the answer.
5. Deterministic quality and policy gates control whether the workflow can continue.
6. BigQuery executes only approved read-only analytical SQL.
7. The answer returns with methodology, SQL, evidence references, and limitations.

A separate offline indexing and evaluation plane prepares text, embeddings, search indexes, cached predicate evaluations, metadata, and test datasets.

---

## 4. Agent catalogue

There are **nine specialist agents** in the current architecture, in addition to the Query Planner Agent and Orchestrator Agent. The planner selects the specialists required by the question. The orchestrator invokes them and coordinates their execution.

### 4.1 A1 — Schema & Metadata Agent

**Purpose:** Identify the approved data structures needed to answer the question and provide reliable context to downstream agents.

**Typical implementation:** A narrowly scoped ADK agent supported by deterministic metadata tools. The LLM may help interpret business language; the actual schema discovery and permission checks should be deterministic.

**Inputs:**

- Parsed question intent and requested measures/dimensions.
- Authorized project, dataset, and table scope.
- Cached metadata snapshot identifier, if available.

**Responsibilities:**

1. Discover relevant tables and views using approved metadata.
2. Read column names, types, descriptions, partitioning, clustering, and relevant table options.
3. Map business terms to physical columns and governed metrics.
4. Retrieve table grain, primary/business keys, join relationships, and known cardinalities from a curated metadata registry.
5. Identify mandatory partition filters, sensitivity constraints, and data-access restrictions.
6. Return a concise `SchemaContext` for the planner, SQL Generation Agent, and validators.

**Tools it may call:**

- BigQuery `INFORMATION_SCHEMA` queries.
- Metadata registry lookup.
- Schema snapshot cache.
- Join-relationship and metric-definition resolver.

**Output contract — `SchemaContext`:**

- Approved tables and views.
- Approved columns and data types.
- Business-to-physical field mappings.
- Table grain and unique-key assumptions.
- Permitted join paths and cardinality.
- Mandatory date/partition filters.
- Sensitive-field labels and policy constraints.
- Metadata version and freshness timestamp.

**Failure behavior:** If the business concept maps to multiple plausible columns or no approved join path exists, return an ambiguity or schema-gap result. Do not invent a column, table, metric definition, or join.

**Important:** `INFORMATION_SCHEMA` is useful for physical metadata, but it does not replace business metadata. Definitions such as “claim amount,” “eligible population,” and “sponsor” must be curated and governed separately. BigQuery documents `INFORMATION_SCHEMA` as its read-only metadata interface: [Introduction to INFORMATION_SCHEMA](https://docs.cloud.google.com/bigquery/docs/information-schema-intro).

### 4.2 A2 — Exact Match Agent

**Purpose:** Find records that satisfy exact identifiers, exact phrases, known codes, or deterministic business rules.

**Typical implementation:** Deterministic Python/BigQuery execution with optional LLM assistance only for mapping the user’s words to known fields or approved terms.

**Inputs:**

- Exact identifiers, phrases, codes, or rules extracted from the question.
- Approved text columns and structured columns from `SchemaContext`.
- Structured prefilters such as date range, sponsor, or status.

**Responsibilities:**

1. Normalize safe comparison forms (case folding, whitespace normalization, Unicode normalization where appropriate).
2. Execute exact equality, exact phrase containment, identifier lookup, or governed rule matching.
3. Preserve source table, source row key, field name, and matched span where appropriate.
4. Distinguish an exact phrase match from a tokenized or approximate match.

**Tools it may call:**

- Parameterized BigQuery SQL using `STRPOS`, `CONTAINS_SUBSTR`, equality, or governed lookup tables.
- Exact identifier lookup service.

**Output contract — `ExactMatchSet`:**

- Stable source row references.
- Match type and matched field.
- Exact phrase/identifier that matched.
- Structured filters applied.
- Query and source lineage.

**Failure behavior:** If the requested phrase is not unambiguous (for example, whether punctuation and plural variations count), the agent must preserve the strict interpretation or request clarification. It must not silently expand an exact match into fuzzy matching.

### 4.3 A3 — Lexical Search Agent

**Purpose:** Find records containing relevant words or token patterns, including known synonyms and phrase variants.

**Typical implementation:** Deterministic retrieval agent using BigQuery `SEARCH`, optionally assisted by constrained query expansion.

**Inputs:**

- Search terms and lexical variants.
- Approved narrative columns.
- Structured prefilters and search-index metadata.

**Responsibilities:**

1. Search only the narrative columns relevant to the question.
2. Use BigQuery `SEARCH` for tokenized lexical matching where appropriate.
3. Use exact phrase logic when phrase semantics are required; `SEARCH` tokenization must not be mistaken for exact phrase matching.
4. Return record IDs, matched columns, lexical match features, and provenance.
5. Report index usage and search status where available.

**Tools it may call:**

- BigQuery `SEARCH` function.
- Search-index status checks.
- Deterministic token/phrase scoring.

**Output contract — `LexicalEvidenceSet`:**

- Candidate row IDs.
- Matched term(s) and fields.
- Lexical score/features.
- Source lineage and structured filters.
- Search execution metadata.

**Important:** BigQuery `SEARCH` is tokenized search; it is not BM25 by default. Do not label it BM25. Scope the search to relevant columns and test tokenization behavior for the actual narrative data. See [BigQuery Search](https://docs.cloud.google.com/bigquery/docs/search) and [Search functions](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/search_functions).

### 4.4 A4 — Semantic Retrieval Agent

**Purpose:** Find records whose narrative meaning is similar to the user’s concept, even when the exact words are not present.

**Typical implementation:** Embedding model call plus BigQuery vector search. This is primarily a retrieval operation; it does not decide final semantic membership for an exact aggregate.

**Inputs:**

- Natural-language search concept.
- Approved narrative fields or preprocessed chunks.
- Embedding model identifier and dimension.
- Structured filters and eligible date ranges.
- Requested retrieval depth and cost/latency budget.

**Responsibilities:**

1. Create a query embedding using the same compatible embedding model and preprocessing configuration used for the corpus.
2. Search the relevant vector column using BigQuery `VECTOR_SEARCH`.
3. Return the nearest candidate paragraphs/records with distance, source IDs, and lineage.
4. Apply safe structured filters and preserve whether the result used approximate or brute-force search.
5. Report index usage, freshness, and any known constraints.

**Tools it may call:**

- Vertex AI embedding API.
- BigQuery `VECTOR_SEARCH`.
- Vector-index metadata/status query.

**Output contract — `SemanticEvidenceSet`:**

- Candidate record and paragraph IDs.
- Similarity/distance score.
- Embedding model and version.
- Vector index and search mode.
- Structured filters applied.
- Source lineage and index freshness metadata.

**Embedding recommendation:** Benchmark `gemini-embedding-2` and `gemini-embedding-001` against the actual narrative corpus. Current Google documentation describes `gemini-embedding-2` as suited to long and multimodal inputs, while `gemini-embedding-001` is a text embedding option with up to 3,072 dimensions. Select the model based on measured retrieval quality, latency, cost, region availability, and compatibility—not model name alone. Record model ID and embedding dimension in every indexed row. See [Get text embeddings](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/get-text-embeddings).

**Important:** BigQuery vector indexes can enable approximate nearest-neighbor search, trading some recall for performance. Use brute-force search as a benchmark or where exact nearest-neighbor behavior is required and cost is acceptable. `VECTOR_SEARCH` results are candidates, not proof of complete population coverage. See [Manage vector indexes](https://docs.cloud.google.com/bigquery/docs/vector-index) and [Search functions](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/search_functions).

### 4.5 A5 — Evidence Fusion Agent

**Purpose:** Merge outputs from exact, lexical, and semantic retrieval into a unified, traceable evidence set.

**Typical implementation:** Deterministic Python agent/node. An LLM is not needed to combine ranking lists or deduplicate record IDs.

**Inputs:**

- `ExactMatchSet` (optional).
- `LexicalEvidenceSet` (optional).
- `SemanticEvidenceSet` (optional).
- Ranking weights and deduplication rules.

**Responsibilities:**

1. Normalize evidence into a common record format.
2. Deduplicate by stable source row ID and paragraph/chunk ID.
3. Preserve each retrieval channel’s contribution and source lineage.
4. Combine ranked result lists using Reciprocal Rank Fusion (RRF) or an evaluated alternative.
5. Preserve channel-specific scores rather than hiding them behind one score.
6. Return an ordered candidate set for semantic evaluation and analyst-facing evidence.

**Recommended initial fusion:** RRF with a starting constant `k = 60`, then tune `k` and optional channel weights against a labelled validation set. For document/record `d`:

\[
  RRF(d) = \sum_{r \in R} \frac{w_r}{k + rank_r(d)}
\]

where `R` is the set of retrieval channels, `w_r` is an optional channel weight, and `rank_r(d)` is the rank of `d` in channel `r`. The initial value is a baseline, not a universal optimum.

**Output contract — `UnifiedEvidenceSet`:**

- Deduplicated record references.
- Unified rank and channel-level ranks/scores.
- Evidence snippets/paragraph references.
- Retrieval provenance.
- Candidate-set size and retrieval configuration.

**Failure behavior:** If channels return incompatible identifiers or evidence lineage is missing, fail closed and report an evidence-integrity error. Do not fuse records that cannot be traced to a source row.

### 4.6 A6 — Semantic Evaluation Agent

**Purpose:** Determine whether a record’s narrative satisfies the explicit semantic condition expressed in the user’s question.

**Typical implementation:** Gemini-based classification agent with a strict Pydantic output schema, supported by a scalable batch evaluation service.

**Inputs:**

- A normalized semantic predicate.
- Narrative text for one record or a bounded batch.
- Relevant structured context (only fields needed for interpretation).
- Classification policy and calibrated thresholds.
- Text hash, predicate hash, model version, and prompt version.

**Responsibilities:**

1. Convert the broad business concept into a testable predicate with inclusion and exclusion criteria.
2. Evaluate each input record against the predicate using only the supplied evidence.
3. Return one of `MATCH`, `NOT_MATCH`, or `UNCERTAIN`.
4. Provide a short evidence span or quote, concise rationale, and calibrated score where supported.
5. Treat narrative content as untrusted data, never as instructions.
6. Persist evaluation outcomes so identical source text and predicate versions do not need to be repeatedly evaluated.
7. Report population-coverage state separately from classification confidence/quality.

**Example predicate:**

- **Concept:** delayed approval.
- **Include:** the note describes an approval decision or authorization that remained pending beyond the expected process, was materially delayed, or was received only after repeated follow-up.
- **Exclude:** routine approval with no delay; a delay unrelated to approval; a statement that merely mentions approval without describing a delay.
- **Uncertain:** the text suggests a possible delay but does not establish it clearly.

The actual definition must be refined with business owners and representative labelled examples. The model should not invent policy definitions from the phrase alone.

**Output contract — `SemanticEvaluation`:**

- `predicate_id` and `predicate_version`.
- `source_table`, `source_row_id`, `narrative_field`, and `text_hash`.
- `label`: `MATCH | NOT_MATCH | UNCERTAIN`.
- Evidence span/snippet and rationale.
- Model ID, prompt version, and evaluation timestamp.
- Evaluation status and error information.

**Critical correctness rule — candidate retrieval versus population evaluation:**

- Retrieval finds likely candidates efficiently.
- Semantic evaluation decides whether each supplied record matches the predicate.
- Population completeness requires proving that every record in the eligible population has been evaluated (or covered by an approved deterministic rule).

For an exact aggregate, the evaluator must cover the full eligible population after structured filters are applied. A top-100, top-1,000, or nearest-neighbour result cannot satisfy this requirement on its own. If a question requires a population-wide semantic decision, the system must perform a bounded batch classification across the full eligible set, use a previously validated/cached predicate evaluation, or decline to present a population-exact aggregate.

**Scale strategy:**

- First apply safe structured filters (date, sponsor, status, tenant scope) to define the eligible population.
- Count that population in BigQuery.
- Reuse cached evaluations when the predicate hash, source text hash, model version, prompt version, and policy version match.
- Batch only the unseen eligible rows through Cloud Run Jobs or a bounded worker pool calling Vertex AI.
- Write results incrementally to a BigQuery evaluation table.
- Reconcile evaluated distinct row IDs against the eligible population count.
- Stop or request scope reduction when the estimated evaluation cost or latency exceeds the plan budget.

If full evaluation is too expensive or too slow for an interactive response, the system must return an explicitly labelled exploratory result or ask the user to narrow the scope. It must not silently call a top-K result complete.

### 4.7 A7 — SQL Generation Agent

**Purpose:** Translate the approved analytical intent into BigQuery GoogleSQL using only approved tables, columns, predicates, and join paths.

**Typical implementation:** Gemini agent with a typed SQL-generation contract, constrained schema context, and no direct permission to execute arbitrary SQL.

**Inputs:**

- Validated intent and metrics.
- `SchemaContext`.
- Structured filters and date boundaries.
- Validated semantic predicate and evaluation-table reference, if applicable.
- Join path and table-grain metadata.
- Required output columns and aggregation semantics.

**Responsibilities:**

1. Generate a single read-only `SELECT` query (CTEs are allowed if policy permits).
2. Use parameterized values for user-provided values.
3. Use only approved identifiers from the schema context.
4. Apply mandatory partition filters.
5. Preserve the intended table grain and avoid join fanout.
6. Join semantic evaluation data using stable row keys and predicate/version identifiers.
7. Include explicit metrics, dimensions, and required diagnostic counts where appropriate.
8. Return the SQL, named parameters, and a concise explanation of the intended calculation.

**Important implementation boundary:** The SQL Generation Agent proposes SQL. It does not decide whether that SQL is safe to run. A deterministic SQL policy/validation stage must inspect it before BigQuery execution.

### 4.8 A8 — Result Verification Agent

**Purpose:** Check whether the executed query result is structurally consistent with the plan and evidence, and whether it is safe to present.

**Typical implementation:** Deterministic result checks first, with an optional LLM review for explanation-level consistency. The deterministic checks are authoritative for mechanical invariants.

**Inputs:**

- Execution plan.
- Validated SQL and query job metadata.
- Result schema and rows.
- Population-coverage report.
- Evidence and semantic evaluation references.

**Responsibilities:**

1. Check expected output columns and data types.
2. Check row counts, null rates, duplicate groups, and impossible values where domain rules exist.
3. Check whether the aggregation and grouping match the requested metric and dimensions.
4. Check for possible join-fanout symptoms using row/grain diagnostics.
5. Confirm that the semantic predicate version used by the query matches the validated evaluation version.
6. Confirm that coverage and quality gates passed for the population being reported.
7. Identify empty results, incomplete data, anomalies, or unresolved uncertainty.

**Output contract — `VerificationReport`:**

- `PASS | WARN | FAIL`.
- Checks executed and outcomes.
- Coverage and classifier-quality references.
- Anomaly descriptions.
- Whether bounded repair/re-planning is allowed.

**Failure behavior:** A failed verification must not be converted into a confident answer. The orchestrator may re-run a bounded repair path, request clarification, or abstain.

### 4.9 A9 — Answer Synthesis Agent

**Purpose:** Turn verified analytical results into a clear, traceable answer that directly responds to the user.

**Typical implementation:** Gemini agent with a strict input contract. It receives verified result data and evidence references; it does not run new queries or create new factual claims.

**Inputs:**

- Verified result rows.
- Original user question and normalized intent.
- SQL and query-job metadata.
- Evidence and coverage report.
- Known caveats, uncertainty, and policy-approved response style.

**Responsibilities:**

1. Lead with the answer and relevant units/time period.
2. Explain the calculation and relevant filters.
3. State whether the result is a structured aggregate, exact text match, or model-classified semantic cohort.
4. Include meaningful caveats and counts of uncertain/unclassified rows when relevant.
5. Provide the SQL or a reference to it where the product requires transparency.
6. Cite source tables, columns, and evaluation/predicate versions through internal lineage references.
7. Avoid implying that model classification is infallible.

**Failure behavior:** If required evidence or verification metadata is missing, the agent must report that it cannot produce a verified answer rather than filling the gaps with plausible language.

---

## 5. Query Planner Agent

### 5.1 Role

The Query Planner Agent is the primary reasoning component. It interprets the business question, decomposes it into structured operations, and decides which specialist agents are necessary.

It must not execute SQL, call BigQuery directly, or treat its own explanation as proof that a task is complete. It produces a typed plan that the orchestrator can validate and run.

### 5.2 Planning responsibilities

1. Determine the user’s intent and expected answer type.
2. Identify measures, dimensions, date filters, structured conditions, narrative concepts, and comparison requirements.
3. Determine whether the question needs exact, lexical, semantic, or hybrid retrieval.
4. Determine whether semantic evaluation must cover a full eligible population.
5. Identify relevant schema and relationships that must be resolved by A1.
6. Specify dependencies and which tasks may run in parallel.
7. Declare budget, timeout, and maximum retry/repair limits.
8. Define what evidence and checks are required before answering.
9. Identify ambiguity that requires clarification instead of guessing.

### 5.3 Typed execution plan

The plan should be represented by Pydantic v2 models and validated before execution. The example below is conceptual; exact field names can be refined during implementation.

```python
from typing import Literal
from pydantic import BaseModel, Field

class AgentTask(BaseModel):
    task_id: str
    agent_name: str
    purpose: str
    depends_on: list[str] = Field(default_factory=list)
    required: bool = True
    parallel_group: str | None = None

class ExecutionBudget(BaseModel):
    max_agent_calls: int = 20
    max_replans: int = 1
    max_sql_repairs: int = 2
    max_execution_seconds: int = 90
    max_bytes_billed: int | None = None
    max_semantic_rows: int | None = None
    max_estimated_model_cost: float | None = None

class ExecutionPlan(BaseModel):
    plan_id: str
    intent: str
    answer_type: Literal[
        "structured_aggregate",
        "exact_text_aggregate",
        "lexical_text_aggregate",
        "semantic_aggregate",
        "hybrid_aggregate",
        "record_lookup",
        "comparison",
    ]
    measures: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    structured_filters: list[dict] = Field(default_factory=list)
    narrative_predicates: list[dict] = Field(default_factory=list)
    required_agents: list[AgentTask]
    requires_full_population_evaluation: bool = False
    requires_clarification: bool = False
    clarification_question: str | None = None
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    stop_conditions: list[str] = Field(default_factory=list)
```

The model output must be validated in application code. If validation fails, the system may request one bounded structured re-generation. It must not accept arbitrary agent names, table identifiers, SQL fragments, or tool arguments merely because they appear in model output.

### 5.4 Agent selection rules

The planner should use explicit routing rules and examples in addition to model reasoning.

| User question pattern | Likely specialist route |
|---|---|
| Numeric metric with known dimensions and no narrative condition | A1 → A7 → SQL policy gate → A8 → A9 |
| Exact phrase or identifier | A1 → A2 → A7 → SQL policy gate → A8 → A9 |
| Keyword or phrase variants | A1 → A3 → A5 → A7 → SQL policy gate → A8 → A9 |
| Meaning-based narrative concept | A1 → A4 → A5 → A6 → coverage/quality gate → A7 → SQL policy gate → A8 → A9 |
| Mixed lexical + semantic condition | A1 → A2/A3/A4 in parallel as needed → A5 → A6 → coverage/quality gate → A7 → SQL policy gate → A8 → A9 |
| Multi-table narrative aggregation | A1 with relationship resolution → relevant retrieval agents → A5/A6 as needed → A7 → SQL policy gate → A8 → A9 |

This is a starting routing policy. The planner’s output should be evaluated against a labelled set of real questions and adjusted as failure patterns become clear.

---

## 6. Orchestrator Agent

### 6.1 Role

The Orchestrator Agent is the execution control plane. It reads the validated `ExecutionPlan`, invokes the specified specialist agents, passes structured outputs between them, and decides whether the workflow may continue.

The orchestrator should not independently reinterpret the business question after every tool call. Its normal behavior is to execute the planner’s validated dependency graph. It may request a bounded re-plan only when a defined condition is triggered, such as missing metadata, retrieval failure, a failed validation gate, or a result inconsistency.

### 6.2 Recommended implementation

Use ADK’s current graph-based workflow mechanism for the execution graph, with Python nodes for deterministic steps and ADK agents for LLM reasoning steps. Invoke specialist agents through explicit agent-as-tool calls or a controlled dispatch adapter. Keep the orchestration policy in application code rather than relying on an unconstrained LLM to decide every next action.

This gives the system a useful hybrid:

- **Model-driven planning** where language understanding is required.
- **Deterministic workflow control** where sequencing, budgets, validation, and retries are required.
- **Specialist agent reasoning** only inside narrow tasks where it adds value.
- **Deterministic tools and services** for data access, ranking, SQL validation, and computation.

ADK’s current workflow documentation describes graph nodes as functions, agents, tools, or nested workflows, with dependency-driven scheduling and support for parallel branches. Verify the precise APIs against the project’s pinned ADK version: [ADK Workflow](https://github.com/google/adk-python/blob/main/docs/guides/workflow/workflow/index.md). The current ADK patterns also describe explicit agent invocation through an agent-tool interface: [ADK workflow patterns](https://github.com/google/adk-docs/blob/main/docs/workflows/patterns.md).

### 6.3 Orchestrator responsibilities

- Validate the plan and reject unknown agent names or disallowed operations.
- Build a dependency graph from `AgentTask` entries.
- Invoke independent retrieval tasks concurrently.
- Ensure each agent receives only the minimum context it needs.
- Store intermediate results by invocation/task ID rather than copying large text payloads between prompts.
- Maintain state, deadlines, cancellation, retry counts, and budget consumption.
- Enforce the completeness/quality gate before SQL generation for semantic aggregates.
- Enforce the SQL policy gate before query execution.
- Enforce result verification before answer synthesis.
- Return a structured error or safe abstention when a required stage cannot complete.

### 6.4 Bounded loops

| Failure condition | Allowed response | Suggested bound |
|---|---|---|
| Plan schema invalid | Ask planner for a corrected structured plan | 1 re-generation |
| Metadata ambiguous | Ask for clarification or return schema ambiguity | No silent guessing |
| Retrieval too narrow / failed | Try a bounded alternative query expansion or another retrieval channel | 1–2 attempts per channel |
| Semantic population incomplete | Continue remaining eligible batch rows, narrow scope, or abstain | Bounded by row, cost, and time budgets |
| SQL AST/policy failure | Give structured validator errors to SQL Generation Agent for repair | At most 2 repairs |
| BigQuery transient failure | Retry transient transport/service errors with exponential backoff and jitter | Small fixed retry count |
| Result verification failure | Re-plan or repair only for a classified cause | At most 1 re-plan |

A retry must not repeat a completed expensive task when its output can be safely reused. Agent execution records should be idempotent where practical.

---

## 7. End-to-end example: delayed-approval claim analysis

### 7.1 User question

> “What is the average claim amount by sponsor for cases whose notes describe delayed approvals during the last quarter?”

The planner decomposes the question into four parts:

| Component | Interpretation |
|---|---|
| Measure | Average claim amount |
| Grouping dimension | Sponsor |
| Structured time filter | Last quarter, resolved using the agreed business date field and timezone |
| Narrative condition | Notes describe delayed approval |

The phrase “delayed approval” is a semantic condition. A record may describe an approval delay using different words, so exact string matching alone is not sufficient.

### 7.2 Planned route

1. **A1 — Schema & Metadata:** Resolve the claim amount, sponsor, date field, narrative field, source table grain, and any approved joins. Confirm which date defines the quarter.
2. **A3 — Lexical Search:** Search for known terms such as “delayed approval,” “pending approval,” and “authorization delay.”
3. **A4 — Semantic Retrieval:** Find notes describing the same concept with different wording.
4. **A5 — Evidence Fusion:** Deduplicate and rank candidates while preserving lexical and semantic provenance.
5. **A6 — Semantic Evaluation:** Define the explicit delayed-approval predicate and evaluate the complete eligible population for the requested date range, reusing cached evaluations where valid.
6. **Completeness and quality gate:** Confirm all eligible rows were evaluated and the predicate meets task-specific validation criteria. Unresolved `UNCERTAIN` rows must be surfaced and handled under the policy; they must not disappear silently.
7. **A7 — SQL Generation:** Generate the sponsor-level average using the validated evaluation set and approved source tables.
8. **SQL policy gate:** Parse and inspect the query, check the table/column allowlist, joins, grain, parameters, and read-only restriction; then perform a BigQuery dry run.
9. **BigQuery execution:** Execute the query within the configured bytes, time, and result limits.
10. **A8 — Result Verification:** Check output structure, aggregation semantics, row grain, coverage references, and anomalies.
11. **A9 — Answer Synthesis:** Return the result, SQL, scope, methodology, and relevant caveats.

### 7.3 What happens if the candidate search returns only 2,000 rows?

The system must not assume the 2,000 rows are the entire population. It must count the full structured eligible population and establish whether every eligible record has a valid semantic evaluation.

If there are 80,000 eligible records and only 2,000 have been evaluated, population coverage is 2.5%. That is not sufficient for a population-level average. The orchestrator must:

- continue a bounded batch evaluation over the remaining eligible records if the plan budget permits;
- reuse an approved, version-matched evaluation cache where possible;
- ask the user to narrow the time period or other filters if the job exceeds budget; or
- return only a clearly labelled exploratory candidate-set analysis or abstain.

It must not run `AVG(claim_amount)` over the 2,000 candidate records and describe it as the average for all delayed-approval cases.

### 7.4 Example response contract

A verified response should contain:

- **Answer:** average claim amount by sponsor, with units and date range.
- **Population definition:** what “delayed approval” meant for this run.
- **Coverage:** eligible row count, evaluated row count, match count, non-match count, and uncertain count.
- **Quality:** relevant validation metrics or quality status for the predicate/model version.
- **Method:** retrieval channels used and how the semantic predicate was evaluated.
- **SQL:** the validated query or a query reference.
- **Caveats:** any remaining uncertainty, source freshness, exclusions, or limits.

---

## 8. Retrieval and evidence architecture

### 8.1 Three retrieval modes

DH-RANL2SQL uses three distinct retrieval modes because they solve different problems.

#### Exact retrieval

Use for IDs, codes, exact phrases, known labels, or deterministic rules. This is the strongest option when the user explicitly asks for an exact match.

#### Lexical retrieval

Use BigQuery `SEARCH` for token-based matches over selected text columns. Lexical retrieval is useful when the concept is expressed with familiar words or known variants. It is not a semantic classifier and should not be described as BM25 unless a separate BM25 implementation is actually deployed.

#### Semantic retrieval

Use text embeddings and BigQuery `VECTOR_SEARCH` to find conceptually similar text. Semantic retrieval improves discovery where wording differs, but it is a candidate-generation method. Approximate vector indexes can trade recall for speed.

### 8.2 Hybrid retrieval and fusion

When both lexical and semantic signals matter, the planner can dispatch A3 and A4 concurrently. A5 then fuses their outputs using RRF or another evaluated ranking method.

Keep the first production implementation modular:

- BigQuery `SEARCH` for lexical search.
- Vertex AI embeddings for query vectors and offline document vectors.
- BigQuery `VECTOR_SEARCH` for semantic candidates.
- Python fusion logic for rank combination and deduplication.

BigQuery documentation currently identifies integrated hybrid search functionality as Preview in the search-function reference. Do not make a preview feature a critical dependency for the first production release. Keep a modular fallback until the chosen feature has an acceptable production status and has passed the workload’s correctness and performance tests: [BigQuery Search functions](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/search_functions).

### 8.3 Evidence representation

Every evidence item should have a stable lineage structure. A suggested record shape is:

```json
{
  "source_table": "project.dataset.claims",
  "source_row_id": "claim-12345",
  "narrative_column": "case_notes",
  "chunk_id": "claim-12345:case_notes:0",
  "text_hash": "sha256:...",
  "retrieval_channels": ["lexical", "semantic"],
  "lexical_rank": 4,
  "semantic_rank": 8,
  "rrf_score": 0.0315,
  "evidence_reference": "evidence://..."
}
```

Do not rely on row order, temporary BigQuery result position, or non-unique business labels as evidence identifiers.

### 8.4 Chunking strategy

Start with one embedding unit per meaningful narrative field or paragraph, not arbitrary fixed-size fragments that destroy context. For 2–5 paragraph notes, benchmark:

- whole-field embedding;
- paragraph-level embedding with source-row aggregation;
- paragraph groups where a single paragraph is too short or context dependent.

Store `source_row_id`, `narrative_column`, `chunk_id`, offsets or paragraph numbers, and a content hash. When several chunks from one row match, aggregate at row level without multiplying the row in later SQL.

---

## 9. Offline indexing and knowledge preparation

### 9.1 Purpose

The offline plane prepares metadata and narrative text so that runtime queries do not need to rediscover and re-embed the entire corpus on every request.

### 9.2 Pipeline

1. **Discover:** Read the approved source-table registry and BigQuery metadata.
2. **Extract:** Read only approved narrative fields and required source identifiers.
3. **Normalize:** Apply versioned text normalization; preserve original text and lineage.
4. **Chunk:** Build field/paragraph units according to the selected strategy.
5. **Hash:** Generate a stable content hash per text unit and source row.
6. **Embed:** Call the selected Vertex AI embedding model for changed text only.
7. **Load:** Write chunk text, source metadata, model ID, vector dimension, hash, and embedding into a BigQuery evidence table.
8. **Index:** Maintain BigQuery search indexes and vector indexes as appropriate.
9. **Reconcile:** Detect changed rows, deleted rows, stale embeddings, duplicate keys, and missing source links.
10. **Publish:** Mark the index version/freshness watermark available to the runtime.

### 9.3 Suggested BigQuery objects

Names are illustrative and should be adapted to the enterprise naming standard.

| Object | Purpose |
|---|---|
| `governed_table_registry` | Approved source tables, owners, grain, descriptions, required filters, sensitivity |
| `business_metric_registry` | Metric definitions and valid aggregation behavior |
| `table_relationship_registry` | Approved joins, keys, and cardinality |
| `narrative_chunks` | Normalized narrative units and source lineage |
| `narrative_embeddings` or embedding column in `narrative_chunks` | Embedding vectors and model metadata |
| `semantic_predicate_registry` | Versioned predicate definitions and ownership |
| `semantic_evaluations` | Row-level classification outcomes keyed by predicate and source text version |
| `query_execution_audit` | Plan, agent, SQL, cost, status, and verification metadata |
| `index_refresh_audit` | Source watermark, counts, model version, and index status |

### 9.4 Incremental refresh

The indexing job should be idempotent:

- Re-embed only rows whose normalized text hash changed or whose embedding model/version changed.
- Maintain a source high-water mark or ingestion timestamp when reliable.
- Reconcile deletions and soft deletions.
- Avoid duplicate chunks for the same source row, narrative field, and text version.
- Record partial failures and resume from the last successful checkpoint.
- Publish a new index version only when validation checks pass.

For scheduled refreshes, Cloud Scheduler can trigger Cloud Run services/jobs; Cloud Run Jobs are designed for run-to-completion workloads. See [Execute Cloud Run Jobs](https://docs.cloud.google.com/run/docs/execute/jobs) and [Run Cloud Run jobs on a schedule](https://docs.cloud.google.com/run/docs/execute/jobs-on-schedule).

### 9.5 Freshness policy

Expose a freshness watermark to the runtime. A user-facing response should be able to state when the data and indexes were last refreshed. The runtime must not assume that a vector index is current merely because the source table is current.

---

## 10. Semantic completeness and quality gate

This is the most important correctness control in DH-RANL2SQL.

### 10.1 Define the eligible population first

Apply all deterministic, authorized structured filters before semantic evaluation. For example:

- tenant or user-access scope;
- date range;
- sponsor, region, status, or plan filters;
- approved source tables and join scope.

Then obtain the count of distinct eligible source records. This is the denominator for population coverage.

### 10.2 Track coverage explicitly

For a predicate evaluation run, track:

- `eligible_row_count`;
- `evaluated_distinct_row_count`;
- `match_count`;
- `not_match_count`;
- `uncertain_count`;
- `error_count`;
- duplicate/missing row IDs;
- source data watermark;
- predicate hash and version;
- model ID, prompt version, and classifier version.

A simple coverage calculation is:

\[
  coverage = \frac{evaluated\_distinct\_row\_count}{eligible\_row\_count}
\]

Coverage reaching 1.0 is necessary for exhaustive evaluation, but not sufficient for a trustworthy semantic result. Classification quality must also pass.

### 10.3 Track classification quality separately

Measure classifier performance using human-labelled data, including:

- precision and recall;
- F1 or task-specific weighted score;
- false-positive and false-negative review;
- performance by source table, sponsor, narrative length, and important subgroups;
- calibration of confidence/uncertainty where confidence is used;
- stability across model and prompt versions.

Do not rely on an LLM’s self-reported confidence as the sole quality measure. Thresholds must be chosen using validation data and approved by the business owner.

### 10.4 Gate decision

The gate should return one of three states:

| Gate state | Meaning | Allowed behavior |
|---|---|---|
| `PASS` | Eligible population is covered, required records are resolved, and predicate quality meets policy. | Population-level aggregation may proceed. |
| `PARTIAL` | Some records or classifications remain incomplete, or only retrieval candidates were evaluated. | Exploratory result only, narrow scope, continue batch evaluation, or ask for clarification. |
| `FAIL` | Coverage cannot be established, quality is below policy, or lineage/integrity is broken. | No population-level aggregate. Repair, clarify, or abstain. |

Do not silently exclude uncertain records from the denominator and still describe the result as the average across all qualifying records. The response must either resolve uncertainty under the approved policy or show its impact explicitly. For example, the system may show a lower/upper bound or a separate uncertain count, but only if the calculation is methodologically defined and clearly labelled.

### 10.5 Evaluation cache key

A reusable semantic evaluation should be keyed on at least:

- source table and stable row ID;
- source text hash;
- predicate definition hash/version;
- model ID/version;
- prompt version;
- classification policy version.

If any of these changes, the prior evaluation may not be reusable without an explicit compatibility rule.

---

## 11. SQL generation, validation, and BigQuery execution

### 11.1 SQL generation boundary

The SQL Generation Agent should produce a proposed query and parameter map. It must not have an unrestricted execution path. The orchestrator should pass the proposal to the SQL policy gate.

### 11.2 SQLGlot AST validation

Use SQLGlot with the BigQuery dialect to parse the generated SQL and inspect the abstract syntax tree.

Required checks include:

1. Exactly one permitted query statement.
2. `SELECT`/CTE-only statement policy; reject DML, DDL, scripting, `CALL`, export, or other disallowed statements.
3. All tables and views exist in the approved allowlist.
4. All columns exist in the supplied schema context.
5. No unauthorized wildcard expansion or `SELECT *` where policy disallows it.
6. Join paths match the governed relationship registry.
7. Join cardinality and table grain do not create silent aggregation fanout.
8. Required partition filters are present.
9. User-provided values are parameterized.
10. Functions, subqueries, and UDFs conform to an allowlist.
11. The query returns only approved fields and an allowed result size.
12. Semantic evaluation joins constrain the expected predicate ID/version and source row keys.

SQLGlot is a SQL parser, transpiler, optimizer, and AST library, but its own documentation notes that parsing success is not equivalent to database validation. The BigQuery engine remains the final syntax/semantic execution authority after local policy checks. See [SQLGlot documentation](https://sqlglot.com/sqlglot).

### 11.3 Parameterization and identifiers

Use BigQuery query parameters for user-provided values (dates, labels, IDs, text constants). Parameters cannot substitute table or column identifiers, so identifiers must be selected from a strict allowlist and rendered only by trusted code. See [BigQuery parameterized queries](https://docs.cloud.google.com/bigquery/docs/parameterized-queries).

### 11.4 Dry run and cost controls

Before execution:

- Run a BigQuery dry run.
- Inspect estimated bytes processed.
- Enforce `maximum_bytes_billed` or equivalent query-job limits.
- Require a partition filter when the source table design expects one.
- Apply query timeout, maximum output rows, and maximum response payload size.
- Record the job ID and query hash.

BigQuery supports dry-run query configuration and reports estimated bytes processed; use this to reject queries above the plan budget before they execute. See [Run a query](https://docs.cloud.google.com/bigquery/docs/running-queries) and [Troubleshoot query issues](https://docs.cloud.google.com/bigquery/docs/troubleshoot-queries).

### 11.5 Read-only security

Use a dedicated runtime service account with the minimum BigQuery permissions needed to read approved datasets and create query jobs. It should not have permissions to alter production source tables, change IAM, or administer datasets. Use authorized views or approved views when they simplify access boundaries. Keep user access and row-level restrictions enforced by the platform/data layer, not by prompt instructions.

BigQuery recommends least-privilege IAM for data access. See [BigQuery access control](https://docs.cloud.google.com/bigquery/docs/access-control-intro). Row-level policies can further restrict visible records where appropriate: [BigQuery row-level security](https://docs.cloud.google.com/bigquery/docs/row-level-security-intro).

### 11.6 Join correctness

A query may be syntactically valid and still be analytically wrong. The SQL policy layer should know:

- the grain of each table;
- unique keys and possible duplicate keys;
- one-to-one, one-to-many, or many-to-many relationships;
- the intended aggregate grain;
- whether deduplication is required before aggregation.

Do not rely on the model to infer these properties from column names.

---

## 12. API and application contract

### 12.1 Recommended API surface

A minimal FastAPI interface can expose:

| Endpoint | Purpose |
|---|---|
| `POST /v1/query` | Submit a natural-language analytical question |
| `GET /v1/query/{query_id}` | Retrieve query status/result where execution is asynchronous |
| `GET /healthz` | Process health check |
| `GET /readyz` | Dependency/readiness check |

For small, bounded structured queries, the API may return synchronously. For full-population semantic evaluation, the system should support an asynchronous job model rather than keeping a request open for an unbounded period.

### 12.2 Request model

A suggested request shape:

```json
{
  "question": "Average claim amount by sponsor for cases whose notes describe delayed approvals last quarter",
  "response_mode": "explain",
  "requested_timezone": "Asia/Kolkata",
  "filters": {},
  "client_request_id": "optional-idempotency-key"
}
```

User identity, tenant scope, and permissions must come from authenticated server context—not from untrusted JSON fields supplied by the caller.

### 12.3 Response model

A suggested response shape:

```json
{
  "query_id": "q_...",
  "status": "completed",
  "answer": "...",
  "result": [],
  "sql": "SELECT ...",
  "method": {
    "answer_type": "semantic_aggregate",
    "retrieval_channels": ["lexical", "semantic"],
    "predicate_id": "pred_...",
    "population_coverage": 1.0,
    "quality_status": "passed"
  },
  "provenance": {
    "source_tables": ["project.dataset.claims"],
    "data_watermark": "...",
    "query_job_id": "..."
  },
  "caveats": []
}
```

The actual response contract should be versioned. Large result sets should be stored and referenced rather than embedded in the API response.

---

## 13. Recommended technology stack

### 13.1 Core application

| Layer | Recommended technology | Role |
|---|---|---|
| Language | Python 3.12+ (pin a supported runtime) | Main application and agent logic |
| API | FastAPI | Authenticated query API and job/status endpoints |
| Data contracts | Pydantic v2 | Typed plans, agent inputs/outputs, API requests/responses |
| Agent framework | Google ADK for Python | Agent definitions, session/context management, agent invocation, workflow composition |
| Workflow control | ADK graph-based Workflow plus Python orchestration nodes | Deterministic dependencies, parallel execution, bounded loops |
| LLM | Gemini models on Vertex AI | Planning, semantic interpretation/classification, SQL generation, answer synthesis |
| Embeddings | Vertex AI text embedding model | Query and narrative embeddings |
| SQL parsing/policy | SQLGlot with BigQuery dialect | AST inspection and query policy enforcement |
| BigQuery client | `google-cloud-bigquery` | Metadata access, search, query jobs, dry runs, result retrieval |
| Configuration | Pydantic Settings / environment configuration | Typed runtime configuration |
| Tests | `pytest`, async test support, contract/evaluation tests | Unit, integration, and regression checks |

Use `uv` or another locked dependency workflow to create a reproducible environment. Pin ADK, Pydantic, SQLGlot, Google Cloud SDKs, and model settings in a lockfile. Do not copy version numbers from an example into production without checking compatibility and support status.

### 13.2 GCP services

| GCP service | Use |
|---|---|
| Cloud Run Service | Host FastAPI and the ADK runtime |
| Cloud Run Jobs | Run large semantic-evaluation batches and offline indexing tasks |
| Cloud Scheduler | Trigger scheduled index refresh, reconciliation, or maintenance jobs |
| BigQuery | Source data, metadata, narrative chunks, embeddings, indexes, evaluation results, audit data, and analytical SQL |
| Vertex AI | Gemini inference and embedding generation |
| Cloud Storage | Optional storage for larger artifacts, evaluation datasets, and long-running job outputs |
| Secret Manager | Store secrets and sensitive runtime configuration |
| Cloud Logging | Structured application, agent, and job logs |
| Cloud Monitoring | Metrics, dashboards, and alerting |
| Cloud Trace / OpenTelemetry | End-to-end request and agent/tool spans |
| Artifact Registry | Container image storage |
| Cloud Build or approved CI/CD | Build, test, scan, and deploy containers |
| Terraform | Reproducible infrastructure and IAM configuration |

Cloud Run supports run-to-completion jobs and Scheduler-triggered executions, making it a suitable fit for offline indexing and bounded batch evaluation. See [Cloud Run Jobs](https://docs.cloud.google.com/run/docs/execute/jobs) and [Cloud Scheduler with Cloud Run](https://docs.cloud.google.com/run/docs/execute/jobs-on-schedule).

### 13.3 Recommended model strategy

Do not use one large model for every task.

- **Planner:** a capable Gemini model with structured output; prefer predictable schema adherence and strong intent decomposition.
- **Semantic evaluator:** benchmark a cost-effective Gemini model on a labelled classification set. Escalate uncertain/high-impact cases only where the policy permits.
- **SQL generator:** a model strong at SQL and constrained schema grounding; generation is followed by deterministic validation.
- **Answer synthesis:** a fast model is usually sufficient because it receives verified results and should not perform new analysis.
- **Embedding:** benchmark current supported Vertex AI embedding models for the actual text length and domain language.

All model IDs, region, prompt version, temperature, output schema, and safety configuration must be recorded per invocation. Model aliases such as “latest” should be used only if the associated deployment process explicitly handles model behavior changes and regression evaluation.

### 13.4 Components that are optional initially

- Memorystore/Redis for cache or distributed rate limiting, only if measured concurrency requires it.
- Pub/Sub for near-real-time source-change events, only if scheduled refresh is insufficient.
- Dataflow for large-scale streaming or transformation, only if Cloud Run Jobs/BigQuery SQL are no longer suitable.
- Separate agent microservices, only if scaling, deployment ownership, or isolation requires them. Start with a modular monolith on Cloud Run to reduce operational complexity.

---

## 14. Suggested application structure

A maintainable Python repository could look like this:

```text
dh-ranl2sql/
├── pyproject.toml
├── uv.lock
├── README.md
├── .env.example
├── app/
│   ├── main.py                    # FastAPI app and lifespan
│   ├── api/
│   │   ├── routes_query.py
│   │   ├── routes_health.py
│   │   └── schemas.py
│   ├── core/
│   │   ├── config.py
│   │   ├── auth.py
│   │   ├── logging.py
│   │   ├── errors.py
│   │   └── budgets.py
│   ├── agents/
│   │   ├── planner.py              # Query Planner Agent
│   │   ├── orchestrator.py         # Orchestrator / execution controller
│   │   ├── schema_metadata.py      # A1
│   │   ├── exact_match.py          # A2
│   │   ├── lexical_search.py       # A3
│   │   ├── semantic_retrieval.py   # A4
│   │   ├── evidence_fusion.py      # A5
│   │   ├── semantic_evaluation.py  # A6
│   │   ├── sql_generation.py       # A7
│   │   ├── result_verification.py  # A8
│   │   └── answer_synthesis.py     # A9
│   ├── workflows/
│   │   ├── query_workflow.py
│   │   ├── semantic_eval_workflow.py
│   │   └── recovery_workflow.py
│   ├── contracts/
│   │   ├── plan.py
│   │   ├── schema_context.py
│   │   ├── evidence.py
│   │   ├── predicate.py
│   │   ├── coverage.py
│   │   ├── sql.py
│   │   └── answer.py
│   ├── tools/
│   │   ├── bigquery_metadata.py
│   │   ├── bigquery_search.py
│   │   ├── bigquery_vector_search.py
│   │   ├── embeddings.py
│   │   ├── sql_policy.py
│   │   ├── query_executor.py
│   │   └── evaluation_store.py
│   ├── services/
│   │   ├── metadata_registry.py
│   │   ├── evidence_store.py
│   │   ├── semantic_cache.py
│   │   ├── audit_store.py
│   │   └── job_service.py
│   └── prompts/
│       ├── planner.md
│       ├── semantic_evaluator.md
│       ├── sql_generator.md
│       └── answer_synthesizer.md
├── jobs/
│   ├── refresh_metadata.py
│   ├── refresh_narrative_index.py
│   ├── reconcile_indexes.py
│   └── evaluate_population.py
├── infra/
│   ├── terraform/
│   └── cloudbuild/
├── evals/
│   ├── questions.jsonl
│   ├── semantic_labels.jsonl
│   ├── sql_goldens/
│   └── metrics/
└── tests/
    ├── unit/
    ├── integration/
    ├── security/
    └── end_to_end/
```

This is a suggested separation of responsibilities, not a requirement to create a separate deployable service for every folder.

---

## 15. Security, privacy, and governance

### 15.1 Identity and access

- Authenticate each caller at the API boundary.
- Derive tenant, role, and data scope from verified identity claims or an approved authorization service.
- Pass only the authorized scope into the planner and metadata layer.
- Use separate service accounts for runtime query execution and offline indexing.
- Grant least privilege at the dataset/table level.
- Keep production source tables read-only to the interactive runtime.

### 15.2 Prompt injection and untrusted narrative text

Narrative data may contain instructions, adversarial content, or text that looks like system prompts. Treat all source text as untrusted evidence:

- Never allow narrative text to change system instructions or tool permissions.
- Provide narrative text as quoted/classification input, not as instructions.
- Do not give the Semantic Evaluation Agent arbitrary SQL or unrestricted tool access.
- Require a typed classification response.
- Limit the agent to the supplied record and predicate.
- Store the exact evidence span supporting a `MATCH` decision.

### 15.3 Data minimization

- Send only the narrative fields and structured context required for the predicate.
- Mask or remove sensitive fields not required for classification.
- Avoid logging raw narrative text or personally identifying data by default.
- Use stable record IDs and hashes in logs where possible.
- Apply retention policies to prompts, results, cached classifications, and audit events.

### 15.4 Secrets

Use Secret Manager for secrets. Prefer workload identity and Application Default Credentials over embedded service-account keys. Google recommends Secret Manager for sensitive information used by Cloud Run services: [Cloud Run secrets](https://docs.cloud.google.com/run/docs/configuring/services/secrets).

### 15.5 Auditability

For every query, record enough metadata to reconstruct the execution path:

- query ID and caller/tenant scope (privacy-safe identifier);
- planner model and plan version;
- selected agents and execution order;
- tool calls and timings;
- source tables and metadata snapshot;
- retrieval channels, query/index/model versions;
- predicate ID/version and coverage report;
- generated SQL hash and validation report;
- BigQuery job ID, bytes processed, and execution status;
- result verification and final response status.

Do not store unnecessary raw sensitive text in the audit trail.

---

## 16. Observability and operational monitoring

### 16.1 Trace structure

Use one distributed trace per user query. Suggested spans:

1. API request/authentication.
2. Query Planner Agent.
3. Plan validation.
4. A1 metadata lookup.
5. A2/A3/A4 retrieval calls (parallel where appropriate).
6. A5 evidence fusion.
7. A6 semantic evaluation batches.
8. Completeness/quality gate.
9. A7 SQL generation.
10. SQL AST/policy validation.
11. BigQuery dry run and query execution.
12. A8 result verification.
13. A9 answer synthesis.

### 16.2 Metrics

Track at least:

- end-to-end latency (p50, p95, p99);
- planner latency and plan-schema failure rate;
- agent invocation count and failure rate by agent;
- lexical and semantic retrieval result counts;
- retrieval overlap and fusion metrics;
- vector index usage and freshness;
- semantic evaluation rows processed, cache hit rate, uncertain/error rate;
- population coverage and classifier quality by predicate type;
- SQL validation rejection rate and repair count;
- BigQuery bytes processed, slot time where available, and query failures;
- answer verification failure and abstention rate;
- estimated model cost per query and per evaluated row.

### 16.3 Alerts

Create alerts for:

- index refresh failure or stale watermark;
- elevated BigQuery query errors or bytes processed;
- spikes in semantic evaluation failures or uncertainty;
- coverage gate failures above baseline;
- agent model errors, quota errors, or high latency;
- SQL policy rejection spikes;
- unexpected cost increases.

---

## 17. Evaluation and quality assurance

### 17.1 Build a benchmark dataset

Start with approximately **500–1,000 labelled questions** as an initial benchmark target, then expand based on observed failure modes. Include:

- structured aggregation questions;
- exact identifier and phrase questions;
- lexical variants and synonyms;
- semantic paraphrases;
- hybrid text-plus-numeric aggregation;
- multi-table joins;
- ambiguous business language;
- missing metadata and schema changes;
- large eligible populations that exceed interactive budgets;
- adversarial narrative text and prompt-injection attempts.

For each question, retain:

- expected intent and answer type;
- expected agent route;
- gold SQL or result where feasible;
- labelled semantic predicate and record-level labels for semantic cases;
- allowed answer behavior (answer, clarify, partial/exploratory, or abstain).

### 17.2 Evaluate components independently

| Component | Evaluation measures |
|---|---|
| Query Planner | Intent accuracy, agent-selection precision/recall, plan validity |
| Schema & Metadata | Correct table/column mapping, join-path accuracy, ambiguity detection |
| Exact Match | Precision/recall for exact match rules |
| Lexical Search | Recall@K, precision@K, query expansion quality |
| Semantic Retrieval | Recall@K, nDCG/MRR, per-topic recall |
| Evidence Fusion | Recall/precision improvement, deduplication correctness, provenance integrity |
| Semantic Evaluation | Precision, recall, F1, calibration, subgroup performance |
| SQL Generation | Execution accuracy, gold-result match, semantic metric correctness |
| SQL Policy Gate | Unauthorized query rejection, false rejection rate |
| Result Verification | Detection of known anomalies and incomplete evidence |
| Answer Synthesis | Faithfulness, completeness, citation/provenance alignment |
| End-to-end | Answer accuracy, safe abstention, latency, cost |

### 17.3 Regression process

Every change to the following should trigger the relevant regression suite:

- model ID or model version;
- embedding model or dimension;
- prompt or Pydantic output schema;
- metadata registry or join relationships;
- retrieval query logic, index configuration, or RRF parameters;
- semantic predicate policy or classifier thresholds;
- SQL generation prompt or SQL policy;
- BigQuery schema or source table grain.

Keep a frozen holdout set so that prompt/model tuning does not overfit the evaluation data.

### 17.4 Human review loop

Capture user corrections and reviewed classifications through an explicit feedback workflow. Do not silently turn a single user's correction into a global truth label. Curate reviewed examples, adjudicate disagreements, version the predicate definition, and rerun evaluations before promoting a change.

---

## 18. Performance and cost strategy

### 18.1 Avoid unnecessary agent calls

The planner should skip retrieval for structured-only questions and avoid semantic evaluation when an exact deterministic condition answers the question. The orchestrator should not call all nine specialists on every request.

### 18.2 Use structured filters early

Apply date, sponsor, status, region, and access-scope filters before text retrieval or semantic batch evaluation whenever this is logically valid. This reduces the eligible population and model cost.

### 18.3 Cache reusable work safely

Cache only versioned outputs with explicit invalidation keys:

- metadata cache: dataset/schema snapshot version;
- embeddings: normalized text hash + embedding model/version;
- semantic evaluation: text hash + predicate hash + model/prompt/policy version;
- query result cache: normalized query/plan fingerprint + source freshness + caller authorization scope + model/predicate versions.

Never reuse a cached answer across authorization scopes without rechecking permissions.

### 18.4 Control full-population classification cost

Full-population semantic evaluation is potentially the most expensive part of the architecture. Use:

- eligibility filters before classification;
- cached predicate decisions;
- batching and concurrency limits;
- a per-query row/model-cost budget;
- asynchronous execution for large populations;
- explicit user scope reduction when the budget is exceeded.

A vector search can reduce discovery time but cannot replace exhaustive semantic evaluation when the requested answer is a population-level aggregate and no completeness guarantee exists.

---

## 19. Deployment architecture

### 19.1 Interactive runtime

Deploy the FastAPI + ADK application as a Cloud Run service. The service should:

- authenticate and authorize callers;
- accept questions and create query IDs;
- run the planner and orchestrator for bounded work;
- submit long-running semantic evaluation tasks as Cloud Run Jobs or another approved async execution path;
- return status and verified results;
- emit traces and structured logs.

### 19.2 Offline workers

Deploy separate Cloud Run Jobs for:

- metadata refresh;
- text extraction/chunking and embedding;
- search/vector index maintenance;
- semantic evaluation batches;
- index and evaluation reconciliation;
- periodic benchmark/regression runs where appropriate.

### 19.3 Environments

Use separate development, test/staging, and production GCP projects or strongly isolated environments. Promote container images and configuration through CI/CD. Keep model IDs, limits, dataset names, and IAM bindings environment-specific.

### 19.4 Infrastructure as code

Use Terraform or the organization’s approved IaC platform for:

- Cloud Run services/jobs;
- service accounts and IAM bindings;
- BigQuery datasets, tables, and views;
- Secret Manager references;
- Scheduler triggers;
- logging/monitoring resources;
- Artifact Registry and deployment configuration.

---

## 20. Implementation roadmap

A phased rollout reduces risk and makes each layer measurable.

### Phase 0 — Data and metadata foundation

- Profile source tables, grains, keys, partitioning, text columns, and volumes.
- Create the governed table/metric/join registry.
- Confirm authentication, tenant scope, and read-only service-account design.
- Build labelled questions and a small human-reviewed semantic dataset.

**Exit criteria:** approved schema context can be reliably generated; source grain and key relationships are documented.

### Phase 1 — Structured NL2SQL baseline

- Implement FastAPI and Pydantic contracts.
- Implement Query Planner Agent and Orchestrator control flow.
- Implement A1 metadata and A7 SQL generation.
- Implement SQLGlot AST validation, BigQuery dry run, and read-only execution.
- Implement A8 verification and A9 answer synthesis.

**Exit criteria:** structured questions produce validated BigQuery results with reliable metric and join semantics.

### Phase 2 — Exact and lexical retrieval

- Implement A2 exact match and A3 lexical search.
- Create narrative chunk table and BigQuery search indexes where beneficial.
- Preserve stable row/paragraph lineage.
- Add lexical query evaluation and exact-match tests.

**Exit criteria:** exact and lexical retrieval are traceable and meet agreed recall/precision targets.

### Phase 3 — Semantic retrieval and evidence fusion

- Benchmark embedding models and chunking strategies.
- Build A4 vector retrieval and A5 RRF fusion.
- Add index refresh, freshness, and reconciliation jobs.
- Compare ANN and brute-force behavior on representative data.

**Exit criteria:** retrieval quality is measured; index lifecycle is operational; retrieval outputs remain candidate evidence only.

### Phase 4 — Semantic evaluation and completeness gate

- Implement predicate contracts and A6 classification.
- Build evaluation cache and batch evaluation workers.
- Establish full eligible population counting and row-level reconciliation.
- Calibrate classifier quality using human labels.
- Implement `PASS`, `PARTIAL`, and `FAIL` gate behavior.

**Exit criteria:** the system refuses to present candidate-only semantic aggregates as population-level results; full-population evaluation is bounded and auditable.

### Phase 5 — Production hardening

- Complete IAM and privacy review.
- Add load, concurrency, cost, and failure-injection tests.
- Add dashboards, alerts, audit records, and runbooks.
- Run end-to-end benchmark and regression gates.
- Deploy through CI/CD with rollback strategy.

**Exit criteria:** service-level targets, security controls, evaluation thresholds, and operational ownership are approved.

---

## 21. Key architecture decisions and non-goals

### 21.1 Decisions

1. **Planner-led routing:** the Query Planner Agent selects the specialist agents required for each question.
2. **Controlled orchestration:** the Orchestrator Agent executes a typed dependency plan; it does not run an unconstrained multi-agent conversation.
3. **Agent-as-tool invocation:** specialists are independently defined agents callable by the orchestrator. Their internal deterministic tools/services remain separate from the agent identity.
4. **Hybrid retrieval:** exact, lexical, and semantic retrieval are separate paths that can run independently and be fused.
5. **Deterministic evidence fusion:** use reproducible ranking/deduplication logic rather than asking an LLM to invent a combined ranking.
6. **Population completeness gate:** candidate retrieval is never treated as complete population evaluation.
7. **SQL policy gate:** SQLGlot and application policy inspect the AST before BigQuery dry run and execution.
8. **Read-only execution:** interactive SQL runs with least-privilege read-only data access.
9. **Offline indexing:** embeddings and indexes are refreshed independently from interactive query execution.
10. **Measured quality:** agent routing, retrieval, semantic classification, SQL, and answer synthesis are evaluated independently.

### 21.2 Non-goals for the first production release

- Arbitrary SQL execution or data modification.
- Treating top-K semantic retrieval as a complete dataset filter.
- A separate microservice for every specialist agent.
- An unconstrained autonomous agent swarm.
- Dependence on preview-only hybrid-search functionality.
- A guarantee that an LLM semantic classification is infallible.
- Automatic discovery of undocumented business rules without owner validation.
- A full knowledge graph unless metadata relationships prove insufficient for the supported workload.

---

## 22. Risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Top-K retrieval misses qualifying rows | Population aggregates become biased | Full-population evaluation or safe abstention |
| Semantic classifier errors | False positives/negatives change the cohort | Human-labelled evaluation, calibrated thresholds, uncertainty reporting |
| Schema drift | Generated SQL references stale fields | Metadata snapshots, refresh checks, schema-change alerts |
| Join fanout | Aggregates become inflated | Grain registry, cardinality-aware joins, AST policy, tests |
| Vector-index recall loss | Relevant records may be missed | Benchmark ANN against brute force; do not use it as completeness proof |
| Embedding model migration | Existing and new vectors become incompatible | Store model/version/dimension; re-embed and version indexes |
| Prompt injection in notes | Narrative text may manipulate model behavior | Treat text as untrusted evidence; constrained agent tools and structured outputs |
| Excessive model cost | Full semantic evaluation may be expensive | Scope filters, cache, batch limits, budget gates, async jobs |
| LLM-generated SQL is unsafe | Data exposure or expensive/incorrect queries | SQLGlot AST validation, allowlists, dry runs, IAM, bytes/time limits |
| Stale evidence index | Results do not reflect source updates | Freshness watermarks, incremental refresh, reconciliation |
| Ambiguous business definitions | The system applies the wrong concept | Curated glossary, predicate confirmation, clarification path |

---

## 23. Final design summary

DH-RANL2SQL is not simply an NL2SQL system with a vector database attached. It is a **planner-led, orchestrated analytical pipeline** that separates narrative discovery, semantic decision-making, and structured computation.

Its distinguishing design choices are:

- the planner selects the specialist agents instead of invoking every capability for every question;
- the orchestrator controls execution using a typed plan and bounded workflow;
- exact, lexical, and semantic retrieval are independently measurable and can be combined;
- semantic evidence remains distinct from population-level classification;
- completeness and classifier quality are explicit gates before semantic aggregation;
- SQL generation is separated from deterministic SQL safety validation;
- results are verified before natural-language answer synthesis;
- offline indexing, evaluation caching, observability, and cost controls are part of the architecture from the beginning.

The most important product behavior is not that the system always returns an answer. It is that the system knows when the evidence is sufficient to answer, when more work is required, and when it must not claim more than it has established.

---

## 24. Official technical references

The following references support the technology choices and implementation details. Product behavior and API surfaces should be rechecked against the versions pinned for the project before deployment.

1. [Google ADK — Workflow API](https://github.com/google/adk-python/blob/main/docs/guides/workflow/workflow/index.md)
2. [Google ADK — Multi-agent workflow patterns](https://adk.dev/workflows/patterns/)
3. [Google ADK — Agent invocation and workflow patterns](https://github.com/google/adk-docs/blob/main/docs/workflows/patterns.md)
4. [BigQuery — Introduction to INFORMATION_SCHEMA](https://docs.cloud.google.com/bigquery/docs/information-schema-intro)
5. [BigQuery — Search indexed data](https://docs.cloud.google.com/bigquery/docs/search)
6. [BigQuery — Search functions and VECTOR_SEARCH](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/search_functions)
7. [BigQuery — Manage vector indexes](https://docs.cloud.google.com/bigquery/docs/vector-index)
8. [Vertex AI — Get text embeddings](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/get-text-embeddings)
9. [BigQuery — Parameterized queries](https://docs.cloud.google.com/bigquery/docs/parameterized-queries)
10. [BigQuery — Run a query and dry run](https://docs.cloud.google.com/bigquery/docs/running-queries)
11. [BigQuery — Access control](https://docs.cloud.google.com/bigquery/docs/access-control-intro)
12. [BigQuery — Row-level security](https://docs.cloud.google.com/bigquery/docs/row-level-security-intro)
13. [SQLGlot — API and AST documentation](https://sqlglot.com/sqlglot)
14. [Cloud Run — Execute Jobs](https://docs.cloud.google.com/run/docs/execute/jobs)
15. [Cloud Run — Execute Jobs on a schedule](https://docs.cloud.google.com/run/docs/execute/jobs-on-schedule)
16. [Cloud Run — Configure secrets](https://docs.cloud.google.com/run/docs/configuring/services/secrets)

---

*End of document.*
