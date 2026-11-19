# Principal Architecture Review & Validation Prompt: Single-Call Modular Plugin Text-to-SQL Agent (BPLV + Volatile CTEs)

> **Instructions to Reviewer / AI Validator:**
> Act as a **Principal AI Systems Architect, Senior Data/Actuarial Engineer, and Application Security Reviewer**.
> Critically evaluate the real-world architecture, thought process, and Python implementation below (`google-genai`, `pydantic>=2.0`, and standard `re` — **strictly no `sqlglot` or external SQL parsers**).
>
> Do **not** rubber-stamp this design. Actively hunt for hidden SDK traps, Pydantic 2.0 validation pitfalls, regex SQL bypasses/false positives, real-world BPLV/CTE SQL grain bugs, and security risks. Then deliver a ranked audit and the **best possible hardened final solution**.

---

## 1. What I Am Trying to Build (Deep Domain & Architectural Context)

I am building an enterprise natural-language-to-SQL analytics agent for **healthcare / benefits administration data** that delivers accurate natural-language answers grounded in live SQL execution, along with full reasoning transparency—all inside **one single `google-genai` SDK call** rather than a slow multi-agent chain (`Router LLM -> SQL LLM -> Synthesizer LLM`).

### The 3-Source Data Topology (Why This Architecture Exists)

1. **`bplv` — Benefit Plan Level Value / Provision Info (Physical Table)**
   - **Traffic Share (~80% of user queries):** Users predominantly ask questions purely about benefit provisions—copays, coinsurance, deductibles, out-of-pocket maximums (MOOP), visit/day limits, prior authorization flags, coverage levels (`INDIVIDUAL` vs. `FAMILY`), and network tiers (`IN_NETWORK`, `OUT_OF_NETWORK`, `TIER_1`).
   - **Performance Requirement:** Because `bplv` is a physical table and covers the vast majority of queries, BPLV-only requests must execute **directly against `bplv` with zero CTE overhead and zero unnecessary joins**.
   - **Real-World Data Nuances:**
     - Rows often carry multi-dimensional benefit attributes (e.g., `benefit_category`, `service_type`, `provision_code`, `provision_name`, `network_tier`, `coverage_level`, `copay_amount`, `coinsurance_pct`, `deductible_amount`, `moop_amount`, `deductible_applies`, `limit_value`, `limit_unit`, `plan_year`).
     - Aggregations must respect benefit grain (e.g., never blindly sum `INDIVIDUAL` and `FAMILY` deductibles or mix `IN_NETWORK` and `OUT_OF_NETWORK` tiers without grouping or filtering).

2. **`plan_snapshot_cte` — Plan-Level & Plan Sponsor-Level Snapshot (Volatile Virtual CTE)**
   - **Traffic Share (~10% of user queries):** Users occasionally ask strictly plan-level or plan-sponsor-level questions (e.g., active plan counts by sponsor, funding arrangement `ASO` vs. `FULLY_INSURED`, product type `PPO`/`HMO`/`HDHP`, effective/renewal dates, control/group IDs, plan status).
   - **Why It Is a CTE (Not a Static Table):** Underlying plan and sponsor records change frequently (SCD Type 2 snapshots, mid-year amendments, effective-date cutoffs, renewal feeds). The data/engineering team maintains the latest snapshot resolution logic (e.g., `ROW_NUMBER() OVER (PARTITION BY plan_id ORDER BY snapshot_ts DESC) = 1`) as a **volatile CTE in Python/config**.
   - **Hard Rule:** The LLM must **never** write or see the raw underlying table definitions inside `plan_snapshot_cte`. It treats `plan_snapshot_cte` as a clean virtual table name; Python prepends the CTE dynamically at runtime **only** when plan/sponsor context is required.

3. **`mapping_cte` — Cross-Domain ID Mapping Bridge (Volatile Virtual CTE)**
   - **Traffic Share (~10% of user queries):** Users sometimes ask cross-domain questions that link benefit provisions to specific plans or sponsors (e.g., *"What is the in-network ER copay and deductible for active PPO plans sponsored by Beacon Corp?"* or *"Which active plans under Sponsor X have $0 preventive dental?"*).
   - **Structure:** Contains **ID mapping keys only** (`bplv_id`, `plan_id`) resolved dynamically from volatile xref/effective-dated association tables.
   - **Real-World Data Nuances:**
     - **Many-to-Many Fan-Out Risk:** One `plan_id` maps to dozens/hundreds of `bplv_id` provision rows, and standard benefit templates (`bplv_id`) can map to multiple plans. Naive `COUNT(*)` or `SUM()` across the join causes severe double-counting unless `COUNT(DISTINCT p.plan_id)` or proper grain grouping is enforced.
     - **Missing-Provision Queries:** Questions like *"Which active plans are missing vision provisions?"* require a `LEFT JOIN` from `plan_snapshot_cte` through `mapping_cte` to `bplv` rather than an `INNER JOIN`.

### Core Engineering Constraints
- **Single SDK Call Orchestration:** One `client.models.generate_content()` call using `google-genai` Automatic Function Calling (AFC), with internal self-correction if the validator or DB returns an error.
- **Modular Plugin Prompting + Few-Shot Routing:** Isolate table schemas and domain SQL rules inside XML `<plugin>` modules (`BPLV_PLUGIN`, `PLAN_SPONSOR_PLUGIN`, `HYBRID_JOIN_PLUGIN`) paired with few-shot tool-call examples.
- **Pydantic 2.0 Schema-Guided Reasoning:** Force top-to-bottom planning (`planner_reasoning` -> `activated_plugins` -> `thought_process` -> `sql_query`) and enforce plugin dependency rules using Pydantic 2.0 validators.
- **Lightweight Regex Guardrails Only (No `sqlglot`):** Enforce read-only DQL (`SELECT`/`WITH`), table whitelisting, CTE shadowing protection, and safe CTE merging using Python's `re` module (backed by a read-only DB role + statement timeout).
- **Mandatory Dual Output:** Deliver both a clean user-facing answer (Thought Process + Natural Language Answer grounded strictly in returned rows) and a thread-safe structured audit trace (`AgentResponse`) for logging/UI inspection.

---

## 2. Thought Process & Proposed Code Implementation

### Key Design Decisions Behind This Prototype
1. **Domain Plugins Instead of Pipeline-Role Plugins:** Instead of making plugins for `PLANNER`, `SQL`, and `SYNTHESIZER` (which all run on every request), plugins map directly to **data domains** (`BPLV_PLUGIN`, `PLAN_SPONSOR_PLUGIN`, `HYBRID_JOIN_PLUGIN`). This lets the LLM ignore irrelevant table rules and lets Python inject only the exact CTEs needed.
2. **Flat Tool Arguments + Internal Pydantic 2.0 Validation:** `execute_sql_query` accepts flat primitive arguments (`str`, `list[str]`) for broad compatibility with `google-genai` AFC schema generation, then immediately constructs `QueryExecutionPlan(...)` inside the tool to run Pydantic 2.0 strict validation and `@model_validator(mode="after")` dependency normalization.
3. **Regex Guardrail + Dynamic CTE Compiler:** Strips comments (`--`, `/* */`) and string literals (`'...'`), neutralizes `EXTRACT`/`TRIM`/`SUBSTRING(... FROM ...)` false positives, blocks DML/DDL keywords and stacked statements, blocks reserved CTE shadowing, validates plugin-to-table alignment, and merges injected CTEs cleanly with any LLM-written `WITH` clause.
4. **Concurrency & Memory Safety:** Uses a per-request tool closure inside `ask()` so `audit_log` is thread-safe under FastAPI/Streamlit concurrency, and fetches `max_rows + 1` rows via `fetchmany()` to cap memory and LLM context size.

### Current Python Prototype (`modular_sql_agent.py`)

```python
import os
import re
import time
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from google import genai
from google.genai import types


# ==============================================================================
# 1. CONFIGURATION & VOLATILE CTE REGISTRY
# ==============================================================================
SQL_DIALECT = "postgres"  # e.g., "postgres", "snowflake", "bigquery", "tsql", "duckdb"
MODEL_NAME = "gemini-2.5-flash"
MAX_ROWS = 200
MAX_TOOL_RETRIES = 4  # 1 initial attempt + up to 3 automatic self-corrections

PluginName = Literal["BPLV_PLUGIN", "PLAN_SPONSOR_PLUGIN", "HYBRID_JOIN_PLUGIN"]
ALLOWED_LOGICAL_TABLES = {"bplv", "plan_snapshot_cte", "mapping_cte"}

CTE_REGISTRY: dict[str, str] = {
    "plan_snapshot_cte": """
        plan_snapshot_cte AS (
            SELECT
                plan_id,
                sponsor_id,
                sponsor_name,
                plan_name,
                product_type,
                funding_type,
                plan_status,
                effective_date,
                termination_date
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY plan_id
                        ORDER BY snapshot_timestamp DESC, effective_date DESC
                    ) AS rn
                FROM raw_plan_sponsor_snapshot
            ) latest_plans
            WHERE rn = 1
        )
    """,
    "mapping_cte": """
        mapping_cte AS (
            SELECT DISTINCT
                bplv_id,
                plan_id
            FROM raw_plan_bplv_xref
            WHERE is_active = 1
        )
    """,
}


# ==============================================================================
# 2. MODULAR DOMAIN PLUGINS (Real BPLV + Plan Snapshot + Mapping Semantics)
# ==============================================================================
PLUGIN_MODULES: dict[PluginName, str] = {
    "BPLV_PLUGIN": """
    <plugin name="BPLV_PLUGIN">
      <domain>Benefit Plan Level Value &amp; Provision Info (BPLV)</domain>
      <table_type>Physical Table: `bplv` (Direct fast query; ZERO CTE overhead)</table_type>
      <schema>
        - `bplv_id` (PK): Unique benefit provision record ID
        - `benefit_category`: e.g., 'Medical', 'Pharmacy', 'Dental', 'Vision', 'Behavioral Health'
        - `service_type`: e.g., 'Emergency Room', 'Urgent Care', 'Preventive Care', 'Specialist Visit', 'Inpatient Hospital'
        - `provision_code`: Standardized benefit/provision code (e.g., 'MED-ER-01', 'DNT-PREV-01')
        - `provision_description`: Human-readable provision rule text
        - `network_tier`: 'IN_NETWORK', 'OUT_OF_NETWORK', or 'TIER_1'
        - `coverage_level`: 'INDIVIDUAL' or 'FAMILY'
        - `copay_amount`: Fixed copay dollar amount (numeric, nullable)
        - `coinsurance_pct`: Member coinsurance percentage 0-100 (numeric, nullable)
        - `deductible_amount`: Deductible dollar amount (numeric, nullable)
        - `moop_amount`: Out-of-pocket maximum dollar amount (numeric, nullable)
        - `deductible_applies`: Boolean flag indicating whether deductible applies first
        - `limit_value`: Numeric visit/day/dollar limit (nullable)
        - `limit_unit`: e.g., 'VISITS_PER_YEAR', 'DAYS_PER_ADMISSION', 'DOLLARS_PER_YEAR'
        - `prior_auth_required`: Boolean flag
        - `plan_year`: Integer benefit year (e.g., 2025, 2026)
      </schema>
      <rules>
        1. Activate alone whenever the question is strictly about benefit provisions, copays, coinsurance, deductibles, MOOP, limits, or prior auth rules without specifying a plan or sponsor.
        2. Query `bplv` directly; NEVER join `mapping_cte` or `plan_snapshot_cte` unless `HYBRID_JOIN_PLUGIN` is also active.
        3. Respect benefit grain: explicitly include or filter `network_tier` and `coverage_level` so IN_NETWORK/OUT_OF_NETWORK or INDIVIDUAL/FAMILY values are never silently mixed in aggregations.
        4. Use case-insensitive matching (`ILIKE` or `LOWER(...)`) for free-text fields like `service_type` and `provision_description`.
      </rules>
    </plugin>
    """,
    "PLAN_SPONSOR_PLUGIN": """
    <plugin name="PLAN_SPONSOR_PLUGIN">
      <domain>Plan-Level &amp; Plan Sponsor-Level Snapshot</domain>
      <table_type>Virtual CTE Table: `plan_snapshot_cte` (Injected dynamically at runtime)</table_type>
      <schema>
        - `plan_id` (PK): Unique plan identifier
        - `sponsor_id`: Employer / plan sponsor identifier
        - `sponsor_name`: Plan sponsor legal or display name
        - `plan_name`: Official plan name
        - `product_type`: e.g., 'PPO', 'HMO', 'HDHP', 'EPO'
        - `funding_type`: e.g., 'ASO', 'FULLY_INSURED'
        - `plan_status`: 'ACTIVE', 'TERMINATED', or 'PENDING'
        - `effective_date`: Plan effective date
        - `termination_date`: Plan termination date (nullable)
      </schema>
      <rules>
        1. Reference `plan_snapshot_cte` by name directly in `FROM` or `JOIN`.
        2. NEVER define `WITH plan_snapshot_cte AS (...)` yourself.
        3. Default to `plan_status = 'ACTIVE'` unless the user explicitly asks about terminated, pending, or historical plans.
        4. Always use `COUNT(DISTINCT plan_id)` when counting plans.
      </rules>
    </plugin>
    """,
    "HYBRID_JOIN_PLUGIN": """
    <plugin name="HYBRID_JOIN_PLUGIN">
      <domain>Cross-Domain ID Mapping Bridge</domain>
      <table_type>Virtual CTE Table: `mapping_cte` (Injected dynamically at runtime; ID keys ONLY)</table_type>
      <schema>
        - `bplv_id` (FK -&gt; `bplv.bplv_id`)
        - `plan_id` (FK -&gt; `plan_snapshot_cte.plan_id`)
      </schema>
      <rules>
        1. Activate ONLY when linking benefit provisions (`bplv`) with plan or sponsor metadata (`plan_snapshot_cte`).
        2. NEVER define `WITH mapping_cte AS (...)` yourself.
        3. Always bridge via `mapping_cte` with explicit table aliases (`b`, `m`, `p`) and prefix EVERY column:
           FROM bplv AS b
           INNER JOIN mapping_cte AS m ON b.bplv_id = m.bplv_id
           INNER JOIN plan_snapshot_cte AS p ON m.plan_id = p.plan_id
        4. For "missing provision" / anti-join questions (e.g., plans without dental benefits), start from `plan_snapshot_cte AS p` and use `LEFT JOIN` through `mapping_cte` to `bplv`.
        5. Guard against many-to-many fan-out: use `COUNT(DISTINCT p.plan_id)` for plan counts and group cleanly by `p.plan_id`, `b.network_tier`, and `b.coverage_level` when comparing provisions across plans.
      </rules>
    </plugin>
    """,
}


# ==============================================================================
# 3. PYDANTIC 2.0 MODELS (Schema-Guided Reasoning + Structured Audit Trace)
# ==============================================================================
class QueryExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    planner_reasoning: str = Field(
        min_length=10,
        description="Stage 1: Identify whether the user needs benefit provisions (bplv), plan/sponsor metadata, or both.",
    )
    activated_plugins: list[PluginName] = Field(
        min_length=1,
        description="Stage 1: Exact list of domain plugins activated for this question.",
    )
    thought_process: str = Field(
        min_length=10,
        description="Stage 2-3: Step-by-step SQL design (grain, columns, filters, join path, aggregations).",
    )
    sql_query: str = Field(
        min_length=10,
        description="Stage 3: Single read-only SELECT query referencing bplv, plan_snapshot_cte, and/or mapping_cte.",
    )

    @model_validator(mode="after")
    def normalize_and_enforce_plugin_dependencies(self) -> "QueryExecutionPlan":
        unique_plugins: list[PluginName] = list(dict.fromkeys(self.activated_plugins))
        plugin_set = set(unique_plugins)

        if "HYBRID_JOIN_PLUGIN" in plugin_set:
            for required in ("BPLV_PLUGIN", "PLAN_SPONSOR_PLUGIN"):
                if required not in plugin_set:
                    unique_plugins.insert(0, required)
        elif {"BPLV_PLUGIN", "PLAN_SPONSOR_PLUGIN"}.issubset(plugin_set):
            unique_plugins.append("HYBRID_JOIN_PLUGIN")

        self.activated_plugins = unique_plugins
        return self


class ToolExecutionAudit(BaseModel):
    attempt: int
    status: Literal["SUCCESS", "ERROR"]
    activated_plugins: list[str]
    planner_reasoning: str
    thought_process: str
    raw_sql: str
    compiled_sql: str | None = None
    row_count: int = 0
    truncated: bool = False
    error_message: str | None = None
    latency_ms: float


class AgentResponse(BaseModel):
    question: str
    final_answer: str
    total_latency_ms: float
    tool_calls: list[ToolExecutionAudit] = Field(default_factory=list)


# ==============================================================================
# 4. SYSTEM PROMPT BUILDER (Modular XML + Few-Shot Routing Examples)
# ==============================================================================
def build_system_prompt(dialect: str = SQL_DIALECT) -> str:
    plugins_xml = "\n".join(PLUGIN_MODULES.values())
    return f"""
<system_instructions>
  <role_and_objective>
    You are an Autonomous Healthcare Benefits Data Analyst and SQL Orchestrator targeting `{dialect}` SQL.
    For every data question, execute a strict single-workflow pipeline:
    1. Plan the query scope and activate ONLY the minimum required domain `<plugin>` modules.
    2. Write a clean, read-only `SELECT` query obeying ONLY the schemas and rules of your activated plugins.
    3. Call `execute_sql_query` to run the SQL. If it returns `status: "ERROR"`, read `error_message`, fix your query or plugin routing, and call `execute_sql_query` again.
    4. Synthesize a transparent Thought Process and an accurate natural-language Answer grounded strictly in the returned rows.
    If the user asks a general greeting or schema capability question that does not require database data, answer directly without calling `execute_sql_query`.
  </role_and_objective>

  <routing_rules>
    - Benefit / provision details only (default, ~80% of questions) -&gt; `["BPLV_PLUGIN"]`
    - Plan or sponsor metadata only -&gt; `["PLAN_SPONSOR_PLUGIN"]`
    - Benefits linked to plans or sponsors -&gt; `["BPLV_PLUGIN", "PLAN_SPONSOR_PLUGIN", "HYBRID_JOIN_PLUGIN"]`
    - NEVER define `WITH plan_snapshot_cte AS ...` or `WITH mapping_cte AS ...`; the backend injects them automatically when their plugins are activated.
  </routing_rules>

  <plugins>
{plugins_xml}
  </plugins>

  <sql_guardrails>
    - Read-only `SELECT` queries only. Never write DML (`INSERT`, `UPDATE`, `DELETE`) or DDL (`DROP`, `CREATE`, `ALTER`).
    - Query ONLY `bplv`, `plan_snapshot_cte`, and `mapping_cte` (plus any uniquely named local helper CTE for aggregations).
    - Always use explicit `JOIN` syntax (never comma joins in `FROM`).
    - Select only necessary columns; avoid `SELECT *` on joins.
  </sql_guardrails>

  <few_shot_examples>
    <example name="1_BPLV_ONLY">
      <user_question>What is the in-network Emergency Room copay and does the deductible apply for 2026 provisions?</user_question>
      <tool_call name="execute_sql_query">
        {{
          "planner_reasoning": "The question asks strictly about Emergency Room copay and deductible rules for plan year 2026 with no plan or sponsor filter. Only the physical bplv table is required.",
          "activated_plugins": ["BPLV_PLUGIN"],
          "thought_process": "Query bplv directly filtering service_type ILIKE '%Emergency Room%', network_tier = 'IN_NETWORK', and plan_year = 2026. Include coverage_level so individual and family rules stay distinct.",
          "sql_query": "SELECT provision_code, provision_description, coverage_level, copay_amount, deductible_applies FROM bplv WHERE service_type ILIKE '%Emergency Room%' AND network_tier = 'IN_NETWORK' AND plan_year = 2026"
        }}
      </tool_call>
    </example>

    <example name="2_PLAN_SPONSOR_ONLY">
      <user_question>How many active ASO plans does Acme Health Systems sponsor, broken down by product type?</user_question>
      <tool_call name="execute_sql_query">
        {{
          "planner_reasoning": "The user wants active plan counts by product_type for a specific plan sponsor and funding_type. No benefit provision data is requested, so only PLAN_SPONSOR_PLUGIN is needed.",
          "activated_plugins": ["PLAN_SPONSOR_PLUGIN"],
          "thought_process": "Query plan_snapshot_cte filtering sponsor_name ILIKE '%Acme Health Systems%', funding_type = 'ASO', and plan_status = 'ACTIVE'. Group by sponsor_name and product_type using COUNT(DISTINCT plan_id).",
          "sql_query": "SELECT sponsor_name, product_type, COUNT(DISTINCT plan_id) AS active_plan_count FROM plan_snapshot_cte WHERE sponsor_name ILIKE '%Acme Health Systems%' AND funding_type = 'ASO' AND plan_status = 'ACTIVE' GROUP BY sponsor_name, product_type"
        }}
      </tool_call>
    </example>

    <example name="3_HYBRID_JOIN">
      <user_question>Compare in-network individual medical coinsurance and deductible amounts for active PPO plans sponsored by Beacon Corp.</user_question>
      <tool_call name="execute_sql_query">
        {{
          "planner_reasoning": "This combines medical coinsurance and deductible rules from bplv with sponsor_name, product_type, and active status from plan_snapshot_cte. Both domains and mapping_cte are required.",
          "activated_plugins": ["BPLV_PLUGIN", "PLAN_SPONSOR_PLUGIN", "HYBRID_JOIN_PLUGIN"],
          "thought_process": "Bridge bplv (b) to plan_snapshot_cte (p) via mapping_cte (m). Filter p.sponsor_name ILIKE '%Beacon Corp%', p.product_type = 'PPO', p.plan_status = 'ACTIVE', b.benefit_category = 'Medical', b.network_tier = 'IN_NETWORK', and b.coverage_level = 'INDIVIDUAL'.",
          "sql_query": "SELECT p.sponsor_name, p.plan_name, b.service_type, b.provision_code, b.coinsurance_pct, b.deductible_amount, b.deductible_applies FROM bplv AS b INNER JOIN mapping_cte AS m ON b.bplv_id = m.bplv_id INNER JOIN plan_snapshot_cte AS p ON m.plan_id = p.plan_id WHERE p.sponsor_name ILIKE '%Beacon Corp%' AND p.product_type = 'PPO' AND p.plan_status = 'ACTIVE' AND b.benefit_category = 'Medical' AND b.network_tier = 'IN_NETWORK' AND b.coverage_level = 'INDIVIDUAL' ORDER BY p.plan_name, b.service_type"
        }}
      </tool_call>
    </example>
  </few_shot_examples>

  <final_output_format>
    After `execute_sql_query` returns `status: "SUCCESS"`, respond with two clear sections:

    ### 1. Thought Process &amp; Query Plan
    - **Activated Plugins:** Exact plugins activated and why.
    - **Reasoning:** Brief explanation of grain, filters, join path, and aggregations.
    - **Executed SQL:** Display the executed SQL query in a ```sql code block.

    ### 2. Answer
    - Provide a clear, direct natural-language answer grounded strictly in the returned data.
    - Use a Markdown table for multi-row comparisons.
    - If `returned_rows` is `0`, explicitly state that no matching records were found—never guess values.
    - If `truncated` is `true`, summarize the displayed rows and state that results were capped at the top {MAX_ROWS} rows.
  </final_output_format>
</system_instructions>
"""


# ==============================================================================
# 5. REGEX SQL GUARDRAIL & DYNAMIC CTE COMPILER (No sqlglot)
# ==============================================================================
FORBIDDEN_KEYWORDS_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|GRANT|REVOKE|EXEC|EXECUTE|CALL|REPLACE|INTO\s+OUTFILE)\b",
    flags=re.IGNORECASE,
)
LOCAL_CTE_RE = re.compile(r"\b([a-zA-Z_]\w*)\s+AS\s*\(", flags=re.IGNORECASE)
TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN)\s+(?!\()([\"`\[]?[a-zA-Z_][\w$]*(?:\.[a-zA-Z_][\w$]*)?[\"`\]]?)",
    flags=re.IGNORECASE,
)
NON_TABLE_FROM_FN_RE = re.compile(
    r"\b(?:EXTRACT|TRIM|SUBSTRING)\s*\([^()]*\)",
    flags=re.IGNORECASE,
)


def _strip_comments_and_literals(sql: str) -> str:
    cleaned = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    cleaned = re.sub(r"--[^\n]*", " ", cleaned)
    cleaned = re.sub(r"'(?:''|[^'])*'", " '' ", cleaned)
    return cleaned


def validate_and_compile_sql(plan: QueryExecutionPlan) -> str:
    raw_sql = plan.sql_query.strip().rstrip(";")
    cleaned = _strip_comments_and_literals(raw_sql).strip()

    if ";" in cleaned:
        raise ValueError("Multiple SQL statements separated by ';' are not permitted.")

    if not re.match(r"^(?:SELECT|WITH)\b", cleaned, flags=re.IGNORECASE):
        raise ValueError("Query must begin with SELECT or WITH.")

    forbidden_match = FORBIDDEN_KEYWORDS_RE.search(cleaned)
    if forbidden_match:
        raise ValueError(f"Forbidden SQL keyword detected: '{forbidden_match.group(1).upper()}'.")

    local_ctes = {name.lower() for name in LOCAL_CTE_RE.findall(cleaned)}
    shadowed_ctes = local_ctes & set(CTE_REGISTRY.keys())
    if shadowed_ctes:
        raise ValueError(
            f"Do not define reserved CTE(s) {sorted(shadowed_ctes)} in your SQL; "
            "the backend injects them automatically when their plugins are activated."
        )

    sql_for_tables = NON_TABLE_FROM_FN_RE.sub(" ", cleaned)
    referenced_tables = {
        match.strip('"`[]').lower() for match in TABLE_REF_RE.findall(sql_for_tables)
    }

    unauthorized_tables = referenced_tables - ALLOWED_LOGICAL_TABLES - local_ctes
    if unauthorized_tables:
        raise ValueError(
            f"Unauthorized table reference(s): {sorted(unauthorized_tables)}. "
            f"Query ONLY {sorted(ALLOWED_LOGICAL_TABLES)}."
        )

    active_plugins = set(plan.activated_plugins)
    if "plan_snapshot_cte" in referenced_tables and "PLAN_SPONSOR_PLUGIN" not in active_plugins:
        raise ValueError("SQL references `plan_snapshot_cte` without activating `PLAN_SPONSOR_PLUGIN`.")
    if "mapping_cte" in referenced_tables and "HYBRID_JOIN_PLUGIN" not in active_plugins:
        raise ValueError("SQL references `mapping_cte` without activating `HYBRID_JOIN_PLUGIN`.")

    ctes_to_inject: list[str] = []
    if "PLAN_SPONSOR_PLUGIN" in active_plugins:
        ctes_to_inject.append(CTE_REGISTRY["plan_snapshot_cte"].strip().rstrip(";"))
    if "HYBRID_JOIN_PLUGIN" in active_plugins:
        ctes_to_inject.append(CTE_REGISTRY["mapping_cte"].strip().rstrip(";"))

    if not ctes_to_inject:
        return raw_sql

    with_match = re.match(r"^WITH\s+(RECURSIVE\s+)?", raw_sql, flags=re.IGNORECASE)
    if with_match:
        recursive_kw = "RECURSIVE " if with_match.group(1) else ""
        llm_body = raw_sql[with_match.end():]
        return f"WITH {recursive_kw}" + ",\n".join(ctes_to_inject) + f",\n{llm_body}"

    return "WITH " + ",\n".join(ctes_to_inject) + f"\n{raw_sql}"


# ==============================================================================
# 6. THREAD-SAFE SINGLE-CALL AGENT ORCHESTRATOR
# ==============================================================================
DBExecutorFn = Callable[[str, int], tuple[list[dict[str, Any]], bool]]


class ModularSQLAgent:
    def __init__(
        self,
        db_executor: DBExecutorFn,
        api_key: str | None = None,
        model: str = MODEL_NAME,
        dialect: str = SQL_DIALECT,
        max_rows: int = MAX_ROWS,
    ) -> None:
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.db_executor = db_executor
        self.model = model
        self.dialect = dialect
        self.max_rows = max_rows
        self.system_prompt = build_system_prompt(dialect=dialect)

    def ask(self, user_question: str) -> AgentResponse:
        start_total = time.perf_counter()
        audit_log: list[ToolExecutionAudit] = []

        def execute_sql_query(
            planner_reasoning: str,
            activated_plugins: list[str],
            thought_process: str,
            sql_query: str,
        ) -> dict[str, Any]:
            """
            Validates the query plan and read-only SQL, dynamically injects required CTEs,
            and executes the compiled SQL against the read-only database.
            """
            attempt_no = len(audit_log) + 1
            t0 = time.perf_counter()
            compiled_sql: str | None = None

            try:
                plan = QueryExecutionPlan(
                    planner_reasoning=planner_reasoning,
                    activated_plugins=activated_plugins,  # type: ignore[arg-type]
                    thought_process=thought_process,
                    sql_query=sql_query,
                )
                compiled_sql = validate_and_compile_sql(plan)
                rows, truncated = self.db_executor(compiled_sql, self.max_rows)
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

                audit_log.append(
                    ToolExecutionAudit(
                        attempt=attempt_no,
                        status="SUCCESS",
                        activated_plugins=list(plan.activated_plugins),
                        planner_reasoning=plan.planner_reasoning,
                        thought_process=plan.thought_process,
                        raw_sql=plan.sql_query,
                        compiled_sql=compiled_sql,
                        row_count=len(rows),
                        truncated=truncated,
                        latency_ms=elapsed_ms,
                    )
                )
                return {
                    "status": "SUCCESS",
                    "activated_plugins": plan.activated_plugins,
                    "executed_sql": compiled_sql,
                    "returned_rows": len(rows),
                    "truncated": truncated,
                    "rows": rows,
                }

            except (ValidationError, ValueError, Exception) as exc:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                err_msg = str(exc)
                audit_log.append(
                    ToolExecutionAudit(
                        attempt=attempt_no,
                        status="ERROR",
                        activated_plugins=activated_plugins,
                        planner_reasoning=planner_reasoning,
                        thought_process=thought_process,
                        raw_sql=sql_query,
                        compiled_sql=compiled_sql,
                        error_message=err_msg,
                        latency_ms=elapsed_ms,
                    )
                )
                return {
                    "status": "ERROR",
                    "error_message": err_msg,
                    "instruction": "Review error_message, fix the activated_plugins or SQL query, and call execute_sql_query again.",
                }

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_question,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                tools=[execute_sql_query],
                temperature=0.0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=MAX_TOOL_RETRIES
                ),
            ),
        )

        total_ms = round((time.perf_counter() - start_total) * 1000, 2)
        return AgentResponse(
            question=user_question,
            final_answer=response.text or "No response was generated.",
            total_latency_ms=total_ms,
            tool_calls=audit_log,
        )
```

---

## 3. Deep Theoretical Audit Checklist (Inspect Every Python Module & Domain Layer)

Evaluate this solution theoretically and systematically across **all six dimensions** below:

### A. `google-genai` SDK & Automatic Function Calling (AFC)
1. **Tool Schema Extraction & Argument Ordering:** Does `google-genai` reliably infer the JSON schema, types, and parameter order from the nested closure `execute_sql_query(...)`? Does Gemini respect argument ordering in function calls (for Schema-Guided Reasoning), or can function-call JSON keys be emitted alphabetically or out-of-order by the model/API?
2. **AFC Exhaustion & Non-Text Parts:** What happens to `response.text` if AFC hits `maximum_remote_calls=4` without succeeding, or when using `gemini-2.5-flash` with thinking parts enabled? Can `response.text` raise a `ValueError` or return empty string?
3. **JSON/Protobuf Serialization of Database Row Types:** Real `bplv` and `plan_snapshot_cte` queries return Python `Decimal` (`copay_amount`, `coinsurance_pct`, `deductible_amount`), `datetime.date` (`effective_date`), `datetime.datetime`, and `UUID`. Will `google-genai`'s tool-response serializer crash on raw `Decimal` or `date` objects inside `rows` unless sanitized to JSON-native primitives first?
4. **Prompt Caching & Token Efficiency:** How well does the static XML `system_instruction` pair with Gemini implicit/explicit context caching across thousands of BPLV queries?

### B. `pydantic>=2.0` Validation & Schema-Guided Reasoning
1. **Dependency Auto-Correction Trap:** Inspect `normalize_and_enforce_plugin_dependencies()`. If the LLM outputs `activated_plugins=["BPLV_PLUGIN", "PLAN_SPONSOR_PLUGIN"]` and writes a broken SQL query that forgets `mapping_cte` (e.g., tries `JOIN plan_snapshot_cte ON b.plan_id = p.plan_id`), silently appending `"HYBRID_JOIN_PLUGIN"` injects `mapping_cte` even though the SQL didn't use it! Should the validator **reject** missing `HYBRID_JOIN_PLUGIN` with an instructive error (or check `mapping_cte` presence when both tables are referenced) instead of silently mutating?
2. **ValidationError Readability for Self-Correction:** When `QueryExecutionPlan` raises a Pydantic v2 `ValidationError` (e.g., invalid plugin name), `str(exc)` includes verbose URLs and traceback lines (`For further information visit https://errors.pydantic.dev/...`). Is it better to serialize `exc.errors(include_url=False)` into concise actionable feedback?
3. **Model Serialization & Immutability:** Are `AgentResponse` and `ToolExecutionAudit` cleanly serializable for FastAPI/JSON logging, and should `QueryExecutionPlan` sanitize SQL markdown fences (e.g., if the LLM wraps `sql_query` in ` ```sql ... ``` `)?

### C. Python `re` (Regex) Guardrails & Dynamic CTE Compiler (Without `sqlglot`)
Stress-test `validate_and_compile_sql()` against real SQL edge cases:
1. **Leading Comments Before `WITH` Bug:** `cleaned` strips comments before checking `^(?:SELECT|WITH)`, **but** `re.match(r"^WITH...", raw_sql)` runs against `raw_sql`. If Gemini outputs `-- Fetch active plans\nWITH agg AS (...) SELECT ...`, `with_match` fails to match and Python prepends a second `WITH`, causing a fatal syntax error!
2. **CTE Definition & Shadowing Gaps:**
   - What about column-list CTEs like `WITH agg(plan_id, total) AS (...)` or PostgreSQL `WITH agg AS MATERIALIZED (...)` or `NOT MATERIALIZED`? `LOCAL_CTE_RE` misses both!
   - What if `CTE_REGISTRY` itself contains a CTE that references another CTE, or what if `raw_sql` starts with a parenthesized query `(SELECT ... FROM bplv)`?
3. **Table Extraction Bypasses & False Positives:**
   - Comma joins (`FROM bplv, raw_plan_sponsor_snapshot`) — `TABLE_REF_RE` only sees `bplv` and misses the second comma-separated table!
   - `LATERAL` joins (`JOIN LATERAL ...`), `FROM ONLY bplv`, schema-qualified names (`public.bplv` vs `public.raw_plan_sponsor_snapshot`), or nested parentheses inside `EXTRACT(YEAR FROM COALESCE(effective_date, CURRENT_DATE))` (where `[^()]*` fails!).
   - Dollar-quoted strings (`$$...$$` in PostgreSQL) or backslash-escaped quotes (`\'`) in `_strip_comments_and_literals`.
4. **Forbidden Keyword Gaps:** Missing dialect-specific hazards such as `COPY`, `LOAD`, `PRAGMA`, `ATTACH`, `SET`, `SHOW`, `VACUUM`, `SELECT ... INTO`, `PG_SLEEP`, `DBLINK`, `OPENROWSET`.
5. **Defense-in-Depth Boundary:** Explicitly clarify what regex **can** guarantee vs. what **must** be enforced at the database layer (read-only role, least-privilege view/table grants, statement timeout, row limit).

### D. Real-World BPLV, Plan Snapshot CTE & Mapping CTE Semantics
1. **Raw Table Name Leakage in Tool Return:** Notice `execute_sql_query` returns `"executed_sql": compiled_sql` back to Gemini! That exposes `raw_plan_sponsor_snapshot` and `raw_plan_bplv_xref` into the model's conversation context—which can tempt Gemini to query raw tables directly on retries AND exposes internal warehouse table names in the user-facing answer. Should the tool return `logical_sql` (virtual table SQL) to the LLM while keeping `compiled_sql` strictly inside backend `audit_log`?
2. **BPLV Grain & Fan-Out Safety:** Verify whether the plugin rules and few-shot examples adequately prevent:
   - Mixing `INDIVIDUAL` and `FAMILY` or `IN_NETWORK` and `OUT_OF_NETWORK` rows in averages/sums.
   - Fan-out double-counting across `bplv <-> mapping_cte <-> plan_snapshot_cte`.
   - Missing-provision queries requiring `LEFT JOIN` (anti-join) instead of `INNER JOIN`.
   - `NULL` handling on optional columns (`copay_amount` is often `NULL` when a service is coinsurance-only, and vice versa).
3. **Volatile CTE Dependency Safety:** What if `plan_snapshot_cte` or `mapping_cte` changes its output column names or filters? How should schema metadata in `PLUGIN_MODULES` stay synchronized with `CTE_REGISTRY` (e.g., co-located dataclass/Pydantic plugin definitions + startup smoke test)?

### E. Security, Prompt Injection & Production Operations
1. **Indirect Prompt Injection via DB Rows:** What if a `provision_description` or `sponsor_name` cell contains text resembling instructions? How should tool output wrap data rows and how should `<system_instructions>` instruct the model to treat DB rows strictly as untrusted data?
2. **DB Timeout & Connection Resilience:** How should `db_executor` enforce per-query timeouts, handle transient DB errors vs. SQL syntax errors (avoiding wasting LLM retries on connection refused/auth errors!), and prevent runaway queries?
3. **Observability & Evaluation:** What minimal telemetry and automated `pytest` assertions should ship alongside this agent before production rollout?

---

## 4. Required Output Format

Please deliver your review in these exact three sections:

1. **Architecture Verdict:** Clear, candid verdict on whether this single-call Modular Plugin + Pydantic 2.0 + Regex architecture is production-ready for a 3-source BPLV + volatile CTE enterprise system, and where its scaling boundary lies.
2. **Ranked Theoretical Audit Findings:** List every bug, edge case, data-grain trap, SDK quirk, and security gap discovered across Sections 3A–3E, ranked by severity (**Critical**, **High**, **Moderate**, **Minor**) with a brief technical explanation for each.
3. **Final Hardened Best Solution:** Provide the **complete, copy-paste-ready Python implementation** (`modular_sql_agent.py`) that fixes **every single issue** identified in your audit—using strictly `google-genai`, `pydantic>=2.0`, and standard Python (`re`, `json`, `decimal`, `datetime`, etc.) with **no `sqlglot`**. Include:
   - Co-located plugin + CTE registry design so prompts and CTEs cannot drift.
   - Hardened regex validator & CTE compiler (handling leading comments, comma joins, nested `EXTRACT/TRIM/SUBSTRING`, `MATERIALIZED`/column-list CTEs, markdown-fenced SQL stripping, and safe `WITH` merging).
   - Safe JSON/type serializer for `Decimal`, `date`, `datetime`, `UUID` database rows.
   - Separation of `logical_sql` (returned to LLM/user) vs. `compiled_sql` (logged in backend audit trace).
   - Distinction between retryable SQL/validation errors vs. non-retryable infrastructure/DB connection errors.
   - Prompt-injection-resistant data synthesis instructions and real BPLV/CTE few-shot examples (including an anti-join / grain-safe example).