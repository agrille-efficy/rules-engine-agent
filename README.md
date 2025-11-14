# Rules Engine Agent - Technical Documentation

**Version:** 1.0  
**Last Updated:** November 7, 2025  
**Project Type:** Automated Data Integration System  
**Primary Language:** Python 3.10+

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Product Overview](#2-product-overview)
3. [Business Context](#3-business-context)
4. [System Architecture](#4-system-architecture)
5. [Technical Implementation](#5-technical-implementation)
6. [Core Components](#6-core-components)
7. [Workflow Execution](#7-workflow-execution)
8. [Data Models](#8-data-models)
9. [Configuration & Setup](#9-configuration--setup)
10. [Development Guidelines](#10-development-guidelines)
11. [Testing Strategy](#11-testing-strategy)
12. [Troubleshooting](#12-troubleshooting)
13. [API Reference](#13-api-reference)

---

## 1. Executive Summary

### 1.1 What is the Rules Engine Agent?

The Rules Engine Agent is an intelligent automation system that maps CSV file columns to database tables and fields with minimal human intervention. It replaces manual data mapping processes that typically take hours with an automated workflow that completes in minutes while providing confidence scores and validation for every mapping decision.

### 1.2 Core Value Proposition

**Problem Solved:**  
Data integration teams spend significant time manually analyzing CSV files, identifying appropriate database tables, and creating field-to-field mappings. This process is:
- Time-consuming (2-4 hours per file)
- Error-prone (manual mapping mistakes)
- Not scalable (requires expert knowledge)
- Difficult to audit (no confidence metrics)

**Solution Provided:**  
An automated agent that:
- Analyzes CSV structure and semantics in seconds
- Uses semantic search to find matching database tables
- Creates intelligent field mappings across multiple tables
- Provides confidence scores and validation metrics
- Flags low-confidence mappings for human review

### 1.3 Key Metrics

- **Speed:** Reduces mapping time from hours to 2-5 minutes
- **Accuracy:** 70-85% automated mapping coverage with confidence scoring
- **Scalability:** Can process multiple files concurrently
- **Auditability:** Complete logging and confidence tracking for compliance

---

## 2. Product Overview

### 2.1 Product Capabilities

#### 2.1.1 Automated File Analysis
- Reads CSV files and extracts structural information
- Detects data types, null values, and sample data
- Translates French column names to English for better semantic matching
- Identifies data quality issues automatically

#### 2.1.2 Intelligent Table Matching
- Uses RAG (Retrieval-Augmented Generation) with vector search
- Entity-first architecture prioritizes core business entities
- Two-stage search: entities first, then relationships
- Handles both single-table and multi-table scenarios

#### 2.1.3 Multi-Strategy Field Mapping
- **Exact matching:** Direct name matches
- **Fuzzy matching:** String similarity algorithms
- **Semantic matching:** Word overlap and prefix detection
- **LLM-based matching:** GPT-4o for complex cases

#### 2.1.4 Quality Assurance
- Judge system analyzes mapping quality
- Refinement process removes suspicious mappings
- Validation with coverage and confidence metrics
- Human review workflow for low-confidence results

### 2.2 Use Cases

#### Use Case 1: CRM Data Import
**Scenario:** Marketing team receives lead data from external partner  
**Process:**  
1. Upload CSV with 30 columns of lead information
2. Agent identifies "Opportunity" as primary table
3. Maps core fields to Opportunity, relationships to Contact/Company
4. Generates 85% coverage with high confidence
5. Flags 5 unmapped columns for manual review

#### Use Case 2: Multi-Table Relationship Mapping
**Scenario:** Integration of complex sales data spanning multiple entities  
**Process:**  
1. CSV contains mixed entity data (opportunities, contacts, companies)
2. Agent detects multi-entity structure through semantic grouping
3. Distributes columns across 3-5 related tables
4. Determines insertion order (entities before relationships)
5. Validates referential integrity requirements

#### Use Case 3: Data Quality Validation
**Scenario:** Existing mapping needs validation before migration  
**Process:**  
1. Agent analyzes existing CSV structure
2. Compares against current database schema
3. Identifies mismatched data types or missing fields
4. Provides confidence scores for each mapping
5. Recommends improvements or corrections

---

## 3. Business Context

### 3.1 Target Users

**Primary Users:**
- Data Engineers: Define and validate data mappings
- ETL Developers: Integrate mappings into data pipelines
- Business Analysts: Review and approve mapping decisions

**Secondary Users:**
- Product Managers: Monitor automation effectiveness
- QA Teams: Validate mapping accuracy
- Database Administrators: Ensure schema compatibility

### 3.2 Integration Points

The Rules Engine Agent integrates with:

1. **Efficy CRM System**
   - Fetches database schema via DICO API
   - Validates field compatibility
   - Ensures naming convention compliance

2. **Vector Database (Qdrant)**
   - Stores database schema embeddings
   - Enables semantic search for table matching
   - Maintains knowledge base of historical mappings

3. **OpenAI API**
   - GPT-4o for semantic understanding
   - Embeddings for vector search
   - LLM-based field matching for complex cases

4. **Workflow Orchestration**
   - Can be invoked via CLI for batch processing
   - API-ready for integration into larger pipelines
   - Supports checkpointing for long-running processes

### 3.3 Workflow Position

The agent sits between data ingestion and ETL execution:

```
Data Source → Rules Engine Agent → Validated Mappings → ETL Pipeline → Database
```

**Input:** Raw CSV file + optional context  
**Output:** Validated field mappings with confidence scores  
**Decision Point:** Routes to execution or human review based on confidence

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Rules Engine Agent                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Workflow Engine (LangGraph)                │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  State Machine with Conditional Routing        │  │   │
│  │  │  - Checkpointing for resilience                │  │   │
│  │  │  - Step-by-step execution tracking             │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ File Analysis│  │ RAG Matching │  │Table Selection│      │
│  │     Node     │→ │     Node     │→ │     Node      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                              ↓              │
│                                    ┌──────────────┐         │
│                                    │Field Mapping │         │
│                                    │     Node     │         │
│                                    └──────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
┌────────────────┐  ┌────────────────┐  ┌──────────────────┐
│  OpenAI API    │  │  Qdrant Vector │  │  Efficy DICO API │
│  (GPT-4o)      │  │  Database      │  │  (Schema Source) │
└────────────────┘  └────────────────┘  └──────────────────┘
```

### 4.2 Architectural Principles

#### 4.2.1 State Machine Design
The system uses LangGraph's StateGraph for orchestration:
- **Deterministic flow:** Each step has defined next-step routing
- **Stateful execution:** WorkflowState carries all data between nodes
- **Conditional branching:** Routing logic decides paths based on results
- **Error handling:** Dedicated error and review nodes for failure cases

#### 4.2.2 Separation of Concerns
- **Nodes:** Execute single-purpose transformations on state
- **Routing:** Makes decisions about workflow direction
- **Services:** Contain business logic (file analysis, mapping, matching)
- **Models:** Define data structures with validation

#### 4.2.3 Resilience Patterns
- **Circuit Breaker:** Prevents cascading failures in external API calls
- **Retry Logic:** Exponential backoff for transient failures
- **Checkpointing:** State persistence for workflow recovery
- **Graceful Degradation:** Falls back to simpler strategies when advanced ones fail

---

## 5. Technical Implementation

### 5.1 Technology Stack

**Core Framework:**
- Python 3.10+
- LangGraph (state machine orchestration)
- LangChain (LLM integration)

**AI/ML Services:**
- OpenAI GPT-4o (semantic understanding, field matching)
- OpenAI text-embedding-3-small (vector embeddings)
- Qdrant (vector database)

**Data Processing:**
- Pandas (CSV manipulation)
- Pydantic (data validation)

**Supporting Libraries:**
- Requests (HTTP client)
- Python-dotenv (configuration management)
- Logging (comprehensive audit trail)

### 5.2 Project Structure

```
rules-engine-agent/
├── Agent/
│   ├── __init__.py
│   ├── main.py                    # Entry point and CLI
│   │
│   ├── config/                     # Configuration management
│   │   ├── settings.py            # Pydantic settings with validation
│   │   ├── constants.py           # System constants
│   │   └── logging_config.py      # Logging setup
│   │
│   ├── core/                       # Orchestration layer
│   │   ├── workflow_engine.py     # Main execution controller
│   │   ├── graph_builder.py       # LangGraph workflow definition
│   │   └── resilience.py          # Circuit breaker, retry logic
│   │
│   ├── models/                     # Data models (Pydantic)
│   │   ├── workflow_state.py      # State machine data structure
│   │   ├── file_analysis_model.py # File analysis results
│   │   ├── rag_match_model.py     # Table matching results
│   │   └── validators.py          # Input validation and sanitization
│   │
│   ├── nodes/                      # Workflow processing nodes
│   │   ├── file_analysis_node.py  # Step 1: Analyze file structure
│   │   ├── rag_matching_node.py   # Step 2: Find candidate tables
│   │   ├── table_selection_node.py # Step 3: Select best tables
│   │   └── field_mapping_node.py  # Step 4: Map columns to fields
│   │
│   ├── routing/                    # Workflow decision logic
│   │   └── routing_logic.py       # Conditional routing functions
│   │
│   ├── services/                   # Business logic layer
│   │   ├── file_analyzer.py       # CSV analysis and translation
│   │   ├── mapper.py              # Multi-strategy field mapping
│   │   ├── table_matcher.py       # RAG-based table matching
│   │   ├── translator.py          # Column name translation
│   │   └── clients/               # External API clients
│   │       ├── openai_client.py   # Resilient OpenAI wrapper
│   │       └── qdrant_client.py   # Resilient Qdrant wrapper
│   │
│   ├── rag/                        # RAG pipeline components
│   │   ├── pipeline.py            # Entity-first RAG orchestration
│   │   ├── vector_store_builder.py # Schema embedding generator
│   │   ├── vector_search.py       # Semantic search engine
│   │   ├── query_generator.py     # Search query optimization
│   │   ├── domain_classifier.py   # Business domain detection
│   │   ├── dico_client.py         # Efficy DICO API client
│   │   ├── memory.py              # Entity-relation memory
│   │   └── chunk_generator.py     # Schema chunking for embeddings
│   │
│   ├── tools/                      # Utility functions
│   │   └── file_tools.py          # File I/O operations
│   │
│   └── tests/                      # Unit and integration tests
│       ├── test_workflow_engine.py
│       ├── test_file_analysis.py
│       ├── test_rag_matching.py
│       ├── test_table_selection.py
│       ├── test_field_mapping.py
│       └── test_multi_table_mapping.py
│
├── .env                            # Environment configuration
├── .gitignore
├── DOCUMENTATION.md                # This file
├── workflow_graph.png              # Visual workflow diagram
└── visualize_graph.py              # Graph visualization script
```

### 5.3 Design Patterns

#### 5.3.1 State Machine Pattern
All workflow logic is expressed as a directed graph with conditional edges:
- **Nodes:** Stateless transformation functions
- **Edges:** Deterministic routing between nodes
- **State:** Immutable WorkflowState passed through pipeline

#### 5.3.2 Service Layer Pattern
Business logic is separated from workflow orchestration:
- **Nodes call Services:** Keep nodes thin, logic in services
- **Services are testable:** Can be unit tested independently
- **Services are reusable:** Can be called from multiple nodes

#### 5.3.3 Repository Pattern
External data access is abstracted through client wrappers:
- **Resilient clients:** Wrap external APIs with retry/circuit breaker
- **Interface consistency:** All clients provide similar error handling
- **Dependency injection:** Clients can be mocked for testing

---

## 6. Core Components

### 6.1 Workflow Engine

**File:** `Agent/core/workflow_engine.py`

**Purpose:** Main orchestrator that initializes, configures, and executes the workflow.

**Key Responsibilities:**
- Initialize the LangGraph state machine
- Create initial WorkflowState from user input
- Execute workflow with step-by-step streaming
- Track execution history and timing
- Log comprehensive summaries

**Key Methods:**

```python
def __init__(self, use_checkpointer: bool = True)
    """Initialize with optional state checkpointing."""

def initialize()
    """Build and compile the workflow graph."""

def run(file_path: str, user_context: Optional[str] = None, 
        table_preference: Optional[str] = None) -> Dict[str, Any]
    """Execute complete workflow for a file."""
```

**Configuration:**
- `use_checkpointer`: Enable state persistence for recovery
- Execution tracked in `execution_history` list
- Supports visualization via `visualize_workflow()`

### 6.2 Graph Builder

**File:** `Agent/core/graph_builder.py`

**Purpose:** Constructs the LangGraph state machine with nodes, edges, and routing logic.

**Workflow Graph:**

```
START
  ↓
file_analysis (Step 1: Analyze CSV structure)
  ├─ success → rag_matching
  └─ error → error_handler
  
rag_matching (Step 2: Find candidate tables via RAG)
  ├─ success → table_selection
  └─ error → error_handler
  
table_selection (Step 3: Select best table(s))
  ├─ success → field_mapping
  └─ error → error_handler
  
field_mapping (Step 4: Map columns to fields)
  ├─ high confidence → END
  ├─ medium confidence → review_handler
  └─ error → error_handler
  
error_handler → END
review_handler → END
```

**Key Components:**

1. **Nodes:**
   - `file_analysis`: Analyzes CSV structure and semantics
   - `rag_matching`: Uses vector search to find candidate tables
   - `table_selection`: Selects best matching tables
   - `field_mapping`: Maps columns to database fields
   - `error`: Handles failures with detailed logging
   - `review`: Flags mappings needing human validation

2. **Conditional Edges:**
   - `route_after_file_analysis`: Success → rag_matching, Failure → error
   - `route_after_rag_matching`: Tables found → table_selection, None → error
   - `route_after_table_selection`: Selected → field_mapping, Failed → error
   - `route_after_field_mapping`: Valid → end, Review needed → review, Failed → error

3. **Checkpointing:**
   - Uses LangGraph's `MemorySaver` for state persistence
   - Enables recovery if execution fails mid-workflow
   - Thread-safe execution with unique thread IDs

### 6.3 Workflow State

**File:** `Agent/models/workflow_state.py`

**Purpose:** TypedDict that defines all state fields passed through the workflow.

**State Structure:**

```python
class WorkflowState(TypedDict, total=False):
    # Input fields
    file_path: str
    user_context: Optional[str]
    table_preference: Optional[str]
    
    # Step results
    file_analysis_result: Optional[FileAnalysisResult]
    rag_match_result: Optional[TableMatchResult]
    selected_table: Optional[str]
    selected_schema: Optional[str]
    selected_table_metadata: Optional[dict]
    field_mapping_result: Optional[FieldMappingResult]
    
    # Workflow control
    workflow_step: str
    workflow_status: str
    steps_completed: list
    
    # Error tracking
    errors: list
    last_error: Optional[str]
    
    # Messaging
    messages: list
```

**State Flow:**
- Each node receives state, processes it, returns updated state
- LangGraph automatically merges state updates
- Immutable pattern: nodes don't modify state directly

### 6.4 Processing Nodes

#### 6.4.1 File Analysis Node

**File:** `Agent/nodes/file_analysis_node.py`  
**Service:** `Agent/services/file_analyzer.py`

**Purpose:** Analyzes CSV file structure, data types, and translates column names.

**Process:**
1. Read CSV file using Pandas
2. Detect data types for each column
3. Extract sample values and calculate null counts
4. Translate French column names to English
5. Detect data quality issues
6. Return structured `FileAnalysisResult`

**Key Features:**
- Supports UTF-8 and Latin-1 encodings
- Handles various delimiters (comma, semicolon, tab)
- Translation improves semantic matching accuracy
- Quality metrics identify potential issues

**Output Model:**
```python
FileAnalysisResult:
    structure: FileStructureInfo
    columns: List[ColumnMetadata]  # Original + English names
    quality_metrics: DataQualityMetrics
    sample_data: List[dict]
    delimiter: str
    encoding: str
```

#### 6.4.2 RAG Matching Node

**File:** `Agent/nodes/rag_matching_node.py`  
**Service:** `Agent/services/table_matcher.py`  
**Pipeline:** `Agent/rag/pipeline.py`

**Purpose:** Uses semantic search to find database tables matching the file.

**Two-Stage Entity-First Architecture:**

**Stage 1: Entity Discovery**
- Generate semantic search queries from file analysis
- Search vector database for Entity tables only
- Rank by composite score (similarity + query coverage)
- Store high-confidence entities (threshold: 0.6)

**Stage 2: Relationship Discovery**
- If relationship data detected: search Relation tables
- Otherwise: search tables related to best entity
- Include critical relationship patterns (Company, User, Contact)
- Store high-confidence relations (threshold: 0.5)

**Vector Search Process:**
1. Generate embeddings for search queries
2. Query Qdrant vector database
3. Filter by table kind (Entity vs Relation)
4. Calculate composite scores
5. Rank and return top candidates

**Output Model:**
```python
TableMatchResult:
    matched_tables: List[TableMatch]
    search_query: str
    total_candidates: int

TableMatch:
    table_name: str
    similarity_score: float
    confidence: str  # high/medium/low
    matching_columns: List[str]
    metadata: dict  # table_code, table_kind, fields
```

#### 6.4.3 Table Selection Node

**File:** `Agent/nodes/table_selection_node.py`

**Purpose:** Selects the primary table for mapping from RAG candidates.

**Selection Strategy:**
1. Use `table_preference` if provided and found in candidates
2. Otherwise select highest-scoring Entity table
3. Fallback to highest-scoring table of any kind
4. Validate selection has sufficient fields

**Criteria:**
- Prioritize Entity tables over Relation tables
- Require minimum composite score (0.3)
- Consider field count and coverage
- Log selection reasoning

**Output:**
- `selected_table`: Primary table name
- `selected_schema`: Table schema name (if applicable)
- `selected_table_metadata`: Full table metadata for mapping

#### 6.4.4 Field Mapping Node

**File:** `Agent/nodes/field_mapping_node.py`  
**Service:** `Agent/services/mapper.py`

**Purpose:** Maps CSV columns to database fields using multi-strategy approach.

**Mapping Process:**

**Step 1: Semantic Grouping**
Categorizes columns by domain:
- Entity core fields (name, status, type)
- Foreign keys (IDs, unique identifiers)
- User references (owner, creator)
- Company/Contact references
- Relationship data
- Metadata (dates, timestamps)
- Financial data (amounts, prices)

**Step 2: Multi-Strategy Matching**
For each column, tries 4 strategies:
1. **Exact Match:** Direct name equality (confidence: 1.0)
2. **Fuzzy Match:** String similarity via Levenshtein (threshold: 0.69)
3. **Semantic Match:** Word overlap, prefix matching (threshold: 0.6)
4. **LLM Match:** GPT-4o batch inference for complex cases (threshold: 0.5)

**Step 3: Smart Multi-Table Assignment**
Distributes columns across multiple tables:
- Entity fields → Primary table
- Foreign keys → Relationship tables
- Remaining columns → Best matching table

**Step 4: Judge Analysis**
Evaluates mapping quality:
- Detects overloaded fields (>2 columns → same field)
- Identifies identical confidence scores (algorithmic issues)
- Flags excessive generic field mappings
- Calculates average confidence

**Step 5: Refinement (if needed)**
Removes suspicious mappings:
- Drops low-confidence mappings to overloaded fields
- Removes generic field assignments
- Adjusts based on type mismatches

**Step 6: Validation**
Calculates metrics:
- Coverage percentage
- Average confidence
- Unmapped columns
- Data type compatibility

**Output Model:**
```python
MultiTableMappingResult:
    source_file: str
    total_source_columns: int
    table_mappings: List[TableFieldMapping]
    overall_coverage: float
    overall_confidence: str
    unmapped_columns: List[str]
    is_valid: bool
    requires_review: bool

TableFieldMapping:
    table_name: str
    table_type: str  # Entity/Relation
    mappings: List[FieldMapping]
    validation: MappingValidationResult
    confidence: float
    insertion_order: int

FieldMapping:
    source_column: str
    target_column: str
    confidence_score: float
    match_type: str  # exact/fuzzy/semantic/llm
    data_type_compatible: bool
```

### 6.5 Routing Logic

**File:** `Agent/routing/routing_logic.py`

**Purpose:** Conditional routing functions that determine workflow paths.

**Routing Functions:**

```python
def route_after_file_analysis(state) -> Literal["rag_matching", "error"]
    """Route based on file analysis success."""

def route_after_rag_matching(state) -> Literal["table_selection", "error"]
    """Route based on candidate tables found."""

def route_after_table_selection(state) -> Literal["field_mapping", "error"]
    """Route based on table selection success."""

def route_after_field_mapping(state) -> Literal["validation", "review", "error", "end"]
    """Route based on mapping quality."""
```

**Decision Criteria:**
- Check for errors in state
- Validate required results exist
- Evaluate confidence and coverage metrics
- Route to review if quality below threshold
- Route to error on failures

### 6.6 RAG Pipeline

**File:** `Agent/rag/pipeline.py`

**Purpose:** Orchestrates Entity-First RAG architecture for semantic table matching.

**Components:**

#### 6.6.1 Query Generator
**File:** `Agent/rag/query_generator.py`

Generates optimized search queries from file analysis:
- Domain-specific queries (Sales, Finance, HR)
- Column group queries (entity core, relationships)
- Context-aware queries (user context integration)
- Sanitizes input to prevent injection attacks

#### 6.6.2 Vector Search Service
**File:** `Agent/rag/vector_search.py`

Performs semantic search in Qdrant:
- `search_entity_tables_only()`: Filter by table_kind = Entity
- `search_relation_tables_only()`: Filter by table_kind = Relation
- `search_related_tables()`: Find tables referencing an entity
- `rank_tables_by_relevance()`: Composite scoring algorithm

#### 6.6.3 Domain Classifier
**File:** `Agent/rag/domain_classifier.py`

Classifies data by business domain:
- Detects Sales, Finance, HR, Operations, Marketing domains
- Identifies relationship/mapping data patterns
- Provides domain-specific search hints

#### 6.6.4 Entity-Relation Memory
**File:** `Agent/rag/memory.py`

Maintains discovered entities and relationships:
- Stores high-confidence entities (threshold: 0.6)
- Stores relations with entity references
- Provides best entity/relation retrieval
- Tracks relationship data flag

#### 6.6.5 Vector Store Builder
**File:** `Agent/rag/vector_store_builder.py`

Populates Qdrant with database schema:
- Fetches schema from DICO API
- Chunks table metadata for embedding
- Generates embeddings via OpenAI
- Indexes in Qdrant with metadata filters

### 6.7 Mapper Service

**File:** `Agent/services/mapper.py`

**Purpose:** Unified multi-table mapping with 4-strategy approach and refinement.

**Key Features:**

1. **Multi-Strategy Matching:**
   - Exact: 1.0 confidence for identical names
   - Fuzzy: Levenshtein similarity (threshold: 0.69)
   - Semantic: Word overlap + prefix (threshold: 0.6)
   - LLM: GPT-4o batch inference (threshold: 0.5)

2. **Penalties:**
   - Generic field penalty: 0.5x (metadata, memo, blob)
   - Type mismatch penalty: 0.6x (structured → unstructured)

3. **Judge System:**
   - Detects overloaded fields (>2 mappings to same field)
   - Identifies identical confidence scores
   - Flags excessive generic mappings
   - Triggers refinement when quality issues found

4. **Refinement:**
   - Removes low-confidence mappings to overloaded fields
   - Threshold: 0.72 for keeping suspicious mappings
   - Logs all removed mappings for audit

5. **Validation:**
   - Coverage percentage calculation
   - Confidence level determination (high/medium/low)
   - Unmapped column tracking
   - Data type compatibility checking

**Configuration:**
```python
exact_match_threshold = 1.0
fuzzy_match_threshold = 0.69
semantic_match_threshold = 0.6
llm_match_threshold = 0.5
generic_field_penalty = 0.5
type_mismatch_penalty = 0.6
refinement_confidence_threshold = 0.72
min_confidence_threshold = 0.5
min_columns_per_table = 1
```

### 6.8 Resilient Clients

**Files:**
- `Agent/services/clients/openai_client.py`
- `Agent/services/clients/qdrant_client.py`

**Purpose:** Wrap external APIs with resilience patterns.

**Features:**

1. **Retry Logic:**
   - Exponential backoff (base: 1s, max: 60s)
   - Jitter to prevent thundering herd
   - Configurable max retries (default: 3)

2. **Circuit Breaker:**
   - Opens after N consecutive failures (threshold: 5)
   - Half-open state for recovery testing
   - Automatic reset after timeout (default: 60s)

3. **Error Handling:**
   - Distinguishes transient vs permanent errors
   - Logs all failures with context
   - Raises informative exceptions

4. **Rate Limiting:**
   - Respects API rate limits
   - Backs off on 429 responses
   - Prevents cascading failures

---

## 7. Workflow Execution

### 7.1 Execution Flow

**Entry Point:** `Agent/main.py`

**CLI Usage:**
```bash
python -m Agent.main <file_path> [options]

Options:
  --context TEXT       Optional context about the data
  --table TEXT         Optional preferred table name
  --log-level TEXT     Logging level (DEBUG, INFO, WARNING, ERROR)
  --max-file-size INT  Maximum file size in MB (default: 100)
```

**Programmatic Usage:**
```python
from Agent.main import run_workflow

result = run_workflow(
    file_path="data/opportunities.csv",
    user_context="Sales pipeline data from Q3 2025",
    table_preference="Opportunity",
    log_level="INFO"
)

if result['workflow_status'] == 'completed':
    mapping = result['field_mapping_result']
    # Process mapping...
```

### 7.2 Step-by-Step Execution

#### Step 1: Initialization
```python
engine = WorkflowEngine(use_checkpointer=True)
engine.initialize()  # Builds LangGraph
```

#### Step 2: State Creation
```python
initial_state = {
    "file_path": validated_file_path,
    "user_context": sanitized_context,
    "table_preference": table_preference,
    "workflow_step": "start",
    "workflow_status": "in_progress",
    "steps_completed": [],
    "messages": [],
    "errors": []
}
```

#### Step 3: Workflow Execution
```python
config = {
    "configurable": {
        "thread_id": f"workflow_{sanitized_path}"
    }
}

for step_output in graph.stream(initial_state, config):
    for node_name, node_state in step_output.items():
        logging.info(f"Completed node: {node_name}")
        final_state = node_state
```

#### Step 4: Result Processing
```python
if final_state['workflow_status'] == 'completed':
    # Success: use mappings
    mapping_result = final_state['field_mapping_result']
    
elif final_state['workflow_status'] == 'requires_review':
    # Review needed: flag for human
    review_issues = final_state['field_mapping_result'].validation.issues
    
else:
    # Failed: log and handle
    error = final_state['last_error']
```

### 7.3 State Transitions

```
Initial State
    └─> file_analysis
        ├─> file_analysis_result populated
        ├─> workflow_step = "file_analysis"
        └─> steps_completed += ["file_analysis"]

    └─> rag_matching
        ├─> rag_match_result populated
        ├─> workflow_step = "rag_matching"
        └─> steps_completed += ["rag_matching"]

    └─> table_selection
        ├─> selected_table populated
        ├─> selected_table_metadata populated
        ├─> workflow_step = "table_selection"
        └─> steps_completed += ["table_selection"]

    └─> field_mapping
        ├─> field_mapping_result populated
        ├─> workflow_step = "field_mapping"
        └─> steps_completed += ["field_mapping"]

    └─> end (or review/error)
        └─> workflow_status = "completed" (or "requires_review"/"failed")
```

### 7.4 Error Handling

**Error Propagation:**
1. Exception occurs in node
2. Node catches exception, logs details
3. Node updates state with error information
4. Routing function detects error, routes to error node
5. Error node logs comprehensive error details
6. Workflow ends with failed status

**Error State Fields:**
```python
state['last_error'] = error_message
state['errors'].append(error_message)
state['workflow_step'] = 'error'
state['workflow_status'] = 'failed'
```

**Recovery:**
- Checkpointing enables resumption from last successful step
- Thread ID allows retrieving persisted state
- Manual intervention can modify state and continue

---

## 8. Data Models

### 8.1 File Analysis Models

**File:** `Agent/models/file_analysis_model.py`

```python
class FileStructureInfo(BaseModel):
    file_name: str
    file_type: str  # csv, excel, json
    file_path: str
    file_size_bytes: Optional[int]
    total_rows: int
    total_columns: int

class ColumnMetadata(BaseModel):
    name: str                    # Original column name
    english_name: str            # Translated name
    translation_used: bool       # Whether translation applied
    data_type: str              # Inferred data type
    max_length: Optional[int]   # For string columns
    null_count: int
    unique_count: int
    sample_values: List[Any]

class DataQualityMetrics(BaseModel):
    total_null_values: int
    null_percentage: float
    duplicate_rows: int
    potential_issues: List[str]

class FileAnalysisResult(BaseModel):
    structure: FileStructureInfo
    columns: List[ColumnMetadata]
    quality_metrics: DataQualityMetrics
    sample_data: List[dict]
    delimiter: str
    encoding: str
    has_header: bool
    content_preview: str
    analysis_success: bool
    analysis_timestamp: str
    error_message: Optional[str]
```

### 8.2 RAG Matching Models

**File:** `Agent/models/rag_match_model.py`

```python
class TableMatch(BaseModel):
    table_name: str
    schema_name: Optional[str]
    similarity_score: float      # 0.0 to 1.0
    confidence: str              # high/medium/low
    matching_columns: List[str]
    reason: str
    metadata: dict               # table_code, fields, etc.

class TableMatchResult(BaseModel):
    matched_tables: List[TableMatch]
    search_query: str
    total_candidates: int

class FieldMapping(BaseModel):
    source_column: str
    source_column_english: str
    target_column: str
    confidence_score: float
    match_type: str              # exact/fuzzy/semantic/llm
    data_type_compatible: bool
    source_data_type: str
    target_data_type: Optional[str]
    sample_source_values: List[Any]

class MappingValidationResult(BaseModel):
    is_valid: bool
    confidence_level: str
    total_mappings: int
    mapped_count: int
    unmapped_source_columns: List[str]
    unmapped_target_columns: List[str]
    issues: List[str]
    warnings: List[str]
    mapping_coverage_percent: float
    requires_review: bool

class TableFieldMapping(BaseModel):
    table_name: str
    table_type: str              # Entity/Relation
    mappings: List[FieldMapping]
    validation: MappingValidationResult
    confidence: float
    insertion_order: int         # 1=Entity, 2=Relation

class MultiTableMappingResult(BaseModel):
    source_file: str
    total_source_columns: int
    table_mappings: List[TableFieldMapping]
    overall_coverage: float
    overall_confidence: str
    unmapped_columns: List[str]
    is_valid: bool
    requires_review: bool
    requires_refinement: bool    # Internal flag from Judge
```

### 8.3 Validation Models

**File:** `Agent/models/validators.py`

```python
class PipelineInput(BaseModel):
    """Validates all workflow inputs."""
    file_path: str
    user_context: Optional[str]
    workspace_root: str
    max_file_size_mb: int = 100
    
    @field_validator('file_path')
    def validate_file_path(cls, v, info):
        # Prevents path traversal attacks
        # Validates file exists and is readable
        # Checks file size limits

class UserContextInput(BaseModel):
    """Sanitizes user context to prevent injection."""
    raw_context: str
    max_length: int = 1000
    
    def get_sanitized(self) -> str:
        # Removes HTML/SQL injection patterns
        # Strips control characters
        # Truncates to max length
```

---

## 9. Configuration & Setup

### 9.1 Prerequisites

**System Requirements:**
- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- Internet connection for API access

**External Services:**
1. **OpenAI Account:**
   - API key with GPT-4o access
   - Sufficient credits for embeddings and completions

2. **Qdrant Instance:**
   - Cloud or self-hosted
   - Collection created for schema storage

3. **Efficy CRM Access (optional):**
   - DICO API credentials
   - Read access to database schema

### 9.2 Installation

**Step 1: Clone Repository**
```bash
git clone <repository-url>
cd rules-engine-agent
```

**Step 2: Create Virtual Environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Required Packages:**
```
langchain>=0.1.0
langgraph>=0.1.0
langchain-openai>=0.1.0
qdrant-client>=1.7.0
pydantic>=2.0.0
pandas>=2.0.0
python-dotenv>=1.0.0
requests>=2.31.0
```

### 9.3 Environment Configuration

**Create .env File:**
```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...your-key...

# Qdrant Configuration
QDRANT_URL=https://your-instance.qdrant.io
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION_NAME=database_schema_v1

# LLM Configuration
LLM_MODEL=gpt-4o
TEMPERATURE=0.1

# RAG Configuration
TOP_K_RESULTS=10
MAX_REFINEMENTS=3

# Efficy Configuration (Optional)
EFFICY_CUSTOMER=YOUR_CUSTOMER
EFFICY_BASE_URL=https://your-instance.efficy.cloud
EFFICY_USER=your-username
EFFICY_PASSWORD=your-password
```

**Settings Validation:**
The application validates all settings on startup using Pydantic:
```python
from Agent.config import get_settings

settings = get_settings()  # Raises ValueError if invalid
```

### 9.4 Vector Store Setup

**Option 1: Use Existing Vector Store**
If collection already exists, set `QDRANT_COLLECTION_NAME` in `.env`.

**Option 2: Build Vector Store from DICO**
```bash
python -m Agent.rag.pipeline feed
```

This will:
1. Fetch database schema from DICO API
2. Generate embeddings for all tables and fields
3. Create Qdrant collection with metadata
4. Index all schema information

**Collection Schema:**
```python
collection_config = {
    "vectors": {
        "size": 1536,  # OpenAI embedding dimension
        "distance": "Cosine"
    },
    "payload_schema": {
        "table_name": "keyword",
        "table_code": "keyword",
        "table_kind": "keyword",  # Entity or Relation
        "content": "text",
        "fields": "text"
    }
}
```

### 9.5 Running Tests

**Unit Tests:**
```bash
pytest Agent/tests/ -v
```

**Specific Test Files:**
```bash
pytest Agent/tests/test_workflow_engine.py
pytest Agent/tests/test_field_mapping.py
pytest Agent/tests/test_multi_table_mapping.py
```

**Test Coverage:**
```bash
pytest --cov=Agent --cov-report=html
```

---

## 10. Development Guidelines

### 10.1 Code Style

**Follow PEP 8:**
- 4 spaces for indentation
- Max line length: 100 characters
- Use descriptive variable names
- Add docstrings to all functions/classes

**Type Hints:**
```python
def process_file(file_path: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Process file with optional context."""
    pass
```

**Pydantic Models:**
- Use for all data structures
- Enable validation and serialization
- Document fields with descriptions

### 10.2 Logging Standards

**Use Structured Logging:**
```python
import logging

logging.info(f"Processing file: {file_path}")
logging.warning(f"Low confidence mapping: {column} -> {field}")
logging.error(f"API call failed: {error}", exc_info=True)
```

**Log Levels:**
- **DEBUG:** Detailed diagnostic information
- **INFO:** Workflow progress, key decisions
- **WARNING:** Issues that don't stop execution
- **ERROR:** Failures requiring attention

**Sensitive Data:**
- Never log API keys or passwords
- Sanitize user input before logging
- Use `sanitize_for_logging()` utility

### 10.3 Error Handling

**Exception Patterns:**
```python
try:
    result = risky_operation()
except SpecificException as e:
    logging.error(f"Operation failed: {e}", exc_info=True)
    return create_error_response(str(e))
except Exception as e:
    logging.critical(f"Unexpected error: {e}", exc_info=True)
    raise
```

**Node Error Handling:**
```python
def my_node(state: WorkflowState) -> WorkflowState:
    try:
        # Node logic
        result = process(state)
        return {**state, "result": result}
    except Exception as e:
        error_msg = f"Node failed: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return {
            **state,
            "last_error": error_msg,
            "errors": state.get("errors", []) + [error_msg],
            "workflow_step": "error",
            "workflow_status": "failed"
        }
```

### 10.4 Testing Best Practices

**Unit Tests:**
- Test individual functions in isolation
- Mock external dependencies (OpenAI, Qdrant)
- Use fixtures for test data
- Assert on expected outputs

**Integration Tests:**
- Test full workflow execution
- Use test CSV files with known mappings
- Validate end-to-end results
- Check error handling paths

**Test Data:**
- Store in `Agent/tests/fixtures/`
- Use small CSV files (<100 rows)
- Cover edge cases (empty columns, special characters)

### 10.5 Adding New Nodes

**Steps to Add a Node:**

1. **Create Node Function:**
```python
# Agent/nodes/my_new_node.py
from ..models.workflow_state import WorkflowState

def my_new_node(state: WorkflowState) -> WorkflowState:
    """New processing step."""
    logging.info("Executing my_new_node...")
    
    try:
        # Node logic here
        result = process_something(state)
        
        return {
            **state,
            "my_result": result,
            "workflow_step": "my_new_step",
            "steps_completed": state.get("steps_completed", []) + ["my_new_step"]
        }
    except Exception as e:
        # Error handling
        pass
```

2. **Add to Graph Builder:**
```python
# Agent/core/graph_builder.py
from ..nodes.my_new_node import my_new_node

def build(self):
    graph = StateGraph(WorkflowState)
    
    # Add new node
    graph.add_node("my_new_step", my_new_node)
    
    # Add edges
    graph.add_conditional_edges(
        "previous_step",
        route_to_new_step,
        {"my_new_step": "my_new_step", "error": "error"}
    )
```

3. **Create Routing Logic:**
```python
# Agent/routing/routing_logic.py
def route_to_new_step(state: WorkflowState) -> Literal["my_new_step", "error"]:
    """Route to new step or error."""
    if state.get("workflow_step") == "error":
        return "error"
    if not state.get("required_field"):
        return "error"
    return "my_new_step"
```

4. **Update State Model:**
```python
# Agent/models/workflow_state.py
class WorkflowState(TypedDict, total=False):
    # ... existing fields ...
    my_result: Optional[MyResultType]
```

5. **Write Tests:**
```python
# Agent/tests/test_my_new_node.py
def test_my_new_node():
    state = {"required_field": "value"}
    result = my_new_node(state)
    assert result["my_result"] is not None
```

### 10.6 Performance Optimization

**Caching:**
- Cache OpenAI embeddings for repeated queries
- Cache file analysis results for retries
- Use LangGraph checkpointing to avoid reprocessing

**Batching:**
- Batch LLM calls for multiple columns
- Use batch embedding generation
- Group database queries

**Concurrency:**
- Use async/await for I/O operations
- Parallel processing for independent columns
- Connection pooling for database access

---

## 11. Testing Strategy

### 11.1 Test Coverage

**Current Test Files:**
- `test_workflow_engine.py`: End-to-end workflow tests
- `test_file_analysis.py`: CSV parsing and translation
- `test_rag_matching.py`: Vector search and ranking
- `test_table_selection.py`: Selection logic
- `test_field_mapping.py`: Single-table mapping
- `test_multi_table_mapping.py`: Multi-table mapping
- `test_validators.py`: Input validation and sanitization

### 11.2 Test Data

**Sample CSV Files:**
Located in project root for testing:
- `oppo_combi.csv`: Opportunity data with relationships
- `cont_combi_full.csv`: Contact data
- `combi_full.csv`: Multi-entity data
- `Mails.csv`: Email data for relationship testing

### 11.3 Mocking External Services

**OpenAI Mocking:**
```python
from unittest.mock import Mock, patch

@patch('Agent.services.clients.openai_client.OpenAI')
def test_llm_mapping(mock_openai):
    mock_openai.return_value.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content='{"matches": []}'))]
    )
    # Test logic
```

**Qdrant Mocking:**
```python
@patch('Agent.services.clients.qdrant_client.QdrantClient')
def test_vector_search(mock_qdrant):
    mock_qdrant.return_value.search.return_value = [
        Mock(payload={"table_name": "Opportunity"}, score=0.9)
    ]
    # Test logic
```

---

## 12. Troubleshooting

### 12.1 Common Issues

#### Issue: "Collection does not exist" Error

**Cause:** Qdrant collection not initialized.

**Solution:**
```bash
python -m Agent.rag.pipeline feed
```

#### Issue: OpenAI Rate Limit Errors

**Cause:** Too many API calls too quickly.

**Solution:**
- Increase retry delays in `resilient_client.py`
- Reduce `max_retries` to fail faster
- Use lower-tier model for development

#### Issue: Low Mapping Coverage

**Cause:** Vector database doesn't contain relevant tables.

**Solution:**
- Verify DICO fetch included all tables
- Check table_kind filtering isn't too restrictive
- Lower confidence thresholds temporarily

#### Issue: Workflow Hangs

**Cause:** Infinite loop or blocking operation.

**Solution:**
- Check logs for last completed step
- Review routing logic for circular paths
- Add timeout to LLM calls

### 12.2 Debug Mode

**Enable Detailed Logging:**
```bash
python -m Agent.main <file> --log-level DEBUG
```

**Inspect State at Each Step:**
```python
for step_output in graph.stream(initial_state, config):
    for node_name, node_state in step_output.items():
        print(f"Node: {node_name}")
        print(f"State: {json.dumps(node_state, indent=2, default=str)}")
```

### 12.3 Performance Issues

**Slow File Analysis:**
- Check CSV encoding detection
- Reduce sample data size
- Disable translation for English columns

**Slow Vector Search:**
- Reduce `top_k_results` setting
- Optimize Qdrant queries with filters
- Use smaller embedding model

**Slow LLM Mapping:**
- Reduce columns sent to LLM
- Use batch calls instead of individual
- Lower temperature for faster inference

---

## 13. API Reference

### 13.1 Main Entry Points

#### run_workflow()

```python
def run_workflow(
    file_path: str,
    user_context: Optional[str] = None,
    table_preference: Optional[str] = None,
    log_level: str = "INFO",
    max_file_size_mb: int = 100
) -> dict
```

**Description:** Execute complete mapping workflow for a CSV file.

**Parameters:**
- `file_path` (str): Absolute or relative path to CSV file
- `user_context` (Optional[str]): Business context about the data
- `table_preference` (Optional[str]): Preferred table name for mapping
- `log_level` (str): Logging verbosity (DEBUG/INFO/WARNING/ERROR)
- `max_file_size_mb` (int): Maximum file size limit

**Returns:**
- `dict`: Final workflow state with mapping results

**Raises:**
- `ValueError`: Invalid input or security violation
- `FileNotFoundError`: File does not exist

**Example:**
```python
result = run_workflow(
    file_path="data/opportunities.csv",
    user_context="Q3 2025 sales pipeline",
    table_preference="Opportunity",
    log_level="INFO"
)

if result['workflow_status'] == 'completed':
    mapping = result['field_mapping_result']
    for table_mapping in mapping.table_mappings:
        print(f"Table: {table_mapping.table_name}")
        for field in table_mapping.mappings:
            print(f"  {field.source_column} -> {field.target_column}")
```

### 13.2 WorkflowEngine API

#### WorkflowEngine.run()

```python
def run(
    self,
    file_path: str,
    user_context: Optional[str] = None,
    table_preference: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]
```

**Description:** Execute workflow with initialized engine.

**Parameters:**
- `file_path` (str): Path to CSV file
- `user_context` (Optional[str]): User-provided context
- `table_preference` (Optional[str]): Preferred table name
- `**kwargs`: Additional workflow parameters

**Returns:**
- `Dict[str, Any]`: Final workflow state as dictionary

### 13.3 Service APIs

#### FileAnalyzerService.analyze()

```python
def analyze(self, file_path: str) -> FileAnalysisResult
```

**Description:** Analyze CSV file structure and translate columns.

**Returns:** `FileAnalysisResult` with structure, columns, quality metrics

#### TableMatcherService.find_matching_tables()

```python
def find_matching_tables(
    self,
    file_analysis: FileAnalysisResult,
    user_context: Optional[str] = None
) -> TableMatchResult
```

**Description:** Find database tables matching the file using RAG.

**Returns:** `TableMatchResult` with ranked candidate tables

#### Mapper.map_to_multiple_tables()

```python
def map_to_multiple_tables(
    self,
    file_analysis: FileAnalysisResult,
    candidate_tables: List[Dict],
    primary_table: str = None,
    max_tables: int = 5
) -> MultiTableMappingResult
```

**Description:** Map CSV columns to multiple database tables.

**Returns:** `MultiTableMappingResult` with mappings per table

### 13.4 Configuration API

#### get_settings()

```python
def get_settings(env_file: Optional[Path] = None) -> Settings
```

**Description:** Get validated settings instance (singleton).

**Parameters:**
- `env_file` (Optional[Path]): Path to .env file

**Returns:** `Settings` object with all configuration

**Example:**
```python
from Agent.config import get_settings

settings = get_settings()
print(f"Using model: {settings.llm_model}")
print(f"Qdrant URL: {settings.qdrant_url}")
```

---

## Appendix A: Glossary

**Agent:** The automated system that orchestrates the mapping workflow.

**Circuit Breaker:** Resilience pattern that prevents cascading failures by temporarily blocking failing operations.

**Confidence Score:** Numeric value (0-1) indicating mapping certainty.

**Entity Table:** Database table representing a core business object (Opportunity, Contact).

**Field Mapping:** Association between a CSV column and a database field.

**Judge:** Internal system that evaluates mapping quality and decides if refinement is needed.

**LangGraph:** Framework for building stateful, multi-step workflows with LLMs.

**Node:** Single processing step in the workflow graph.

**RAG (Retrieval-Augmented Generation):** Technique combining vector search with LLM generation.

**Refinement:** Process of removing suspicious or low-quality mappings.

**Relation Table:** Database table representing relationships between entities.

**Routing Logic:** Conditional functions that determine workflow paths.

**State Machine:** Workflow model where each step has defined inputs, outputs, and transitions.

**Vector Embedding:** Numeric representation of text enabling semantic similarity search.

**WorkflowState:** TypedDict containing all data passed through the workflow.

---

## Appendix B: FAQ

**Q: Why Entity-First architecture?**  
A: Prioritizing entities ensures the primary business objects are identified first, then relationships are discovered in context. This improves accuracy for mixed-entity data.

**Q: Why not use a single LLM call for everything?**  
A: Multi-strategy approach (exact/fuzzy/semantic/LLM) balances speed, cost, and accuracy. Simple matches use fast algorithms; complex cases use expensive LLM calls.

**Q: What's the difference between Judge and Validation?**  
A: Judge analyzes if refinement is needed (internal decision). Validation calculates coverage and confidence (external metrics).

**Q: Can I add custom matching strategies?**  
A: Yes. Add methods to `Mapper` class and call them in `_map_columns_to_all_tables()`.

**Q: How do I handle files with >100 columns?**  
A: Increase `max_file_size_mb` parameter. Consider batching columns or pre-filtering irrelevant ones.

**Q: Can this work with databases other than Efficy?**  
A: Yes. Replace DICO client with your schema fetcher. Ensure metadata includes table_name, fields, and table_kind.

**Q: How do I improve mapping accuracy?**  
A: 1) Enrich vector store with more schema details, 2) Provide user context, 3) Lower confidence thresholds, 4) Add domain-specific rules.

**Q: What happens if the workflow crashes mid-execution?**  
A: LangGraph checkpointing saves state at each step. Use the same thread_id to resume from last checkpoint.

---

## Appendix C: Change Log

**Version 1.0 (November 7, 2025)**
- Initial production release
- Entity-First RAG architecture
- Multi-table mapping with Judge and Refinement
- Resilient clients with circuit breaker
- Comprehensive logging and validation
- CLI and programmatic interfaces

---

## Appendix D: Contact & Support

**Project Repository:** [Insert repository URL]

**Issue Tracking:** [Insert issue tracker URL]

**Documentation Updates:** This document should be updated whenever significant architectural or functional changes occur.

**For Questions:**
- Technical issues: Create GitHub issue
- Architecture discussions: Contact engineering team
- Product feedback: Contact product management

---

