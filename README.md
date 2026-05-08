---

## System Architecture Overview

This system is a **modular Flask-based RAG pipeline** for PDF documents containing both text and embedded JSON data visualizations. It follows a clean separation-of-concerns pattern with 16 modules. Here's the complete breakdown:

---

## 1. **Configuration Layer** (`config.py`)

### Core Settings
| Category | Key Values | Purpose |
|----------|-----------|---------|
| **Hardware** | `DEVICE` = cuda/cpu, `EMBEDDING_MODEL` = BAAI/bge-base-en-v1.5 | BGE model with query prefixing convention |
| **Hybrid Retrieval** | Dense 0.65 / Sparse 0.35 | FAISS + BM25 fusion weights |
| **Metadata Rerank** | Section 0.50 / Tag 0.30 / Dense 0.20 | Three-signal re-ranking |
| **Thresholds** | `SECTION_THRESHOLD=0.60`, `COMBINED_THRESHOLD=0.70` | Two-gate viz matching |
| **Item Retrieval** | `ITEM_COSINE_THRESHOLD=0.50` | Minimum combined score for item hits |
| **Condition Filter** | `CONDITION_FILTER_THRESHOLD=0.42` | Semantic gate for filter queries |

### Critical Design Decision: `_FIELD_ALIASES`
The field alias map  to software development terminology**:
- `"full stack"` → `["full_stack_finished", "fullstack"]`
- `"ios"` → `["ios_finished", "ios"]`
- `"backend"` → `["backend_finished", "backend"]`
- `"mvp"` → `["ios_finished", "ios"]` (maps to iOS completion)
- `"cicd"`/`"ci/cd"` → `["full_stack_finished", "fullstack"]`


---

## 2. **Vector Math Engine** (`vector_math.py`)

### BGE Embedding Convention
```python
def embed(text: str) -> list[float]:
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return get_embed_model().embed_query(prefixed)
```

**Critical:** BGE models require this exact prefix for query embedding. Document embedding (`embed_doc`) skips the prefix — this asymmetry is intentional and model-specific.

### Singleton Pattern
The `_embed_model` is lazily initialized once and shared across all modules. This prevents reloading the 438MB BGE-base model on every request.

---

## 3. **PDF Parsing Pipeline** (`pdf_parsing.py`)

### Three-Stage Extraction

```
Raw PDF → PyPDF2 text extraction → OCR repair → JSON block detection → 
    ├─ JSON blocks → metadata extraction → full_json_store population
    └─ Text segments → CharacterTextSplitter → FAISS chunks
```

### JSON Block Detection (`extract_json_blocks_with_positions`)
Uses a **bracket-depth parser** with string-escape awareness:
- Tracks `{`/`}` and `[`/`]` depth
- Handles nested structures
- Skips strings (won't be fooled by `{"key": "}"}`)
- Returns `(start, end, sanitized_text, parsed_obj)` tuples

### Section Heading Extraction (`extract_section_heading`)
Two-tier fallback:
1. **Embedded meta**: If JSON has `{"meta": {"section": "..."}}`, use that
2. **Heuristic scan**: Look backward up to 300 chars for keywords like "Table", "Chart", "Summary", "Analysis", etc.

### Viz Type Normalization (`get_chunks`)
Massive alias map (`_VIZ_ALIAS`) handles:
- Bar family: `bar`, `column`, `grouped_bar`, `stacked_bar`, `horizontal_bar`
- Line family: `line`, `multi_line`
- Area family: `area`, `stacked_area`
- Pie/Donut: `pie`, `donut`, `semi_donut`
- Statistical: `histogram`, `density`, `box_plot`, `frequency_polygon`
- Other: `scatter`, `bubble`, `radar`, `table`

Each JSON block gets a UUID (`json_block_id`) and is stored in `full_json_store` with:
- `raw_json`: serialized data payload
- `viz_type`: normalized chart type
- `section`: section heading
- `tags`: embedded tags
- `source`: PDF filename

### Chunking Strategy
- **Text docs**: `CharacterTextSplitter` (1500 chars, 300 overlap)
- **JSON docs**: Custom `custom_json_splitter` (1000 byte max per chunk)
  - Preserves header: `Section: X | Tags: Y | Source: Z`
  - Metadata carries `full_json_id` for reconstruction

---

## 4. **Retrieval System** (`retrieval.py`)

### Dual-Mode Retrieval with 3 Stages

| Stage | Condition | Action |
|-------|-----------|--------|
| **1** | No chat history | Raw similarity search |
| **2** | Topic switch detected (`sim < 0.45`) | Raw search (no history contamination) |
| **3a** | Ambiguous tokens ("it", "they", "that") | Condense with LLM → search condensed |
| **3b** | Self-contained same-topic | Raw search |

### Topic Switch Detection
```python
def _is_topic_switch(current_q, prev_q):
    sim = _cosine(embed(current_q), embed(prev_q))
    return sim < TOPIC_SHIFT_THRESHOLD  # 0.45
```

**Insight:** Uses BGE embeddings on raw questions. A switch from "show Latin America" to "what about Europe?" will have low cosine → detected as topic switch → skips history condensing.

### History Condenser
Uses the LLM to rephrase follow-ups as standalone questions:
```
"How many?" → "How many projects are in LATAM North"
```

### Hybrid Variant (`hybrid_dual_mode_retrieval`)
Drop-in replacement that swaps `vectorstore.similarity_search()` with `hybrid_similarity_search()` from `hybrid_retrieval.py`. Preserves all stage logic.

---

## 5. **Hybrid Retrieval Engine** (`hybrid_retrieval.py`)

### BM25 + FAISS Fusion (Reciprocal Rank Fusion variant)

```
Dense:  FAISS similarity_search_with_score → L2 distance → sim = 1/(1+dist)
Sparse: BM25Okapi.get_scores(query_tokens) → normalize by max

Final:  0.65 * dense_norm + 0.35 * bm25_norm
```

### Critical Implementation Detail
```python
chunk_by_content: dict[str, int] = {
    c.page_content[:120]: i for i, c in enumerate(chunks)
}
```

Maps dense results back to chunk indices using **first 120 chars as key**. This is fragile if two chunks share the same prefix, but works for the typical case.

### Fallback
If `rank_bm25` not installed → pure dense retrieval.

---

## 6. **Metadata Reranking** (`metadata_rerank.py`)

### Three-Signal Composite Score

```
score = 0.50 * section_cosine + 0.30 * tag_cosine + 0.20 * content_cosine
```

- **Section**: Embeds section heading, compares to query
- **Tags**: Joins tags into sentence, embeds, compares
- **Content**: First 512 chars of doc content, embedded as document (no query prefix)

**Key behavior:** Re-orders docs but **never removes** any. Caps at 6 docs. This means a doc with terrible content but perfect section match can still surface.

---

## 7. **Item-Level Retrieval** (`item_retrieval.py`)

### Two-Layer Specificity-Aware Matching

**Purpose:** Find a single row/record when user asks "details of Project X" or "show me Goli".

#### Layer 1: Token/Phrase Match (Precision Signal)
Four-tier scoring:

| Tier | Condition | Score |
|------|-----------|-------|
| 1 | Exact phrase match in cell values | 1.0 |
| 2 | All significant query tokens match + high specificity | 0.85 × (0.5 + 0.5×spec) |
| 3 | Partial token coverage | coverage × 0.70 × (0.4 + 0.6×spec) |
| 4 | No match | 0.0 |

**Specificity calculation:** `len(token) / len(cell_value_without_spaces)`
- "projects" in "IntentProjects" → 8/14 = 57% specificity → penalized
- "projects" in "Projects" → 8/8 = 100% specificity → rewarded

#### Layer 2: BGE Cosine (Semantic Signal)
Embeds context-enriched row string: `"Section: X Field1: Value1 Field2: Value2..."`

#### Combined Score
```
final = 0.45 * cosine + 0.55 * match_score
```

**Threshold:** `ITEM_COSINE_THRESHOLD = 0.50`

---

## 8. **Condition Filter Engine** (`condition_filter.py`)

### Hardcoded Rule-Based Filter (Fallback Path)

#### Detection Gate (`detect_condition_query`)
Requires:
1. At least one condition keyword rule fires (e.g., "100%", "active", ">50")
2. AND (trigger token like "list"/"show" OR field alias like "nt"/"full bible")

#### Condition Resolution (`resolve_conditions`)
Two-phase parsing:

**Phase 1: Field Hints**
- Scans `_FIELD_ALIASES` for word-boundary matches
- Longest-alias-first ordering ("new testament" beats "nt")
- Deduplicates hints while preserving order

**Phase 2: Condition Rules**
- Regex patterns for: 100%, 0%, active, inactive, >X%, <X%, between, etc.
- Captures numeric values from patterns like "more than 50%"
- Deduplicates by (type, value) to prevent "100%" + "finished" both firing

#### Field Resolution (`_resolve_field_in_row`)
Priority-based matching against actual row keys:
1. Exact normalized match (`full_bible` == `full_bible`)
2. Hint contained in key (`bible` in `full_bible_finished`)
3. Key contained in hint

**Guard:** Hints ≤2 chars only match via exact equality (prevents "ot" matching "not")

#### "Honest Zero Results" Behavior
**Critical fix:** If user provides explicit field alias (e.g., "OT"), the system commits to that field in the best-matching block. If 0 rows match, returns `no_results=True` **immediately** rather than falling through to other blocks where a different field might accidentally match.

This prevents the bug where "OT 100%" would return NT=100% rows because NT happened to match first.

---

## 9. **LLM Condition Filter** (`llm_condition_filter.py`)

### ContextAwareJSONFilter Bridge (Primary Path)

#### Architecture
Wraps `condition_based_filter_new.ContextAwareJSONFilter` as a lazy singleton:
```python
_cajf_singleton: "_CAJFClass | None" = None

def _get_cajf():
    if _cajf_singleton is None:
        _cajf_singleton = _CAJFClass()
    return _cajf_singleton
```

#### Detection Gate (`detect_condition_query_llm`)
Three triggers:
1. `_LLM_FILTER_TRIGGER_TOKENS` ∩ query words (e.g., "list", "show", "filter")
2. `_FIELD_ALIASES` substring match (e.g., "nt", "full bible")
3. Numeric comparison pattern: `([><=]|under|above|between|over|below|than)\s*[\d.]+`

Plus: Must have at least one JSON block with list-of-dicts data.

#### Block Ranking (Modified from Hardcoded Path)
```python
score = sec_score * 0.35 + tag_score * 0.45 + data_boost * 0.20
```

**Data boost:** Scans first 2000 chars of raw JSON for query words. If found, adds 0.35. This is **critical for specific items** (Category D queries like "Goli, Liga, Nila") that don't match section headings.

#### Result Building
- **Table type:** Uses `_filtered_rows_to_html()` (styled table with percentage bars, status badges)
- **Chart type:** Builds `data-viz-json` div for frontend rendering
- **Zero results:** Styled "No results found" notice

---

## 10. **LLM Filter Core** (`condition_based_filter_new.py`)

### ContextAwareJSONFilter — The "Brain"

#### Initialization
- **LlamaIndex**: `HuggingFaceEmbedding` + `Ollama` (llama3.2)
- **LangChain**: `OllamaLLM` + `PydanticOutputParser` for structured output
- **Schema Index**: `VectorStoreIndex` built from column documents for RAG-augmented filter generation

#### Processing Pipeline
```
process_query(input_json, query)
    ├─ _extract_context() → schema, columns, dynamic hints
    ├─ _apply_advanced_logic() → highest/lowest/topN/full view (Python)
    ├─ _dynamic_ground_query() → value-to-column mapping, OR detection, range detection
    ├─ _extract_numeric_conditions() → explicit comparisons from text
    ├─ _extract_semantic_conditions() → "half completed" → >=40 AND <=60
    ├─ _build_deterministic_filter() → high-confidence direct filter
    ├─ _generate_filter_json() → LLM with 3-layer fallback parsing
    ├─ _validate_and_repair_filter() → 5 fix layers
    └─ _apply_jmespath_filter() → execute
```

#### Advanced Python Logic (Pre-LLM)
Handles queries that don't need LLM:
- **Full view**: "show all", "full table" (short, broad, no comparisons)
- **Top N / Bottom N**: "top 5", "bottom 3"
- **Highest/Lowest**: Returns all rows sharing the extreme value

#### Dynamic Ground Query
**Value-to-Column Mapping:**
- Scans all column values (up to 2000 rows) against query text
- Longest-match-first to avoid sub-phrase overlap
- Builds `col_value_map`: `{column: [value1, value2]}`

**OR Logic Detection:**
- Explicit "or"/"either"/"any of" → `logic: "OR"`
- Multiple values in same column → `logic: "OR"` (via `in` operator)

**Range Detection:**
- "between X and Y" → two conditions with AND
- "X-Y%" → same
- Stored in `range_info_struct` for validation layer

#### LLM Prompt Engineering
The system prompt is ~40 lines with explicit rules:

| Rule Category | Examples |
|--------------|----------|
| **Operators** | `==`, `!=`, `contains`, `in`, `>`, `<`, `>=`, `<=` |
| **Semantic mappings** | "Incomplete" → `< 100`, "half done" → `40-60`, "nearly finished" → `≥ 90` |
| **Anti-hallucination** | "DO NOT pick specific project names from sample data unless mentioned in query" |
| **Structural** | "Every distinct requirement MUST be its own condition" |
| **Range** | "Create TWO conditions with AND logic for ranges" |

#### 3-Layer Fallback Parsing
1. **PydanticOutputParser** (strictest) → `FilterLogic` model
2. **Manual JSON extraction** → brace-matching, code block stripping
3. **Pydantic direct construction** → validate then serialize

If all fail → regex extraction as last resort.

#### 5-Fix Validation Layer (`_validate_and_repair_filter`)

| Fix | Condition | Action |
|-----|-----------|--------|
| **0** | Impossible AND on same column (e.g., `Projects=='A' AND Projects=='B'`) | Convert to `in` operator |
| **1** | OR logic with multiple values, LLM missed some | Add `in` condition with all values |
| **2** | Range detected but missing bound | Add `>= low` or `<= high` |
| **3** | Explicit numeric comparison dropped | Add/fix on correct column, remove wrong-column conflicts |
| **4** | Semantic condition missing | Add from `_extract_semantic_conditions` |
| **5** | Single value from `col_value_map` missing | Add `==` condition |

#### JMESPath Filtering
Converts filter JSON to JMESPath expressions:
- `==` string: `[?col=='val']`
- `==` numeric: `[?col==\`val\`]`
- `contains`: `[?contains(col, 'val')]`
- `in`: `[?contains(\`["val1","val2"]\`, col)]`

**Fallback to Python** for: `>`, `<`, `>=`, `<=` (JMESPath can't parse "53%"), complex OR logic.

---

## 11. **Viz Block Matching** (`viz_matching.py`)

### Two-Gate Scoring
```
section_score  = cosine(query, section_embedding)
tags_score     = cosine(query, tags_embedding)  [if tags exist]
combined_score = 0.75 * section_score + 0.25 * tags_score
```

**Gates:** `section_score >= 0.60` AND `combined_score >= 0.70`

### JSON Block Summary for LLM Context
When a block passes the gates but no condition filter triggered, extracts:
- Column names
- First 10 rows as sample records
- Formatted as: `[Source Data Context: SectionName]`

This gives the LLM structured data awareness even for non-filter queries.

---

## 12. **Image Engine** (`image_engine.py`)

### Dual Extraction Strategy
1. **Embedded raster images**: PyMuPDF (fitz) → size filters (≥80×80px, ≤50MB)
2. **Google Drive links**: Regex extraction from text layer → normalize to `uc?export=view&id=`

### Caption Generation
- Takes last 300 chars of text before image position
- Strips JSON blobs (keeps human-readable)
- Used for semantic matching

### Question Matching
Two-pass scoring:
- **Keyword**: `|query_words ∩ image_keywords| / |image_keywords|`
- **Embedding**: BGE cosine on caption

**Threshold:** `emb_score >= 0.50` OR `kw_score >= 0.10`

**Pre-filter:** Only runs if query contains `IMAGE_TRIGGER_WORDS` (e.g., "image", "chart", "show", "see").

---

## 13. **Answer Generation** (`answer.py`)

Simple wrapper around LangChain prompt:
```python
QA_PROMPT = """You are a helpful assistant. Use the following context...
Answer ONLY what is asked. Do not output raw JSON or code blocks.
Summarize structured data briefly in plain English."""
```

---

## 14. **Flask Application** (`flask_app_19.py`)

### Session State
```python
sessions[session_id] = {
    "vectorstore": FAISS index,
    "bm25_index": BM25Okapi,
    "chunks": list[Document],
    "llm": Ollama,
    "memory": ConversationBufferWindowMemory(k=5),
    "chain": ConversationalRetrievalChain,
    "history": list[dict],
    "full_json_store": dict,
    "embed_cache": dict,
    "image_store": list[dict],
}
```

### Chat Endpoint Pipeline (`/api/chat`)

```
Step 1: Condition Filter (LLM path → hardcoded fallback)
    ├─ If hit → cond_result = {filtered_rows, html_table, llm_context, ...}
    └─ Skip item retrieval if cond_result exists (isolation guarantee)

Step 2: Item-Level Retrieval (only if no condition hit)
    ├─ If hit → item_hit = {row, row_html, row_text}
    └─ Skip if cond_result exists

Step 3: Hybrid Retrieval (FAISS + BM25)
    ├─ dual_mode logic (topic switch, ambiguous token condensing)
    └→ best_docs, query_used

Step 4: Metadata Rerank
    └→ Re-ordered docs (max 6)

Step 5: Build Extra Context
    ├─ cond_result → filtered rows text
    ├─ item_hit → row detail text
    └─ matched_blocks (viz) → JSON summary (if combined_score >= 0.60)

Step 6: Generate Answer
    └→ LLM(QA_PROMPT + extra_context + doc_context)

Step 7: Inject HTML
    ├─ cond_result → prepend styled table/chart div
    ├─ item_hit → prepend row HTML table
    └→ answer_html

Step 8: Sync Memory
    └→ ConversationBufferWindowMemory.save_context()

Step 9: Viz Injection
    └→ Append data-viz-json divs for matched blocks

Step 10: Image Injection
    └→ Append auto-image divs for matched images
```

### Memory Synchronization
```python
session["memory"].save_context(
    {"question": question}, 
    {"answer": answer_text}
)
```
This enables the `ConversationalRetrievalChain` to handle follow-ups, while the custom `hybrid_dual_mode_retrieval` runs in parallel for specialized retrieval.

---

## 15. **Utilities** (`utils.py`)

Single helper:
```python
def _tags_sentence(tags: list[str]) -> str:
    return " ".join(t.replace("_", " ").strip() for t in tags if t.strip())
```
Converts `["projects_list", "individual_projects"]` → `"projects list individual projects"` for embedding.

---

## Key Design Patterns & Trade-offs

| Pattern | Implementation | Trade-off |
|---------|---------------|-----------|
| **Lazy Singletons** | `_embed_model`, `_cajf_singleton` | Memory efficiency vs. first-request latency |
| **Two-Path Fallback** | LLM filter → hardcoded filter | Flexibility vs. reliability (LLM may fail) |
| **Isolation Guarantee** | Skip item retrieval if condition hit | Clean answers vs. potential missed matches |
| **Embedding Cache** | `embed_cache: dict[str, list[float]]` | Speed vs. memory (unbounded growth) |
| **Chunk Content Key** | `page_content[:120]` for BM25 mapping | Simplicity vs. collision risk |
| **Honest Zero Results** | Commit to field, don't fall through | Accuracy vs. recall |

---

## Potential Vulnerabilities & Edge Cases

1. **Embedding Cache Growth**: `embed_cache` is never pruned. Long sessions with many unique sections/tags could cause memory bloat.

2. **Chunk Key Collision**: `page_content[:120]` as BM25→dense mapping key. If two chunks share identical first 120 chars, mapping fails.

3. **BM25 Index Staleness**: Built once at upload. No incremental updates if chunks change.

4. **LLM Timeout**: `ContextAwareJSONFilter` uses `request_timeout=120.0`. Slow LLM responses could hang the chat endpoint.

5. **Field Alias Domain Mismatch**: `_FIELD_ALIASES` contains software dev terms, but `_CONDITION_KEYWORD_RULES` and test data use Bible translation terms. The `condition_filter.py` hardcoded path may not trigger correctly for software dev queries if the regex patterns don't match.

6. **Image Caption Quality**: `_extract_image_caption` takes last 300 chars of page text, which may be completely unrelated to the image (e.g., footer text, next section).

7. **JSON Splitter Size**: `max_size=1000` bytes for JSON chunks. Large rows may still exceed this, but the splitter doesn't handle nested objects gracefully.

---

## Test Coverage in `condition_based_filter_new.py`

The module includes extensive test cases across 5 categories:

| Category | Count | Examples |
|----------|-------|----------|
| **Original** | 16 | "show inactive projects", "regions with missing data over 100" |
| **A: Superlatives** | 10 | "highest full stack projects", "top 5", "bottom 3" |
| **B: Semantic/Fuzzy** | 10 | "half completed", "barely started", "nearly finished" |
| **C: Explicit Range** | 10 | "between 40% and 60%", "iOT > 40%" |
| **D: Specific Items (OR)** | 10 | "Helix, Vega, and Terra", "LATAM and LATAM North" |
| **E: Complex Combined** | 10 | "Active projects in LATAM North with >50% completion" |

---

Your system demonstrates sophisticated **multi-modal RAG architecture** with clean module boundaries, robust fallback chains, and thoughtful handling of structured data. The LLM-driven condition filter is particularly well-designed with its multi-layer parsing and validation repair system.
