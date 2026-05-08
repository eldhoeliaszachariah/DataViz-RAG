import re
import json

from config import (
    _LLM_FILTER_TRIGGER_TOKENS,
    _FIELD_ALIASES,
    CONDITION_FILTER_THRESHOLD,
    CONDITION_CONTEXT_MAX_ROWS,
)
from vector_math import embed, _cosine
from utils import _tags_sentence
from condition_filter import (
    _build_no_results_html,
    _filtered_rows_to_html,
    _filtered_rows_to_llm_text,
)


# ─── ContextAwareJSONFilter — LLM-driven condition-based filtering ────────────
# Imported from the standalone module; instantiated once as a lazy singleton so
# the embedding model and Ollama connection are shared with the rest of the app.
try:
    from condition_based_filter_new import ContextAwareJSONFilter as _CAJFClass
    _CAJF_AVAILABLE = True
except ImportError:
    _CAJF_AVAILABLE = False
    print("⚠️  condition_based_filter.py not found — LLM-driven filter disabled.")

_cajf_singleton: "_CAJFClass | None" = None   # lazy init on first use

def _get_cajf():
    """Return the ContextAwareJSONFilter singleton, creating it on first call."""
    global _cajf_singleton
    if not _CAJF_AVAILABLE:
        return None
    if _cajf_singleton is None:
        print("🔧 Initialising ContextAwareJSONFilter singleton…")
        _cajf_singleton = _CAJFClass()
        print("✅ ContextAwareJSONFilter ready.")
    return _cajf_singleton


# =========================================================================
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  LLM-DRIVEN CONDITION FILTER  (ContextAwareJSONFilter bridge)       ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║                                                                      ║
# ║  This section wraps condition_based_filter.ContextAwareJSONFilter   ║
# ║  into the same interface expected by the /api/chat pipeline:        ║
# ║                                                                      ║
# ║  detect_condition_query_llm() — decides whether to engage           ║
# ║  apply_condition_filter_llm() — runs LLM filter, returns a         ║
# ║    result dict compatible with the existing cond_result consumer    ║
# ║                                                                      ║
# ║  For "table" viz_type results: the existing styled HTML table path  ║
# ║    is used (_filtered_rows_to_html).                                ║
# ║                                                                      ║
# ║  For non-table viz_type results (bar, donut, line, etc.): the       ║
# ║    filtered JSON is packed as a data-viz-json div so the frontend   ║
# ║    renders it as the appropriate chart type.                        ║
# ║                                                                      ║
# ║  For meta/section text-gen questions: filtered rows are serialised  ║
# ║    into the LLM context exactly as the hardcoded path does.        ║
# ║                                                                      ║
# ║  ISOLATION GUARANTEE: item-level retrieval, FAISS, BM25, metadata  ║
# ║    re-ranking, viz-block matching and image injection are ALL       ║
# ║    untouched — the new code runs only in the condition-filter slot. ║
# ╚══════════════════════════════════════════════════════════════════════╝
# =========================================================================


def _reconstruct_input_json(block_id: str, info: dict) -> dict:
    """
    Reconstruct the full {type, meta, data} envelope from a full_json_store entry.

    full_json_store stores:
        raw_json  → JSON-serialised data_payload  (the "data" array / object)
        viz_type  → normalised chart type string
        section   → section heading string
        tags      → list[str]

    ContextAwareJSONFilter.process_query() expects exactly:
        {"type": "...", "meta": {"section": "...", "tags": [...]}, "data": [...]}
    """
    try:
        data_payload = json.loads(info.get("raw_json", "[]"))
    except Exception:
        data_payload = []

    return {
        "type": info.get("viz_type", "table"),
        "meta": {
            "section": info.get("section", ""),
            "tags":    info.get("tags", []),
        },
        "data": data_payload if isinstance(data_payload, list) else [data_payload],
    }


def _build_filtered_viz_html(filtered_json: dict, section: str) -> str:
    """
    Build the frontend injection HTML for a non-table filtered result.

    For table types: returns "" (caller falls through to _filtered_rows_to_html).
    For chart types: returns a data-viz-json div that the frontend renders as
                     the appropriate chart.  The JSON is the full filtered
                     {type, meta, data} envelope so the frontend has all context.
    """
    viz_type = filtered_json.get("type", "table")
    if viz_type == "table":
        return ""   # table path handled separately

    # Serialise filtered JSON for the data-viz-json attribute
    safe_json = json.dumps(filtered_json.get("data", []), ensure_ascii=False).replace("'", "&#39;")
    return (
        f'<div class="auto-viz" '
        f'data-viz-type="{viz_type}" '
        f'data-viz-json=\'{safe_json}\'></div>'
    )


def detect_condition_query_llm(
    question: str,
    full_json_store: dict,
) -> bool:
    """
    Fast heuristic gate: should we attempt the LLM-driven filter?

    Returns True when ALL of the following hold:
      1. ContextAwareJSONFilter is available
      2. At least one trigger token is present in the question
      3. There is at least one JSON block in the store that has tabular data
         (i.e., data is a non-empty list of dicts)

    This gate is intentionally lightweight — it does NOT call the LLM.
    The real intelligence lives inside ContextAwareJSONFilter.process_query().
    No keywords or field names are hardcoded here.
    """
    if not _CAJF_AVAILABLE:
        return False

    if not full_json_store:
        return False

    q_words = set(re.findall(r'[a-zA-Z]+', question.lower()))

    # Triggered by:
    # 1. Explicit trigger tokens
    has_trigger = bool(_LLM_FILTER_TRIGGER_TOKENS & q_words)
    # 2. Field aliases (e.g. "nt", "ot", "full bible")
    has_alias   = any(alias in question.lower() for alias in _FIELD_ALIASES)
    # 3. Numeric comparisons (e.g. "under 80%", "> 50")
    has_num     = bool(re.search(r'([><=]|under|above|between|over|below|than)\s*[\d.]+', question.lower()))

    if not (has_trigger or has_alias or has_num):
        return False

    # Must have at least one block with a list of dicts
    for info in full_json_store.values():
        try:
            payload = json.loads(info.get("raw_json", "[]"))
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                return True
        except Exception:
            continue

    return False


def apply_condition_filter_llm(
    question: str,
    full_json_store: dict,
    embed_cache: dict,
) -> dict | None:
    """
    LLM-driven condition filter.  Replaces the hardcoded apply_condition_filter()
    for queries that pass detect_condition_query_llm().

    Pipeline:
      1. Rank candidate blocks by semantic similarity (section + tag cosine),
         same logic as the existing hardcoded path.
      2. For the best candidate block, reconstruct the full {type, meta, data}
         envelope and call ContextAwareJSONFilter.process_query(input_json, query).
      3. If the LLM engine returns filtered rows, build:
             - html_table  (table type)  or  chart_html (non-table type)
             - llm_context for text generation
             - the standard cond_result dict consumed by the chat endpoint
      4. If the engine returns the same rows as the input (no filtering happened,
         i.e. "all" rows are a valid answer), treat as a match with all rows.
      5. If no candidate block scores above CONDITION_FILTER_THRESHOLD → return None
         so the hardcoded fallback can handle it.

    Return dict keys (same interface as apply_condition_filter):
        filtered_rows   list[dict]
        field           str  (first column name of filtered data)
        condition_label str
        section         str
        total_rows      int
        html_table      str  (HTML for table-type; chart div for chart-type)
        llm_context     str
        no_results      bool
        viz_type        str  (NEW — "table" | "bar" | "donut" | …)
        filtered_json   dict (NEW — full {type,meta,data} for chart rendering)
    """
    cajf = _get_cajf()
    if cajf is None:
        return None

    if not full_json_store:
        return None

    # ── Step 1: Rank candidate blocks by semantic similarity ─────────────────
    q_vec = embed(question)
    q_words = set(re.findall(r'[a-z0-9]{3,}', question.lower()))
    candidate_blocks = []
    
    for block_id, info in full_json_store.items():
        section  = info.get("section", "")
        tags_str = _tags_sentence(info.get("tags", []))
        if not section:
            continue
            
        sec_vec = embed_cache.get(section) or embed(section)
        embed_cache[section] = sec_vec
        sec_score = _cosine(q_vec, sec_vec)
        
        tag_score = 0.0
        if tags_str:
            tag_vec = embed_cache.get(tags_str) or embed(tags_str)
            embed_cache[tags_str] = tag_vec
            tag_score = _cosine(q_vec, tag_vec)
            
        # Data value overlap boost: Check if query terms appear in sample data
        data_boost = 0.0
        try:
            raw_data = info.get("raw_json", "[]")
            # Fast scan of the start of the JSON for query words (e.g. project names)
            sample_json = raw_data[:2000].lower()
            for word in q_words:
                if word in sample_json:
                    data_boost = 0.35
                    break
        except: pass
            
        # Composite score: Section + Tags + Data Overlap
        # Data boost is critical for specific items (Category D) that don't match headers
        score = sec_score * 0.35 + tag_score * 0.45 + data_boost * 0.20
        
        if score >= CONDITION_FILTER_THRESHOLD or data_boost > 0:
            candidate_blocks.append((max(score, data_boost), block_id, info))

    candidate_blocks.sort(key=lambda x: x[0], reverse=True)

    if not candidate_blocks:
        print(f"  [LLM-COND] No blocks above threshold={CONDITION_FILTER_THRESHOLD} — deferring to hardcoded path")
        return None

    print(f"  [LLM-COND] {len(candidate_blocks)} candidate block(s) for question: '{question[:80]}'")

    # ── Step 2: Try blocks in order until we get a useful filtered result ─────
    for block_score, block_id, info in candidate_blocks:
        section  = info.get("section", "")
        viz_type = info.get("viz_type", "table")

        # Reconstruct the full JSON envelope
        input_json = _reconstruct_input_json(block_id, info)
        data       = input_json.get("data", [])

        if not data or not isinstance(data[0], dict):
            print(f"  [LLM-COND] Block '{section}' — skipping (empty or non-dict data)")
            continue

        total_rows = len(data)

        print(f"  [LLM-COND] Trying block '{section}' viz={viz_type} rows={total_rows} score={block_score:.3f}")

        # ── Step 3: Call ContextAwareJSONFilter ─────────────────────────────
        try:
            result_json = cajf.process_query(input_json, question)
        except Exception as e:
            print(f"  [LLM-COND] ContextAwareJSONFilter error: {e}")
            continue

        filtered_data = result_json.get("data", [])
        filtered_rows = filtered_data if isinstance(filtered_data, list) else []

        # Derive a human-readable condition label from the LLM's filter logic
        # (ContextAwareJSONFilter doesn't expose it directly, so we build a
        #  compact summary from question + section)
        condition_label = f"Query: \"{question}\" → Section: {section}"

        # ── Step 4: Handle zero results ──────────────────────────────────────
        if not filtered_rows:
            no_results_html = _build_no_results_html(section, condition_label, total_rows)
            no_results_ctx  = (
                f"Filter applied on section '{section}'.\n"
                f"Result: 0 out of {total_rows} rows matched the query.\n"
                f"There are no records satisfying this condition in the data."
            )
            print(f"  [LLM-COND] 0/{total_rows} rows matched in '{section}'")
            return {
                "filtered_rows":   [],
                "field":           list(data[0].keys())[0] if data else "",
                "condition_label": condition_label,
                "section":         section,
                "total_rows":      total_rows,
                "html_table":      no_results_html,
                "llm_context":     no_results_ctx,
                "no_results":      True,
                "viz_type":        viz_type,
                "filtered_json":   result_json,
            }

        print(f"  ✅ [LLM-COND] '{section}' → {len(filtered_rows)}/{total_rows} rows | viz={viz_type}")

        # ── Step 5: Build HTML output ─────────────────────────────────────────
        # For table type: use the styled cond-table HTML (same as hardcoded path)
        # For chart types: build a data-viz-json div for frontend rendering
        if viz_type == "table":
            html_output = _filtered_rows_to_html(filtered_rows, section, condition_label, total_rows)
        else:
            # Build chart div with filtered data + section title header
            chart_div = _build_filtered_viz_html(result_json, section)
            # Also prepend a small header so the user knows what was filtered
            header_html = (
                f'<div class="cond-table-header">'
                f'<span class="cond-table-icon">⚙</span>'
                f'<span class="cond-table-title">{section}</span>'
                f'<span class="cond-table-badge">{len(filtered_rows)} / {total_rows} rows</span>'
                f'</div>'
                f'<div class="cond-table-filter-bar">'
                f'<span class="cond-active-label">Filter:</span> '
                f'<code class="cond-active-code">{condition_label}</code>'
                f'</div>'
            )
            html_output = header_html + chart_div

        # ── Step 6: Build LLM context text ───────────────────────────────────
        context_rows = filtered_rows[:CONDITION_CONTEXT_MAX_ROWS]
        llm_context  = _filtered_rows_to_llm_text(context_rows, section, condition_label, total_rows)

        return {
            "filtered_rows":   filtered_rows,
            "field":           list(filtered_rows[0].keys())[0] if filtered_rows else "",
            "condition_label": condition_label,
            "section":         section,
            "total_rows":      total_rows,
            "html_table":      html_output,
            "llm_context":     llm_context,
            "no_results":      False,
            "viz_type":        viz_type,
            "filtered_json":   result_json,
        }

    # No block gave useful results
    print(f"  [LLM-COND] No useful filtered result across all candidate blocks")
    return None
