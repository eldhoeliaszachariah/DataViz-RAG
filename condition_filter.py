import re

from config import (
    _CONDITION_TRIGGER_TOKENS,
    _CONDITION_KEYWORD_RULES,
    _FIELD_ALIASES,
    CONDITION_FILTER_THRESHOLD,
    CONDITION_CONTEXT_MAX_ROWS,
)
from vector_math import embed, _cosine
from utils import _tags_sentence


# =========================================================================
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONDITION-BASED JSON FILTER ENGINE                                 ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║                                                                      ║
# ║  Detects when a question is a "condition filter" query, e.g.:       ║
# ║    • "list all full stack finished projects"                         ║
# ║    • "show active projects only"                                     ║
# ║    • "which projects have NT finished = 100%"                        ║
# ║    • "iOT finished projects"                                          ║
# ║                                                                      ║
# ║  Pipeline:                                                           ║
# ║    1. detect_condition_query()  — decide if query is a filter Q      ║
# ║    2. resolve_target_field()    — which JSON field is being filtered  ║
# ║    3. resolve_condition()       — what condition? (100%, active, 0%) ║
# ║    4. apply_condition_filter()  — filter rows from matching blocks   ║
# ║    5. build_condition_response()— HTML table + LLM context text      ║
# ║                                                                      ║
# ╚══════════════════════════════════════════════════════════════════════╝
# =========================================================================


def _pct_value(v) -> float | None:
    """Parse a percentage string or number to float. Returns None on failure."""
    s = str(v).strip().rstrip('%')
    try:
        return float(s)
    except ValueError:
        return None


def _apply_single_condition(row: dict, field: str, cond_type: str, cond_value) -> bool:
    """
    Test one row[field] against a single condition.
    Returns True if the row satisfies the condition.
    """
    raw = row.get(field)
    if raw is None:
        return False

    if cond_type == "eq_str":
        return str(raw).strip().lower() == str(cond_value).lower()

    if cond_type == "contains":
        return str(cond_value).lower() in str(raw).lower()

    # Percentage conditions
    pct = _pct_value(raw)
    if pct is None:
        return False

    if cond_type == "gte_pct":  return pct >= float(cond_value)
    if cond_type == "lte_pct":  return pct <= float(cond_value)
    if cond_type == "gt_pct":   return pct >  float(cond_value)
    if cond_type == "lt_pct":   return pct <  float(cond_value)
    if cond_type == "eq_pct":   return pct == float(cond_value)
    if cond_type == "nonzero_pct": return 0 < pct < 100

    return False


def _resolve_field_in_row(row: dict, field_hints: list[str]) -> str | None:
    """
    Given a list of substrings (field hints), find the actual key in `row`
    that best matches. Matching priority (highest wins):
      1. Exact normalized match      (hint == key_normalized)
      2. Hint fully contained in key (hint is a complete word inside key)
      3. Key fully contained in hint (key is a prefix/suffix of hint)

    Short hints (≤ 2 chars) are never matched as loose substrings to prevent
    "ot" from matching "not", "note", etc.
    """
    # Build normalised-key → original-key map
    row_keys_lower = {k.lower().replace("_", "").replace(" ", ""): k for k in row.keys()}

    best_match: str | None = None
    best_priority: int      = 99  # lower is better

    for hint in field_hints:
        hint_norm = hint.lower().replace("_", "").replace(" ", "")
        for norm_k, orig_k in row_keys_lower.items():
            # Priority 1: exact match
            if hint_norm == norm_k:
                if best_priority > 1:
                    best_match, best_priority = orig_k, 1
                break  # can't do better than exact
            # Priority 2: hint is a complete component inside key
            # Guard: only do substring match if hint is long enough (> 2 chars)
            if len(hint_norm) > 2 and hint_norm in norm_k:
                if best_priority > 2:
                    best_match, best_priority = orig_k, 2
            # Priority 3: key is a component inside hint
            elif len(norm_k) > 2 and norm_k in hint_norm:
                if best_priority > 3:
                    best_match, best_priority = orig_k, 3

    return best_match


def _extract_numeric_capture(pattern: re.Pattern, text: str) -> float | None:
    """Extract the captured numeric group from a regex match, if any."""
    m = pattern.search(text)
    if m:
        try:
            return float(m.group(1))
        except (IndexError, ValueError):
            pass
    return None


def resolve_conditions(question: str) -> list[dict]:
    """
    Parse the question text and return a list of condition dicts:
      [{"cond_type": str, "cond_value": float|str, "field_hints": list[str]}, ...]

    Field hints come from two sources:
      1. _FIELD_ALIASES explicit keyword matches — ALL matches collected, longest-
         alias wins when there is overlap (so "new testament" beats "nt" on same span).
      2. Fallback: applied inside apply_condition_filter when field_hints is empty.

    KEY FIX: We no longer `break` on the first alias match.  Instead we collect every
    alias whose token appears in the question, then deduplicate hints while preserving
    order.  Crucially we use word-boundary matching so "ot" only matches when it
    stands alone — preventing "not" / "note" / "quota" from triggering the OT alias.

    Specificity guard: once we have hints from a specific alias (length > 2 chars in
    the alias key), we skip any *competing* general alias whose hints overlap with an
    already-resolved field family.  This prevents "ot" + "finished" in the same query
    from injecting the generic ["finished","complete"] list alongside ["ot_finished","ot"].
    """
    q_lower = question.lower()
    conditions: list[dict] = []

    # ── Find field hints — collect ALL matching aliases ───────────────────────
    # Sort aliases longest-first so multi-word aliases ("new testament", "full bible")
    # are evaluated before their shorter substrings ("nt", "bible").
    seen_hints: list[str] = []
    seen_set:   set[str]  = set()

    sorted_aliases = sorted(_FIELD_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
    for alias, hints in sorted_aliases:
        # Word-boundary-aware match: alias must appear as a whole token sequence,
        # not as a substring of another word (e.g. "ot" must not match "not").
        alias_pattern = r'(?<![a-z])' + re.escape(alias) + r'(?![a-z])'
        if re.search(alias_pattern, q_lower):
            for h in hints:
                if h not in seen_set:
                    seen_hints.append(h)
                    seen_set.add(h)

    field_hints = seen_hints  # may be empty — fallback handled in apply_condition_filter

    # ── Match condition rules ─────────────────────────────────────────────────
    seen_cond_keys: set[tuple] = set()
    for pattern, cond_type, cond_value in _CONDITION_KEYWORD_RULES:
        m = pattern.search(q_lower)
        if not m:
            continue
        actual_value = cond_value
        if cond_value == "__CAPTURE__":
            try:
                actual_value = float(m.group(1))
            except (IndexError, ValueError):
                continue
        # Deduplicate: skip if we already added the exact same (type, value) pair.
        # This prevents "100%" and standalone "finished" both firing gte_pct=100.
        dedup_key = (cond_type, actual_value)
        if dedup_key in seen_cond_keys:
            continue
        seen_cond_keys.add(dedup_key)
        conditions.append({
            "cond_type":   cond_type,
            "cond_value":  actual_value,
            "field_hints": field_hints,   # same hints shared across all conditions
            "pattern_str": pattern.pattern,
        })

    return conditions


def detect_condition_query(question: str, full_json_store: dict) -> bool:
    """
    Fast heuristic: is this question a "condition filter" request?

    Returns True when:
      - At least one condition keyword rule fires, AND
      - At least one trigger token ("list", "show", "all", etc.) is in the question
      OR the question contains a field alias ("nt finished", "full bible", etc.)
    """
    q_lower = question.lower()
    q_words = set(re.findall(r'[a-zA-Z]+', q_lower))

    has_trigger  = bool(_CONDITION_TRIGGER_TOKENS & q_words)
    has_alias    = any(alias in q_lower for alias in _FIELD_ALIASES)
    has_cond     = bool(resolve_conditions(question))

    # Either (trigger + condition) or (alias + condition)
    return has_cond and (has_trigger or has_alias)


def _build_no_results_html(section: str, condition_label: str, total_rows: int) -> str:
    """Return a styled HTML notice when the filter finds zero matching rows."""
    return (
        f'<div style="border:1px solid #f5c518;background:#fffbe6;border-radius:6px;'
        f'padding:12px 16px;margin:8px 0;font-size:0.93em;">'
        f'<strong>🔍 Filter applied:</strong> {condition_label}<br>'
        f'<span style="color:#c0392b;font-weight:600;">No results found.</span> '
        f'None of the {total_rows} rows in <em>{section}</em> satisfy this condition.'
        f'</div>'
    )


def apply_condition_filter(
    question: str,
    full_json_store: dict,
    embed_cache: dict,
) -> dict | None:
    """
    Main entry point for condition-based filtering.

    Returns a dict with:
      - "filtered_rows":  list[dict]   — matching rows (may be empty)
      - "field":          str          — the JSON field that was filtered
      - "condition_label":str          — human-readable condition description
      - "section":        str          — source section name
      - "total_rows":     int          — total rows in the source block
      - "html_table":     str          — rendered HTML table of filtered rows
      - "llm_context":    str          — plain-text context for LLM
      - "no_results":     bool         — True when filter resolved but 0 rows matched
    OR None if no condition match found.

    KEY FIX — "honest zero results" behaviour:
    ───────────────────────────────────────────
    When field_hints are explicitly resolved from the query (e.g. the user asked
    about "OT"), we commit to that field in the BEST semantically-matching block.
    If 0 rows pass the filter we return immediately with no_results=True rather
    than falling through to other blocks where a DIFFERENT field might accidentally
    match (which caused "OT 100%" to return NT=100% rows).

    Fall-through to the next candidate block is only permitted when:
      (a) field_hints were empty (no explicit alias in the query), AND
      (b) the field could not be resolved in the current block at all.
    """
    import json as _json

    if not full_json_store:
        return None

    conditions = resolve_conditions(question)
    if not conditions:
        return None

    q_lower = question.lower()

    # Detect whether the user supplied an EXPLICIT field alias.
    # If so we must NOT reassign to a different field when 0 rows match.
    explicit_field_hints = bool(conditions[0]["field_hints"])

    # ── Pick candidate blocks ranked by semantic similarity ───────────────────
    q_vec = embed(question)
    candidate_blocks = []
    for block_id, info in full_json_store.items():
        section  = info.get("section", "")
        tags_str = _tags_sentence(info.get("tags", []))
        if not section:
            continue
        sec_vec = embed_cache.get(section) or embed(section)
        score   = _cosine(q_vec, sec_vec)
        if tags_str:
            tag_vec = embed_cache.get(tags_str) or embed(tags_str)
            score   = score * 0.80 + _cosine(q_vec, tag_vec) * 0.20
        if score >= CONDITION_FILTER_THRESHOLD:
            candidate_blocks.append((score, block_id, info))
    candidate_blocks.sort(key=lambda x: x[0], reverse=True)

    if not candidate_blocks:
        print(f"  [COND-FILTER] No blocks above threshold={CONDITION_FILTER_THRESHOLD}")
        return None

    print(f"  [COND-FILTER] {len(candidate_blocks)} candidate block(s) | conditions={len(conditions)} | explicit_field={explicit_field_hints}")

    # Track whether we resolved the field at least once (for zero-result reporting)
    field_resolved_result: dict | None = None

    for block_score, block_id, info in candidate_blocks:
        raw_json = info.get("raw_json", "")
        section  = info.get("section", "")
        if not raw_json:
            continue

        try:
            data = _json.loads(raw_json)
        except Exception:
            continue

        rows: list[dict] = []
        if isinstance(data, list):
            rows = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict):
            rows = [data]

        if not rows:
            continue

        # ── Resolve which field each condition maps to ────────────────────────
        resolved_conditions = []
        for cond in conditions:
            field_hints = cond["field_hints"]
            if not field_hints:
                # No explicit alias — fall back to any pct/status field in the block
                all_keys  = list(rows[0].keys())
                pct_keys  = [k for k in all_keys if "%" in str(rows[0].get(k, "")) or
                              any(w in k.lower() for w in ["finished","complete","pct","percent","status"])]
                field_hints = [k.lower() for k in pct_keys]

            actual_field = _resolve_field_in_row(rows[0], field_hints)
            if not actual_field and cond["cond_type"] == "eq_str":
                # Status-type condition: scan all string fields
                for k, v in rows[0].items():
                    if isinstance(v, str) and any(w in v.lower() for w in ["active","inactive","project"]):
                        actual_field = k
                        break

            if actual_field:
                resolved_conditions.append({**cond, "actual_field": actual_field})

        if not resolved_conditions:
            print(f"  [COND-FILTER] Block '{section}' — could not resolve field in row keys: {list(rows[0].keys())}")
            # If the user gave an explicit field hint and it didn't resolve here,
            # fall through to the next block.
            continue

        # ── Apply ALL conditions (AND logic) ──────────────────────────────────
        filtered_rows = []
        for row in rows:
            if all(
                _apply_single_condition(row, c["actual_field"], c["cond_type"], c["cond_value"])
                for c in resolved_conditions
            ):
                filtered_rows.append(row)

        # ── Build human-readable condition label ──────────────────────────────
        label_parts = []
        for c in resolved_conditions:
            field_display = c["actual_field"].replace("_", " ").title()
            ct, cv = c["cond_type"], c["cond_value"]
            if ct == "gte_pct"      : label_parts.append(f"{field_display} ≥ {cv}%")
            elif ct == "lte_pct"    : label_parts.append(f"{field_display} ≤ {cv}%")
            elif ct == "gt_pct"     : label_parts.append(f"{field_display} > {cv}%")
            elif ct == "lt_pct"     : label_parts.append(f"{field_display} < {cv}%")
            elif ct == "eq_pct"     : label_parts.append(f"{field_display} = {cv}%")
            elif ct == "nonzero_pct": label_parts.append(f"{field_display} In Progress (>0%, <100%)")
            elif ct == "eq_str"     : label_parts.append(f"{field_display} = '{cv}'")
            elif ct == "contains"   : label_parts.append(f"{field_display} contains '{cv}'")
        condition_label = " AND ".join(label_parts)

        if not filtered_rows:
            print(f"  [COND-FILTER] Block '{section}' — 0 rows matched | {condition_label}")

            # KEY FIX ─────────────────────────────────────────────────────────
            # If the user gave an EXPLICIT field alias (e.g. "OT"), we have
            # committed to the correct field.  Zero rows means the answer is
            # genuinely "no such data exists" — do NOT fall through and let a
            # different field match accidentally.
            if explicit_field_hints:
                # Remember this as the canonical zero-result for this query.
                if field_resolved_result is None:
                    no_results_html = _build_no_results_html(section, condition_label, len(rows))
                    no_results_ctx  = (
                        f"Filter applied on section '{section}': {condition_label}.\n"
                        f"Result: 0 out of {len(rows)} rows matched.\n"
                        f"There are no records satisfying this condition in the data."
                    )
                    field_resolved_result = {
                        "filtered_rows":   [],
                        "field":           resolved_conditions[0]["actual_field"],
                        "condition_label": condition_label,
                        "section":         section,
                        "total_rows":      len(rows),
                        "html_table":      no_results_html,
                        "llm_context":     no_results_ctx,
                        "no_results":      True,
                    }
                # Stop looking — we found the right block and field; no rows matched.
                continue  # still try remaining blocks in case a *better* block exists
            else:
                # No explicit field — safe to try the next block
                continue
        # ─────────────────────────────────────────────────────────────────────

        print(f"  ✅ [COND-FILTER] '{section}' → {len(filtered_rows)}/{len(rows)} rows | {condition_label}")

        # ── Build HTML table ──────────────────────────────────────────────────
        html_table = _filtered_rows_to_html(filtered_rows, section, condition_label, len(rows))

        # ── Build LLM context text ────────────────────────────────────────────
        context_rows = filtered_rows[:CONDITION_CONTEXT_MAX_ROWS]
        llm_context  = _filtered_rows_to_llm_text(context_rows, section, condition_label, len(rows))

        return {
            "filtered_rows":   filtered_rows,
            "field":           resolved_conditions[0]["actual_field"],
            "condition_label": condition_label,
            "section":         section,
            "total_rows":      len(rows),
            "html_table":      html_table,
            "llm_context":     llm_context,
            "no_results":      False,
        }

    # ── If we resolved the field but got 0 rows, return the honest no-result ──
    if field_resolved_result is not None:
        print(f"  ℹ️  [COND-FILTER] Field resolved but 0 rows matched — returning no-results response")
        return field_resolved_result

    print(f"  ℹ️  [COND-FILTER] No rows matched across all candidate blocks")
    return None


def _filtered_rows_to_html(
    rows: list[dict],
    section: str,
    condition_label: str,
    total_rows: int,
) -> str:
    """
    Render filtered rows as a styled HTML table with a header showing
    the active filter condition and match count.
    """
    if not rows:
        return ""

    keys = list(rows[0].keys())

    # ── Header ────────────────────────────────────────────────────────────────
    header_html = (
        f'<div class="cond-table-header">'
        f'<span class="cond-table-icon">⚙</span>'
        f'<span class="cond-table-title">{section}</span>'
        f'<span class="cond-table-badge">{len(rows)} / {total_rows} rows</span>'
        f'</div>'
        f'<div class="cond-table-filter-bar">'
        f'<span class="cond-active-label">Filter:</span> '
        f'<code class="cond-active-code">{condition_label}</code>'
        f'</div>'
    )

    # ── Table head ────────────────────────────────────────────────────────────
    th_cells = "".join(
        f'<th class="cond-th">{k.replace("_"," ").title()}</th>' for k in keys
    )
    thead_html = f'<thead><tr>{th_cells}</tr></thead>'

    # ── Table body ────────────────────────────────────────────────────────────
    def _render_cell(k: str, v) -> str:
        s = str(v).strip() if v is not None else "--"
        # Percentage bar
        if re.match(r'^\d+(\.\d+)?%$', s):
            pct = float(s.rstrip('%'))
            color = "#3dd68c" if pct >= 100 else "#4f9cf9" if pct > 0 else "#4a5068"
            return (
                f'<div class="cond-pct-cell">'
                f'<div class="cond-pct-bar" style="width:{min(pct,100):.0f}%;background:{color}"></div>'
                f'<span class="cond-pct-label">{s}</span>'
                f'</div>'
            )
        # Status badge
        if "status" in k.lower():
            cls = "cond-badge-active" if "active" in s.lower() and "inactive" not in s.lower() else "cond-badge-inactive"
            return f'<span class="cond-badge {cls}">{s}</span>'
        return s

    rows_html = ""
    for i, row in enumerate(rows):
        row_class = "cond-tr-odd" if i % 2 == 0 else "cond-tr-even"
        cells = "".join(f'<td class="cond-td">{_render_cell(k, row.get(k))}</td>' for k in keys)
        rows_html += f'<tr class="{row_class}">{cells}</tr>'

    tbody_html = f'<tbody>{rows_html}</tbody>'

    # ── Wrap it all ───────────────────────────────────────────────────────────
    table_html = (
        f'<div class="cond-table-wrapper">'
        f'{header_html}'
        f'<div class="cond-table-scroll">'
        f'<table class="cond-table">{thead_html}{tbody_html}</table>'
        f'</div>'
        f'</div>'
    )
    return table_html


def _filtered_rows_to_llm_text(
    rows: list[dict],
    section: str,
    condition_label: str,
    total_rows: int,
) -> str:
    """
    Convert filtered rows to a compact plain-English block for LLM context injection.
    """
    if not rows:
        return ""

    lines = [
        f"[Filtered Results] Section: {section}",
        f"Condition: {condition_label}",
        f"Found {len(rows)} matching rows out of {total_rows} total.",
        "",
    ]
    for i, row in enumerate(rows[:CONDITION_CONTEXT_MAX_ROWS], 1):
        pairs = "; ".join(f"{k}: {v}" for k, v in row.items())
        lines.append(f"  Row {i}: {pairs}")

    if len(rows) > CONDITION_CONTEXT_MAX_ROWS:
        lines.append(f"  ... ({len(rows) - CONDITION_CONTEXT_MAX_ROWS} more rows not shown)")

    return "\n".join(lines)
