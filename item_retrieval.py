import re
import json

from config import ITEM_COSINE_THRESHOLD
from vector_math import embed, embed_doc, _cosine


# =========================================================================
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ITEM-LEVEL JSON RETRIEVAL  (two-layer)                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
#  Layer 1 — Find the asked item inside JSON rows/records.
#             Tries exact key-value match, then semantic cosine on
#             stringified row content.
#
#  Layer 2 — If an item row is found:
#             (a) returns the row as an HTML table for direct UI injection
#             (b) returns a plain-English text description of the row
#                 to prepend into the LLM context
# =========================================================================

def _row_to_text(row: dict | list, section: str = "") -> str:
    """Convert a single JSON row to a readable text sentence for LLM context."""
    if isinstance(row, dict):
        pairs = "; ".join(f"{k}: {v}" for k, v in row.items())
        prefix = f"[{section}] " if section else ""
        return f"{prefix}{pairs}"
    if isinstance(row, list):
        return (f"[{section}] " if section else "") + ", ".join(str(x) for x in row)
    return str(row)


def _row_to_html_table(row: dict | list, section: str = "") -> str:
    """Convert a single JSON row dict/list to an HTML table string."""
    title_html = f'<caption style="font-weight:bold;text-align:left;padding:4px 0">{section}</caption>' if section else ""

    if isinstance(row, dict):
        rows_html = "".join(
            f"<tr><th style='text-align:left;padding:3px 8px;background:#f0f4ff'>{k}</th>"
            f"<td style='padding:3px 8px'>{v}</td></tr>"
            for k, v in row.items()
        )
        return (
            f'<div class="item-table-wrapper" style="overflow-x:auto;margin:8px 0">'
            f'<table style="border-collapse:collapse;font-size:0.92em;width:100%;border:1px solid #ccc">'
            f"{title_html}{rows_html}</table></div>"
        )

    if isinstance(row, list):
        # Flat list → single-row table
        header = "".join(
            f"<th style='padding:3px 8px;background:#f0f4ff'>{i}</th>"
            for i in range(len(row))
        )
        cells  = "".join(
            f"<td style='padding:3px 8px'>{v}</td>"
            for v in row
        )
        return (
            f'<div class="item-table-wrapper" style="overflow-x:auto;margin:8px 0">'
            f'<table style="border-collapse:collapse;font-size:0.92em;width:100%;border:1px solid #ccc">'
            f"{title_html}<thead><tr>{header}</tr></thead>"
            f"<tbody><tr>{cells}</tr></tbody></table></div>"
        )
    return ""


def _stringify_row(row, section: str = "") -> str:
    """
    Flatten any row type to a rich searchable string.
    Including the section name gives BGE enough context to score
    a short data row against a natural-language question.
    """
    if isinstance(row, dict):
        content = " ".join(f"{k} {v}" for k, v in row.items())
    elif isinstance(row, list):
        content = " ".join(str(x) for x in row)
    else:
        content = str(row)
    # Prepend section so the embedding captures thematic context
    return f"{section} {content}".strip() if section else content


def _score_row_against_query(q_words: set, q_phrase: str, row) -> tuple[float, str]:
    """
    Specificity-aware multi-signal row scorer.

    Returns (score, match_type) where match_type is a debug label.

    Scoring tiers (highest wins, not additive across tiers):
    ─────────────────────────────────────────────────────────
    Tier 1 — EXACT PHRASE MATCH (score 1.0)
        The entire query phrase (lowercased, stripped of stop-words)
        appears verbatim inside a concatenated cell value string.
        e.g. query "intent projects" inside "IntentProjects" → YES

    Tier 2 — FULL COVERAGE MATCH (score 0.85)
        Every significant query token appears inside at least one cell value.
        AND the coverage ratio (matched_tokens / value_tokens) is high,
        meaning the row's value is specific to this query, not generic.
        e.g. "intent projects": both "intent" and "projects" are substrings
        of "IntentProjects" → coverage = 2/2 = 1.0 → YES
        But for row "Projects": only "projects" matches, coverage = 1/2 = 0.5 → lower

    Tier 3 — PARTIAL COVERAGE (score = coverage_ratio * 0.70)
        Some tokens match but not all.
        e.g. "projects" matches "Projects" row with coverage 0.5 → score 0.35

    Tier 4 — NO MATCH (score 0.0)

    Specificity penalty:
        If a value matches multiple rows, rows where the value is a superset
        of the query (generic) are penalised vs rows where the query covers
        the full value (specific).

    Stop-words are excluded from token-level matching so "the", "of", "in"
    don't pollute scores, but they are NOT hardcoded keywords — we only filter
    the most universal English function words.
    """
    _STOP = {
        'the','a','an','is','are','was','were','and','or','in','on','at','to',
        'of','for','with','this','that','these','those','it','its','by','from',
        'has','have','been','be','as','which','also','me','my','show','tell',
        'give','what','how','many','much','about','details','info','please',
        'get','find','list','all','some','any','do','does','did','can','could',
        'would','should','will','i','you','we','they','he','she','his','her',
    }

    # Significant query tokens (stop-words removed)
    sig_q = {w for w in q_words if w not in _STOP and len(w) >= 2}
    if not sig_q:
        return 0.0, "no_sig_tokens"

    # Collect cell values
    if isinstance(row, dict):
        raw_values = [str(v) for v in row.values() if v is not None]
    elif isinstance(row, list):
        raw_values = [str(v) for v in row if v is not None]
    else:
        raw_values = [str(row)]

    # Normalised cell strings (lowercased, with both full-string and tokens)
    cell_norms: list[str] = []          # full lowercased cell strings
    cell_token_sets: list[set] = []     # per-cell token sets
    for rv in raw_values:
        norm = rv.strip().lower()
        cell_norms.append(norm)
        cell_token_sets.append(set(re.findall(r'[a-zA-Z0-9]+', norm)))

    # ── Tier 1: Exact phrase match ──────────────────────────────────────────
    # Build a "compact" query phrase: sig tokens joined, no spaces
    # e.g. "intent projects" → "intentprojects"
    compact_q = "".join(sorted(sig_q))   # sorted for stability, not ordered
    # Also try space-joined and natural order
    phrase_variants = [
        q_phrase.lower().strip(),
        " ".join(sig_q),
        "".join(sig_q),
    ]
    for cell_norm in cell_norms:
        cell_compact = re.sub(r'[^a-z0-9]', '', cell_norm)
        for pv in phrase_variants:
            pv_compact = re.sub(r'[^a-z0-9]', '', pv)
            if pv_compact and pv_compact in cell_compact:
                return 1.0, "exact_phrase"
            if pv and pv in cell_norm:
                return 1.0, "exact_phrase"

    # ── Tier 2 & 3: Token coverage ──────────────────────────────────────────
    # For each sig query token, check if it appears as a substring inside any cell value
    matched_tokens:  set[str] = set()
    unmatched_tokens: set[str] = set()

    for tok in sig_q:
        found = False
        for cell_norm in cell_norms:
            # substring match inside cell value (handles CamelCase compounds)
            if tok in cell_norm:
                found = True
                break
        if found:
            matched_tokens.add(tok)
        else:
            unmatched_tokens.add(tok)

    if not matched_tokens:
        return 0.0, "no_match"

    # Coverage ratio = matched_tokens / sig_q  (how much of query is covered)
    q_coverage = len(matched_tokens) / len(sig_q)

    # Specificity ratio: for each matched token, compute how much of the
    # cell value it "fills".  A token "projects" fills 8/14 chars of
    # "intentprojects" (57%) but 8/8 chars of "projects" (100%).
    # We pick the BEST matching cell for specificity.
    best_specificity = 0.0
    for tok in matched_tokens:
        for cell_norm in cell_norms:
            if tok in cell_norm:
                # specificity = len(tok) / len(cell without spaces)
                cell_compact = re.sub(r'\s+', '', cell_norm)
                if cell_compact:
                    spec = len(tok) / len(cell_compact)
                    best_specificity = max(best_specificity, spec)

    # Tier 2: Full coverage — all sig tokens matched
    if q_coverage >= 1.0:
        # Penalise generic rows: if specificity is low, the row's value is
        # broader than the query (e.g. "projects" matching "AllProjectsCount").
        # Reward specific rows (e.g. "intentprojects" matches "intent"+"projects").
        combined = 0.85 * (0.5 + 0.5 * best_specificity)
        return combined, f"full_cov spec={best_specificity:.2f}"

    # Tier 3: Partial coverage
    combined = q_coverage * 0.70 * (0.4 + 0.6 * best_specificity)
    return combined, f"partial_cov={q_coverage:.2f} spec={best_specificity:.2f}"


def item_level_retrieval(
    question: str,
    full_json_store: dict,
    embed_cache: dict,
) -> dict | None:
    """
    Specificity-aware two-layer item-level retrieval.

    Scoring pipeline per row:
        1. _score_row_against_query()  — tiered token/phrase match (precision signal)
        2. BGE cosine on enriched row string (semantic signal)
        3. final = cosine * SEMANTIC_W + match_score * MATCH_W

    The specificity-aware match scorer ensures that a row whose values
    closely and completely cover the query tokens wins over a generic row
    that only partially overlaps — no field names hardcoded anywhere.
    """
    if not full_json_store:
        return None

    # Weights for combining semantic and match signals
    SEMANTIC_W = 0.45
    MATCH_W    = 0.55

    q_lower = question.lower()
    q_words = set(re.findall(r'[a-zA-Z0-9]+', q_lower))
    q_vec   = embed(question)

    best_score  = 0.0
    best_result = None

    for block_id, info in full_json_store.items():
        section  = info.get("section", "")
        raw_json = info.get("raw_json", "")
        if not raw_json:
            continue

        try:
            data = json.loads(raw_json)
        except Exception:
            continue

        # Normalise to list of rows
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = [data]
        else:
            rows = [data]

        for row in rows:
            # ── Signal 1: specificity-aware match score ──────────────────────
            match_score, match_type = _score_row_against_query(q_words, q_lower, row)

            # Fast skip: if no token/phrase overlap at all, skip embedding
            if match_score == 0.0:
                row_str_raw = _stringify_row(row)
                row_words   = set(re.findall(r'[a-zA-Z0-9]+', row_str_raw.lower()))
                if not (q_words & row_words):
                    continue   

            # ── Signal 2: BGE cosine on context-enriched row string ──────────
            row_str_rich = _stringify_row(row, section)
            cache_key    = row_str_rich[:160]
            if cache_key not in embed_cache:
                embed_cache[cache_key] = embed_doc(row_str_rich[:512])
            row_vec  = embed_cache[cache_key]
            cosine   = _cosine(q_vec, row_vec)

            # ── Combined score ───────────────────────────────────────────────
            combined = cosine * SEMANTIC_W + match_score * MATCH_W

            print(
                f"  [ITEM] cos={cosine:.3f} match={match_score:.3f}({match_type}) "
                f"final={combined:.3f} | '{_stringify_row(row)[:55]}'"
            )

            if combined > best_score:
                best_score  = combined
                best_result = {
                    "row":      row,
                    "row_text": _row_to_text(row, section),
                    "row_html": _row_to_html_table(row, section),
                    "section":  section,
                    "score":    combined,
                }

    if best_result and best_score >= ITEM_COSINE_THRESHOLD:
        print(f"  ✅ Item-level hit: score={best_score:.3f} section='{best_result['section'][:50]}'")
        return best_result

    print(f"  ℹ️  No item-level match (best={best_score:.3f} < {ITEM_COSINE_THRESHOLD})")
    return None
