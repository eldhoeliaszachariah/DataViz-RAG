import json

from config import SECTION_THRESHOLD, COMBINED_THRESHOLD
from vector_math import embed, _cosine
from utils import _tags_sentence


# =========================================================================
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  VIZ BLOCK MATCHING — two-gate scoring (unchanged from v10)         ║
# ╚══════════════════════════════════════════════════════════════════════╝
# =========================================================================

def _extract_json_block_summary(block: dict) -> str:
    """Converts a raw JSON block into a concise text summary for LLM context."""
    raw_json = block.get("raw_json", "[]")
    section  = block.get("section", "Data Table")
    try:
        data = json.loads(raw_json)
        if isinstance(data, list) and data:
            cols = list(data[0].keys())
            # Limit to first 10 rows to keep context window manageable
            rows_text = []
            for row in data[:10]:
                row_str = "; ".join(f"{k}: {v}" for k, v in row.items())
                rows_text.append(f"- {row_str}")
            
            summary = (
                f"\n[Source Data Context: {section}]\n"
                f"The following structured data is highly relevant to your question.\n"
                f"Columns available: {', '.join(cols)}\n"
                f"Sample Records:\n" + "\n".join(rows_text) + "\n"
            )
            return summary
    except Exception as e:
        print(f"  ⚠️ Error generating JSON block summary: {e}")
    return ""


def find_matching_viz_blocks_cached(
    question: str,
    full_json_store: dict,
    embed_cache: dict,
) -> list[dict]:
    if not full_json_store:
        return []

    q_vec   = embed(question)
    matched = []

    for block_id, info in full_json_store.items():
        section  = info.get("section", "").strip()
        tags     = info.get("tags", [])
        tags_str = _tags_sentence(tags)

        if not section:
            continue

        sec_vec  = embed_cache.get(section)  or embed(section)
        tags_vec = embed_cache.get(tags_str) if tags_str else None

        section_score  = _cosine(q_vec, sec_vec)
        tags_score     = _cosine(q_vec, tags_vec) if tags_vec is not None else 0.0
        combined_score = section_score * 0.75 + tags_score * 0.25

        print(
            f"  [VIZ] '{section[:65]}' "
            f"sec={section_score:.3f}(>={SECTION_THRESHOLD}) "
            f"tags={tags_score:.3f} comb={combined_score:.3f}(>={COMBINED_THRESHOLD})"
        )

        if section_score >= SECTION_THRESHOLD and combined_score >= COMBINED_THRESHOLD:
            matched.append({
                "block_id":       block_id,
                "section_score":  section_score,
                "combined_score": combined_score,
                "viz_type":       info.get("viz_type", "table"),
                "raw_json":       info.get("raw_json", ""),
                "section":        section,
            })

    matched.sort(key=lambda x: x["combined_score"], reverse=True)
    return matched
