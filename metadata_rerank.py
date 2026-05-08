from config import META_RERANK_SECTION_W, META_RERANK_TAG_W, META_RERANK_DENSE_W
from vector_math import embed, embed_doc, _cosine


# =========================================================================
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  METADATA-AWARE RE-RANKING (runs after hybrid, before FAISS final)  ║
# ╚══════════════════════════════════════════════════════════════════════╝
# =========================================================================

def metadata_rerank(query: str, docs: list, embed_cache: dict) -> list:
    """
    Re-ranks retrieved docs by combining:
      1. Section-heading cosine similarity  (semantic match to metadata)
      2. Tag overlap cosine                 (tag relevance)
      3. Dense content cosine               (content relevance)

    Returns docs sorted by composite meta-aware score (desc), capped at 6.
    Does NOT remove any doc — just re-orders them so the most relevant
    sections surface first.
    """
    if not docs:
        return docs

    q_vec = embed(query)
    scored = []

    for doc in docs:
        section = doc.metadata.get("section_heading", "").strip()
        tags    = doc.metadata.get("embedded_tags", [])
        content = doc.page_content

        # Section score
        if section:
            sec_vec    = embed_cache.get(section) or embed(section)
            embed_cache[section] = sec_vec
            sec_score  = _cosine(q_vec, sec_vec)
        else:
            sec_score  = 0.0

        # Tag score
        tags_str = " ".join(tags)
        if tags_str:
            tag_vec   = embed_cache.get(tags_str) or embed(tags_str)
            embed_cache[tags_str] = tag_vec
            tag_score = _cosine(q_vec, tag_vec)
        else:
            tag_score = 0.0

        # Content score
        cont_vec   = embed_doc(content[:512])
        cont_score = _cosine(q_vec, cont_vec)

        composite = (
            META_RERANK_SECTION_W * sec_score +
            META_RERANK_TAG_W     * tag_score +
            META_RERANK_DENSE_W   * cont_score
        )
        scored.append((doc, composite))
        print(
            f"  [META-RERANK] sec={sec_score:.3f} tag={tag_score:.3f} "
            f"cont={cont_score:.3f} → {composite:.3f} | '{section[:50]}'"
        )

    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:6]]
