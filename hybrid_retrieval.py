import re
import numpy as np

from config import HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT

# BM25 for hybrid retrieval
try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False
    print("⚠️  rank_bm25 not installed — BM25 hybrid disabled. pip install rank-bm25")


# =========================================================================
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  BM25 INDEX — built once per session alongside FAISS                ║
# ╚══════════════════════════════════════════════════════════════════════╝
# =========================================================================

def _tokenize_bm25(text: str) -> list[str]:
    """Simple whitespace + lower tokenizer for BM25."""
    return re.findall(r'[a-zA-Z0-9]+', text.lower())


def build_bm25_index(chunks: list) -> "BM25Okapi | None":
    """
    Build a BM25Okapi index over the same chunk list that goes into FAISS.
    Returns None if rank_bm25 is not installed.
    """
    if not _BM25_AVAILABLE:
        return None
    corpus = [_tokenize_bm25(c.page_content) for c in chunks]
    print(f"📊 Building BM25 index over {len(corpus)} chunks…")
    return BM25Okapi(corpus)


def hybrid_similarity_search(
    query: str,
    vectorstore,
    bm25_index,
    chunks: list,
    k: int = 10,
) -> list:
    """
    Hybrid retrieval: merges FAISS dense scores with BM25 sparse scores.

    If BM25 is unavailable, falls back to pure dense retrieval.

    Fusion strategy (Reciprocal Rank Fusion variant):
      final_score = HYBRID_DENSE_WEIGHT * dense_norm + HYBRID_SPARSE_WEIGHT * bm25_norm
    """
    # ── Dense retrieval ────────────────────────────────────────────────────────
    dense_results = vectorstore.similarity_search_with_score(query, k=k * 2)
    # FAISS returns L2 distance; convert to similarity: sim = 1 / (1 + dist)
    dense_scored: dict[int, float] = {}
    dense_docs:   dict[int, object] = {}
    for doc, dist in dense_results:
        idx = id(doc)
        sim = 1.0 / (1.0 + dist)
        dense_scored[idx] = sim
        dense_docs[idx]   = doc

    if not _BM25_AVAILABLE or bm25_index is None:
        # Pure dense fallback
        return [d for d, _ in dense_results[:k]]

    # ── Sparse retrieval (BM25) ───────────────────────────────────────────────
    q_tokens    = _tokenize_bm25(query)
    bm25_scores = bm25_index.get_scores(q_tokens)        # len == len(chunks)
    bm25_max    = float(np.max(bm25_scores)) if bm25_scores.max() > 0 else 1.0

    # ── Fuse scores ───────────────────────────────────────────────────────────
    # We need a chunk-index→doc mapping.  Build it from dense_results first,
    # then iterate all chunks for BM25.
    fused: dict[int, dict] = {}   # chunk_list_idx → {doc, score}

    # Map dense docs back to chunk list indices
    chunk_by_content: dict[str, int] = {
        c.page_content[:120]: i for i, c in enumerate(chunks)
    }
    for doc, dist in dense_results:
        key = doc.page_content[:120]
        cidx = chunk_by_content.get(key, -1)
        if cidx == -1:
            continue
        dense_sim  = 1.0 / (1.0 + dist)
        dense_norm = dense_sim  # already in [0,1]
        bm25_norm  = float(bm25_scores[cidx]) / bm25_max
        fused[cidx] = {
            "doc":   doc,
            "score": HYBRID_DENSE_WEIGHT * dense_norm + HYBRID_SPARSE_WEIGHT * bm25_norm,
        }

    # Sweep BM25 top candidates not already in fused
    top_bm25_idxs = np.argsort(bm25_scores)[::-1][:k * 2]
    for cidx in top_bm25_idxs:
        if cidx in fused:
            continue
        bm25_norm = float(bm25_scores[cidx]) / bm25_max
        fused[cidx] = {
            "doc":   chunks[cidx],
            "score": HYBRID_DENSE_WEIGHT * 0.0 + HYBRID_SPARSE_WEIGHT * bm25_norm,
        }

    sorted_fused = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    print(f"  [HYBRID] dense={len(dense_results)} bm25_top={len(top_bm25_idxs)} fused→{k}")
    return [item["doc"] for item in sorted_fused[:k]]
