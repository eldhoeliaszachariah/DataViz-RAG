import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

from config import DEVICE, EMBEDDING_MODEL

# ─── Singleton embedding model ─────────────────────────────────────────────────
_embed_model: HuggingFaceEmbeddings | None = None

def get_embed_model() -> HuggingFaceEmbeddings:
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": DEVICE},
        )
    return _embed_model

def embed(text: str) -> list[float]:
    """
    BGE models perform best when queries are prefixed with the retrieval instruction.
    """
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return get_embed_model().embed_query(prefixed)


def embed_doc(text: str) -> list[float]:
    """Embed a document/passage without query prefix (BGE convention)."""
    return get_embed_model().embed_query(text)


# ========= VECTOR MATH ===========

def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
