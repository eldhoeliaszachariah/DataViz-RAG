import re

from config import TOPIC_SHIFT_THRESHOLD, AMBIGUOUS_TOKENS, RETRIEVAL_K
from vector_math import embed, _cosine
from hybrid_retrieval import hybrid_similarity_search


# ========= TOPIC-SHIFT DETECTION & DUAL-MODE RETRIEVAL (unchanged) ===========

def _last_user_question(chat_history: list) -> str | None:
    for msg in reversed(chat_history):
        if msg["role"] == "user":
            return msg["content"]
    return None

def _has_ambiguous_token(question: str) -> bool:
    return bool(AMBIGUOUS_TOKENS.intersection(set(question.lower().split())))

def _is_topic_switch(current_q: str, prev_q: str) -> bool:
    sim = _cosine(embed(current_q), embed(prev_q))
    print(f"  [TOPIC-SIM] current↔prev = {sim:.3f} (threshold < {TOPIC_SHIFT_THRESHOLD} = switch)")
    return sim < TOPIC_SHIFT_THRESHOLD


def condense_with_history(question: str, chat_history: list, llm, memory=None) -> str:
    if not chat_history:
        return question
    def _strip_html(text: str) -> str:
        return re.sub(r'<[^>]+>', '', text).strip()

    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Bot'}: {_strip_html(m['content'])}"
        for m in chat_history[-6:]
    )
    prompt = (
        "Given the conversation below and a follow-up question, "
        "rephrase the follow-up as a fully self-contained standalone question. "
        "Do NOT answer it — only rephrase it.\n\n"
        f"Conversation:\n{history_text}\n\n"
        f"Follow-up: {question}\n"
        "Standalone question:"
    )
    try:
        result    = llm.invoke(prompt)
        condensed = result.strip() if isinstance(result, str) else result.content.strip()
        print(f"  [CONDENSER] '{question}' → '{condensed}'")
        return condensed
    except Exception as e:
        print(f"  ⚠️  condenser failed: {e}")
        return question


def _filter_docs_for_context(docs: list) -> list:
    text_docs = [d for d in docs if d.metadata.get("type") != "json"]
    if not text_docs:
        print("  [FILTER] No text docs after filtering — falling back to all docs")
        return docs[:6]
    print(f"  [FILTER] {len(docs)} retrieved → {len(text_docs)} text docs kept for LLM context")
    return text_docs[:6]


def dual_mode_retrieval(
    question: str,
    chat_history: list,
    vectorstore,
    llm,
) -> tuple[list, str]:
    """Original dual-mode retrieval (pure dense). Kept for reference / fallback."""
    k = RETRIEVAL_K

    if not chat_history:
        print("  [RETRIEVAL] Stage 1 → no history")
        docs = vectorstore.similarity_search(question, k=k)
        return _filter_docs_for_context(docs), question

    prev_q = _last_user_question(chat_history)
    if prev_q and _is_topic_switch(question, prev_q):
        print("  [RETRIEVAL] Stage 2 → topic switch, raw retrieval")
        docs = vectorstore.similarity_search(question, k=k)
        return _filter_docs_for_context(docs), question

    if _has_ambiguous_token(question):
        print("  [RETRIEVAL] Stage 3 → follow-up with ambiguous token, condensing")
        condensed = condense_with_history(question, chat_history, llm)
        if condensed.strip().lower() != question.strip().lower():
            docs = vectorstore.similarity_search(condensed, k=k)
            return _filter_docs_for_context(docs), condensed
        print("  [RETRIEVAL] Stage 3 → condensed == raw, using raw")

    print("  [RETRIEVAL] Stage 3 → self-contained same-topic, raw retrieval")
    docs = vectorstore.similarity_search(question, k=k)
    return _filter_docs_for_context(docs), question


def hybrid_dual_mode_retrieval(
    question: str,
    chat_history: list,
    vectorstore,
    bm25_index,
    chunks: list,
    llm,
    memory=None,
) -> tuple[list, str]:
    """
    Drop-in replacement for dual_mode_retrieval that uses hybrid_similarity_search
    (FAISS dense + BM25 sparse) at every retrieval stage.

    Preserves all existing retrieval logic (no-history, topic-switch,
    ambiguous-token condensing) — only swaps the underlying search call.
    """
    k = RETRIEVAL_K

    def _search(q: str) -> list:
        return hybrid_similarity_search(q, vectorstore, bm25_index, chunks, k=k)

    if not chat_history:
        print("  [HYBRID-RETRIEVAL] Stage 1 → no history")
        docs = _search(question)
        return _filter_docs_for_context(docs), question

    prev_q = _last_user_question(chat_history)
    if prev_q and _is_topic_switch(question, prev_q):
        print("  [HYBRID-RETRIEVAL] Stage 2 → topic switch")
        docs = _search(question)
        return _filter_docs_for_context(docs), question

    if _has_ambiguous_token(question):
        print("  [HYBRID-RETRIEVAL] Stage 3 → ambiguous token, condensing")
        condensed = condense_with_history(question, chat_history, llm, memory=memory)
        if condensed.strip().lower() != question.strip().lower():
            docs = _search(condensed)
            return _filter_docs_for_context(docs), condensed
        print("  [HYBRID-RETRIEVAL] Stage 3 → condensed == raw, using raw")

    print("  [HYBRID-RETRIEVAL] Stage 3 → same-topic raw")
    docs = _search(question)
    return _filter_docs_for_context(docs), question
