from config import QA_PROMPT


# ========= ANSWER GENERATION ===========

def answer_with_docs(question: str, docs: list, llm, extra_context: str = "") -> str:
    """
    Generate an answer given retrieved docs.
    extra_context is prepended to the context window (used for item-level row injection).
    """
    doc_context = "\n\n".join(d.page_content for d in docs)
    context     = (extra_context + "\n\n" + doc_context).strip() if extra_context else doc_context
    result      = llm.invoke(QA_PROMPT.format(context=context, question=question))
    return result.strip() if isinstance(result, str) else result.content.strip()
