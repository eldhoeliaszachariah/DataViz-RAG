# ========= IMPORTS ===========
import uuid
import tempfile
import os
import json
import markdown

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from langchain.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# ─── Module imports ───────────────────────────────────────────────────────────
from config import (
    LLM_MODEL,
    IMAGE_OUTPUT_DIR,
)
from vector_math import embed
from utils import _tags_sentence
from hybrid_retrieval import build_bm25_index
from metadata_rerank import metadata_rerank
from item_retrieval import item_level_retrieval
from condition_filter import detect_condition_query, apply_condition_filter
from llm_condition_filter import (
    detect_condition_query_llm,
    apply_condition_filter_llm,
)
from viz_matching import (
    find_matching_viz_blocks_cached,
    _extract_json_block_summary,
)
from image_engine import (
    extract_images_from_pdf,
    find_matching_images,
)
from retrieval import hybrid_dual_mode_retrieval
from pdf_parsing import (
    get_separated_documents,
    get_chunks,
    get_vectorstore,
)
from answer import answer_with_docs


# ========= FLASK APP ===========
app = Flask(__name__)
CORS(app)
sessions: dict = {}


# ========= API ROUTES ===========

@app.route('/')
def index():
    return render_template('index.html')


# ── Serve extracted images ─────────────────────────────────────────────────────
@app.route('/static/images/extracted/<path:filename>')
def serve_extracted_image(filename):
    return send_from_directory(IMAGE_OUTPUT_DIR, filename)


@app.route('/api/upload', methods=['POST'])
def upload_documents():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected files"}), 400

    try:
        tmp_pairs, pdf_names = [], []
        for file in files:
            name = file.filename or "unknown.pdf"
            pdf_names.append(name)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            file.save(tf.name)
            tmp_pairs.append((tf.name, name))

        # ── Image extraction (runs in parallel with doc loading) ─────────────
        # We pass tmp_paths so fitz can open the real binary file.
        all_images: list[dict] = []
        for tmp_path, pdf_name in tmp_pairs:
            print(f"📸 Extracting images from {pdf_name}…")
            imgs = extract_images_from_pdf(tmp_path, pdf_name)
            all_images.extend(imgs)
            print(f"   → {len(imgs)} image(s) extracted from {pdf_name}")

        # ── Text / JSON pipeline (unchanged) ─────────────────────────────────
        file_pairs = [(open(p, 'rb'), n) for p, n in tmp_pairs]
        documents  = get_separated_documents(file_pairs, tmp_paths=tmp_pairs)

        json_count = sum(1 for d in documents if d.metadata['type'] == 'json')
        text_count = sum(1 for d in documents if d.metadata['type'] == 'text')
        print(f"📄 {len(documents)} segments ({json_count} JSON, {text_count} text)")

        full_json_store = {}
        text_chunks     = get_chunks(documents, full_json_store)
        print(f"🔪 {len(text_chunks)} chunks, {len(full_json_store)} viz block(s)")

        embed_cache: dict[str, list[float]] = {}
        for block_id, info in full_json_store.items():
            section  = info.get("section", "")
            tags_str = _tags_sentence(info.get("tags", []))
            if section and section not in embed_cache:
                embed_cache[section]  = embed(section)
            if tags_str and tags_str not in embed_cache:
                embed_cache[tags_str] = embed(tags_str)
        print(f"🔢 Pre-computed {len(embed_cache)} metadata embeddings")

        vectorstore = get_vectorstore(text_chunks)
        bm25_index  = build_bm25_index(text_chunks)   # ← NEW: BM25 sparse index
        llm         = Ollama(model=LLM_MODEL)
        session_id  = str(uuid.uuid4())

        # Create LangChain memory with window limit (k=5)
        memory = ConversationBufferWindowMemory(
            k=5, 
            memory_key='chat_history', 
            return_messages=True, 
            output_key='answer'
        )
        
        # Create a ConversationalRetrievalChain for follow-up infrastructure
        # (Uses standard vectorstore retriever; specialized hybrid search still runs in parallel)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
            memory=memory,
            return_source_documents=True
        )

        sessions[session_id] = {
            "vectorstore":     vectorstore,
            "bm25_index":      bm25_index,
            "chunks":          text_chunks,
            "llm":             llm,
            "memory":          memory,             # ← NEW
            "chain":           conversation_chain,  # ← NEW
            "history":         [],
            "full_json_store": full_json_store,
            "embed_cache":     embed_cache,
            "pdf_names":       pdf_names,
            "pdf_count":       len(pdf_names),
            "image_store":     all_images,
        }

        for fh, _ in file_pairs: fh.close()
        for p,  _ in tmp_pairs:  os.remove(p)

        return jsonify({
            "session_id":    session_id,
            "message":       f"Processed {len(pdf_names)} PDF(s) with {len(full_json_store)} viz block(s).",
            "pdf_names":     pdf_names,
            "pdf_count":     len(pdf_names),
            "json_blocks":   len(full_json_store),
            "image_count":   len(all_images),    # ← NEW: tell UI how many images extracted
        }), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/pdfs/<session_id>', methods=['GET'])
def get_pdf_list(session_id):
    if session_id not in sessions:
        return jsonify({"error": "Invalid session."}), 404
    s = sessions[session_id]
    return jsonify({"pdf_names": s.get("pdf_names", []), "pdf_count": s.get("pdf_count", 0)}), 200


@app.route('/api/chat', methods=['POST'])
def chat():
    data       = request.json
    session_id = data.get('session_id')
    question   = (data.get('question') or '').strip()

    if not session_id or not question:
        return jsonify({"error": "Missing session_id or question"}), 400
    if session_id not in sessions:
        return jsonify({"error": "Invalid session."}), 404

    try:
        session         = sessions[session_id]
        vectorstore     = session["vectorstore"]
        bm25_index      = session.get("bm25_index")       # ← NEW
        chunks          = session.get("chunks", [])        # ← NEW
        llm             = session["llm"]
        chat_history    = session["history"]
        full_json_store = session["full_json_store"]
        embed_cache     = session["embed_cache"]
        image_store     = session.get("image_store", [])

        # ── Step 0: Early question condensing for follow-ups ──────────────────
        # If the question has ambiguous tokens ("both", "it", "they", etc.) and
        # there is chat history, condense it FIRST so all downstream steps
        # (condition filter, item retrieval, hybrid retrieval) operate on the
        # resolved question instead of the vague raw one.
        from retrieval import _has_ambiguous_token, condense_with_history
        effective_question = question
        if chat_history and _has_ambiguous_token(question):
            condensed = condense_with_history(question, chat_history, llm, memory=session.get("memory"))
            if condensed.strip().lower() != question.strip().lower():
                print(f"  [EARLY-CONDENSE] '{question}' → '{condensed}'")
                effective_question = condensed
            else:
                print(f"  [EARLY-CONDENSE] condensed == raw, using raw")

        # ── Step 1: Condition-based filter — LLM-driven path (highest priority) ─
        #
        # PRIMARY  — ContextAwareJSONFilter (LLM-driven, no hardcoded patterns).
        #            Handles any field/operator/value combination by generating
        #            filter logic on-the-fly from the JSON schema.
        #
        # FALLBACK — Original hardcoded engine (regex + alias rules).
        #            Kicks in only when the LLM path is unavailable or returns None.
        #
        # ISOLATION: item-level retrieval (Step 2) is skipped whenever *either*
        #            condition path fires — even on a zero-result answer — to
        #            prevent spurious item matches from polluting the response.
        cond_result = None

        if detect_condition_query_llm(effective_question, full_json_store):
            print(f"  [LLM-COND] Detected condition query (LLM path): '{effective_question}'")
            cond_result = apply_condition_filter_llm(effective_question, full_json_store, embed_cache)
            if cond_result is None:
                print(f"  [LLM-COND] LLM path returned None — falling back to hardcoded engine")

        # Fallback: hardcoded engine (runs when LLM path is unavailable or gave None)
        if cond_result is None:
            if detect_condition_query(effective_question, full_json_store):
                print(f"  [COND-FILTER] Detected condition query (hardcoded path): '{effective_question}'")
                cond_result = apply_condition_filter(effective_question, full_json_store, embed_cache)
            else:
                print(f"  [COND-FILTER] Not a condition query — skipping both filter paths")

        # ── Step 2: Item-level retrieval (runs when NO condition filter hit) ──
        item_hit = None
        # Skip item retrieval entirely when a condition filter ran —
        # even if it returned 0 rows we want the honest "no results" answer,
        # not a spurious item match.
        if not cond_result:
            item_hit = item_level_retrieval(effective_question, full_json_store, embed_cache)

        # ── Step 3: Hybrid retrieval (dense FAISS + BM25 sparse) ────────────
        best_docs, query_used = hybrid_dual_mode_retrieval(
            effective_question, chat_history, vectorstore, bm25_index, chunks, llm, session.get("memory")
        )
        print(f"  → {len(best_docs)} docs via: '{query_used}'")

        # ── Step 4: Metadata-aware re-ranking ────────────────────────────────
        best_docs = metadata_rerank(effective_question, best_docs, embed_cache)
        print(f"  → {len(best_docs)} docs after metadata re-rank")

        # ── Step 5: Build extra LLM context ──────────────────────────────────
        extra_context = ""
        # We search for high-confidence viz blocks early to enable potential fallback context.
        matched_blocks = find_matching_viz_blocks_cached(effective_question, full_json_store, embed_cache)

        if cond_result:
            # Condition filter wins: inject filtered rows (or zero-result notice) as context.
            extra_context = cond_result["llm_context"]
        elif item_hit:
            extra_context = f"\n[Item Detail]\n{item_hit['row_text']}\n"
        elif matched_blocks:
            # Fallback: Find high-confidence semantic JSON matches even if no filter triggered.
            # We use the best matching block to provide data-driven context for the text answer.
            best_block = matched_blocks[0]
            # Lowered threshold to be more inclusive for text fallback
            if best_block.get("combined_score", 0) >= 0.60:
                print(f"  [SEMANTIC-FALLBACK] Injecting JSON data summary for block: '{best_block['section'][:40]}'")
                extra_context = _extract_json_block_summary(best_block)

        answer_text = answer_with_docs(effective_question, best_docs, llm, extra_context=extra_context)
        answer_html = markdown.markdown(answer_text)

        # ── Step 6: Inject result HTML at top of response ────────────────────
        if cond_result:
            # html_table is either a styled results table (rows found)
            # or a styled "no results" notice (no_results=True).
            answer_html = cond_result["html_table"] + answer_html
            no_res_flag = cond_result.get("no_results", False)
            row_count   = len(cond_result["filtered_rows"])
            print(
                f"  {'⚠️ ' if no_res_flag else '✅'} Condition result injected: "
                f"{row_count} rows | no_results={no_res_flag}"
            )
        elif item_hit:
            answer_html = item_hit["row_html"] + answer_html

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "bot",  "content": answer_html})
        session["history"] = chat_history

        # ── Step 7: Synchronize LangChain memory (NEW) ──────────────────────
        # This stores the turn in the k=5 window memory for follow-up rephrasing.
        # Include condition filter / item-level text so follow-ups can reference them.
        if "memory" in session:
            memory_answer_parts = []
            if cond_result and cond_result.get("llm_context"):
                memory_answer_parts.append(cond_result["llm_context"])
            if item_hit and item_hit.get("row_text"):
                memory_answer_parts.append(f"[Item Detail] {item_hit['row_text']}")
            memory_answer_parts.append(answer_text)
            memory_answer = "\n".join(memory_answer_parts)
            session["memory"].save_context({"question": question}, {"answer": memory_answer})
            print(f"  [MEMORY] Turn saved to window memory (k=5)")

        # ── Step 8: Two-gate semantic viz injection ─────────────────────────
        # (matched_blocks was already calculated above in Step 5 fallback check)

        if matched_blocks:
            viz_parts = []
            for block in matched_blocks:
                safe_json = block["raw_json"].replace("'", "&#39;")
                print(
                    f"  ✅ viz: '{block['section'][:60]}' "
                    f"type={block['viz_type']} "
                    f"sec={block['section_score']:.3f} comb={block['combined_score']:.3f}"
                )
                viz_parts.append(
                    f'<div class="auto-viz" '
                    f'data-viz-type="{block["viz_type"]}" '
                    f'data-viz-json=\'{safe_json}\'></div>'
                )
            chat_history[-1]["content"] += "<br>" + "".join(viz_parts)
        else:
            print("  ℹ️  No viz blocks matched — text-only response.")

        # ── Step 8: Image injection (unchanged) ──────────────────────────────
        matched_images = find_matching_images(question, image_store)
        if matched_images:
            img_parts = []
            for img in matched_images:
                url      = img["url"]
                caption  = img.get("caption", "")[:120]
                pg       = img.get("page", "?")
                img_type = img.get("type", "embedded")
                print(f"  🖼  Injecting image type={img_type} url={url[:60]}")
                img_parts.append(
                    f'<div class="auto-image" '
                    f'data-img-url="{url}" '
                    f'data-img-caption="{caption}" '
                    f'data-img-page="{pg}" '
                    f'data-img-type="{img_type}"></div>'
                )
            chat_history[-1]["content"] += "<br>" + "".join(img_parts)
        else:
            print("  ℹ️  No images matched.")

        return jsonify({"history": chat_history}), 200

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
