import re
import json
import uuid

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from PyPDF2 import PdfReader
from langchain.vectorstores import faiss

from config import (
    _OCR_REPAIR_MAP,
    TEXT_CHUNK_SIZE,
    TEXT_CHUNK_OVERLAP,
)
from vector_math import get_embed_model


# ========= OCR / PDF TEXT REPAIR ===========

def repair_ocr_text(text: str) -> str:
    for broken, fixed in _OCR_REPAIR_MAP.items():
        text = re.sub(broken, fixed, text)
    text = re.sub(r'(?<=[\[{:,\s"])(\d+) (\d+)(?=[\]},\s"])', r'\1\2', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def sanitize_json_string(raw: str) -> str:
    return repair_ocr_text(raw)


# ========= PDF PARSING ===========

def extract_json_blocks_with_positions(raw_text: str):
    results, n, i = [], len(raw_text), 0
    while i < n:
        char = raw_text[i]
        if char in '{[':
            start, start_char = i, char
            end_char = '}' if start_char == '{' else ']'
            depth, in_str, esc = 1, False, False
            i += 1
            while i < n and depth > 0:
                c = raw_text[i]
                if esc:                     esc = False
                elif c == '\\' and in_str:  esc = True
                elif c == '"':              in_str = not in_str
                elif not in_str:
                    if   c == start_char:   depth += 1
                    elif c == end_char:     depth -= 1
                i += 1
            candidate = sanitize_json_string(raw_text[start:i])
            try:
                parsed = json.loads(candidate)
                results.append((start, i, candidate, parsed))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON parse failed at [{start}:{i}]: {e} | snippet: {candidate[:80]}")
        else:
            i += 1
    return results


def extract_section_heading(text_before: str, max_chars: int = 300) -> str:
    if not text_before:
        return ""

    _NOISE_PATTERNS = re.compile(
        r'(https?://|Reference Link|^ref\b|See also)',
        re.IGNORECASE
    )
    _HEADING_KEYWORDS = re.compile(
        r'\b(Table|List|Chart|Summary|Analysis|Details|Count|Projects|'
        r'Region|Pace|Donut|Donet|Cost|Bible|Status|Histogram|Scatter|'
        r'Line|Bar|Column|Area|Pie|Radar|Bubble|Density|Box|Frequency|'
        r'Distribution|Trend|Comparison|Grouped|Stacked)\b',
        re.IGNORECASE
    )

    lines = [
        l.strip()
        for l in text_before[-max_chars:].strip().splitlines()
        if l.strip() and not _NOISE_PATTERNS.search(l.strip())
    ]

    if not lines:
        return ""

    for line in reversed(lines):
        if _HEADING_KEYWORDS.search(line):
            return line.rstrip(":").strip()

    return lines[-1].rstrip(":").strip()


def get_separated_documents(pdf_files_with_names, tmp_paths: list[tuple[str, str]] | None = None):
    """
    Unchanged document-parsing pipeline.
    tmp_paths is used by the image extractor to get the real file path for fitz.
    """
    all_documents = []

    # Build a name→tmp_path lookup so fitz can open the file
    path_by_name: dict[str, str] = {}
    if tmp_paths:
        for tmp_path, pdf_name in tmp_paths:
            path_by_name[pdf_name] = tmp_path

    for pdf_file, pdf_name in pdf_files_with_names:
        raw_text_dirty = "".join(
            (p.extract_text() or "") + "\n"
            for p in PdfReader(pdf_file).pages
        )
        raw_text = repair_ocr_text(raw_text_dirty)

        json_blocks = extract_json_blocks_with_positions(raw_text)

        if not json_blocks:
            all_documents.append(Document(
                page_content=raw_text.strip(),
                metadata={"type": "text", "source_pdf": pdf_name}
            ))
            continue

        prev_end, text_segments = 0, []
        for (start, end, sanitized, parsed_obj) in json_blocks:
            seg = raw_text[prev_end:start].strip()
            if seg:
                text_segments.append(seg)

            embedded_section, embedded_tags = "", []
            if isinstance(parsed_obj, dict):
                meta = parsed_obj.get("meta", {})
                if isinstance(meta, dict):
                    embedded_section = meta.get("section", "").strip()
                    raw_tags = meta.get("tags", [])
                    if isinstance(raw_tags, list):
                        embedded_tags = [str(t).strip() for t in raw_tags if t]

            section_heading = (embedded_section
                               or extract_section_heading(raw_text[prev_end:start]))
            print(f"  [{pdf_name}] section='{section_heading}' tags={embedded_tags}")

            all_documents.append(Document(
                page_content=sanitized,
                metadata={
                    "type":             "json",
                    "raw_json":         sanitized,
                    "parsed":           json.dumps(parsed_obj, ensure_ascii=False),
                    "section_heading":  section_heading,
                    "embedded_tags":    embedded_tags,
                    "embedded_section": embedded_section,
                    "fallback_heading": extract_section_heading(raw_text[prev_end:start]),
                    "source_pdf":       pdf_name,
                }
            ))
            prev_end = end

        trailing = raw_text[prev_end:].strip()
        if trailing:
            text_segments.append(trailing)
        full_text = "\n\n".join(text_segments).strip()
        if full_text:
            all_documents.append(Document(
                page_content=full_text,
                metadata={"type": "text", "source_pdf": pdf_name}
            ))
    return all_documents


def custom_json_splitter(json_data, max_size=1000):
    chunks = []
    if isinstance(json_data, dict):
        cur = {}
        for k, v in json_data.items():
            tmp = {**cur, k: v}
            if len(json.dumps(tmp, ensure_ascii=False)) > max_size and cur:
                chunks.append(json.dumps(cur, indent=2, ensure_ascii=False))
                cur = {k: v}
            else:
                cur = tmp
        if cur:
            chunks.append(json.dumps(cur, indent=2, ensure_ascii=False))
    elif isinstance(json_data, list):
        cur = []
        for item in json_data:
            tmp = cur + [item]
            if len(json.dumps(tmp, ensure_ascii=False)) > max_size and cur:
                chunks.append(json.dumps(cur, indent=2, ensure_ascii=False))
                cur = [item]
            else:
                cur = tmp
        if cur:
            chunks.append(json.dumps(cur, indent=2, ensure_ascii=False))
    else:
        chunks.append(json.dumps(json_data, indent=2, ensure_ascii=False))
    return chunks


def get_chunks(documents, full_json_store):
    chunks        = []
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
        length_function=len
    )

    for doc in documents:
        if doc.metadata.get("type") != "json":
            chunks.extend(text_splitter.split_documents([doc]))
            continue

        try:
            json_data       = json.loads(doc.page_content)
            section_heading = doc.metadata.get("section_heading", "")
            embedded_tags   = doc.metadata.get("embedded_tags", [])
            source_pdf      = doc.metadata.get("source_pdf", "")

            # ── Extended viz type normalisation ─────────────────────────────
            _VIZ_ALIAS = {
                # bar family
                "bar":              "bar",
                "column":           "bar",
                "column_chart":     "bar",
                "bar_chart":        "bar",
                "horizontal_bar":   "horizontal_bar",
                "hbar":             "horizontal_bar",
                "grouped_bar":      "grouped_bar",
                "grouped":          "grouped_bar",
                "grouped_column":   "grouped_bar",
                "stacked_bar":      "stacked_bar",
                "stacked_column":   "stacked_bar",
                "stacked":          "stacked_bar",
                # line family
                "line":             "line",
                "line_chart":       "line",
                "multi_line":       "multi_line",
                "multiline":        "multi_line",
                "multi_line_chart": "multi_line",
                # area family
                "area":             "area",
                "area_chart":       "area",
                "stacked_area":     "stacked_area",
                "stacked_area_chart": "stacked_area",
                # pie / donut
                "pie":              "pie",
                "pie_chart":        "pie",
                "donut":            "donut",
                "doughnut":         "donut",
                "donut_chart":      "donut",
                "semi_donut":       "semi_donut",
                "semi-donut":       "semi_donut",
                # statistical
                "histogram":        "histogram",
                "frequency_polygon":"frequency_polygon",
                "freq_polygon":     "frequency_polygon",
                "density":          "density",
                "density_plot":     "density",
                "kde":              "density",
                "box":              "box_plot",
                "box_plot":         "box_plot",
                "boxplot":          "box_plot",
                # other
                "scatter":          "scatter",
                "scatter_plot":     "scatter",
                "bubble":           "bubble",
                "bubble_chart":     "bubble",
                "radar":            "radar",
                "spider":           "radar",
                "table":            "table",
            }
            raw_viz_type = json_data.get("type", "table").strip().lower() if isinstance(json_data, dict) else "table"
            viz_type     = _VIZ_ALIAS.get(raw_viz_type, raw_viz_type)
            data_payload = json_data.get("data", json_data) if isinstance(json_data, dict) else json_data

            json_block_id = str(uuid.uuid4())

            full_json_store[json_block_id] = {
                "raw_json": json.dumps(data_payload, ensure_ascii=False),
                "viz_type": viz_type,
                "section":  section_heading,
                "tags":     embedded_tags,
                "source":   source_pdf,
            }

            for chunk_str in custom_json_splitter(data_payload, max_size=1000):
                header_parts = []
                if section_heading: header_parts.append(f"Section: {section_heading}")
                if embedded_tags:   header_parts.append(f"Tags: {', '.join(embedded_tags)}")
                if source_pdf:      header_parts.append(f"Source: {source_pdf}")
                content = "\n".join(header_parts + [chunk_str]) if header_parts else chunk_str

                chunks.append(Document(
                    page_content=content,
                    metadata={
                        "type":            "json",
                        "viz_type":        viz_type,
                        "full_json_id":    json_block_id,
                        "section_heading": section_heading,
                        "embedded_tags":   embedded_tags,
                        "source_pdf":      source_pdf,
                        "raw_json":        chunk_str[:200],
                    }
                ))
        except Exception as e:
            print(f"⚠️  JSON chunk error: {e}")
            chunks.extend(text_splitter.split_documents([doc]))

    return chunks


def get_vectorstore(chunks):
    return faiss.FAISS.from_documents(documents=chunks, embedding=get_embed_model())


def summarize_docs(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs):
        section = doc.metadata.get("section_heading", "")
        tags    = doc.metadata.get("embedded_tags", [])
        src     = doc.metadata.get("source_pdf", "")
        preview = doc.page_content[:200].replace("\n", " ")
        line    = f"[{i+1}] Section='{section}'"
        if tags: line += f" | Tags={tags}"
        if src:  line += f" | PDF='{src}'"
        line   += f" | Preview: {preview}"
        parts.append(line)
    return "\n".join(parts)
