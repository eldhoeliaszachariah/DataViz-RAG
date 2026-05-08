import re
import hashlib

from config import (
    IMAGE_OUTPUT_DIR,
    IMAGE_MIN_WIDTH,
    IMAGE_MIN_HEIGHT,
    IMAGE_MAX_BYTES,
    _GDRIVE_SHARE_RE,
    _GDRIVE_OPEN_RE,
    _GDRIVE_EXPORT_RE,
    IMAGE_TRIGGER_WORDS,
)
from vector_math import embed, _cosine

import fitz                       # PyMuPDF  — image extraction


# =========================================================================
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  IMAGE EXTRACTION ENGINE  (NEW — does not affect existing pipeline) ║
# ╚══════════════════════════════════════════════════════════════════════╝
# =========================================================================

def _gdrive_id_to_embed_url(file_id: str) -> str:
    """Convert any Google Drive file ID to a direct-embed URL."""
    return f"https://drive.google.com/uc?export=view&id={file_id}"


def _extract_gdrive_id(url: str) -> str | None:
    """Extract the file ID from any common Google Drive URL format."""
    for pattern in (_GDRIVE_SHARE_RE, _GDRIVE_OPEN_RE, _GDRIVE_EXPORT_RE):
        m = pattern.search(url)
        if m:
            return m.group(1)
    return None


def _save_image_bytes(img_bytes: bytes, ext: str, pdf_name: str, page_num: int, img_idx: int) -> str:
    """
    Save raw image bytes to IMAGE_OUTPUT_DIR.
    Uses a content hash so the same image embedded in multiple PDFs is saved once.
    Returns the web-accessible URL path.
    """
    digest  = hashlib.md5(img_bytes).hexdigest()[:12]
    safe_pdf = re.sub(r'[^\w\-]', '_', pdf_name.rsplit('.', 1)[0])[:40]
    filename = f"{safe_pdf}_p{page_num+1}_{img_idx}_{digest}.{ext}"
    out_path  = IMAGE_OUTPUT_DIR / filename
    if not out_path.exists():
        out_path.write_bytes(img_bytes)
        print(f"    💾 Saved image → {out_path}  ({len(img_bytes)//1024} KB)")
    else:
        print(f"    ♻️  Reusing cached image → {filename}")
    return f"/static/images/extracted/{filename}"


def extract_images_from_pdf(pdf_path: str, pdf_name: str) -> list[dict]:
    """
    Uses PyMuPDF (fitz) to extract:
      1. Directly embedded raster images (PNG/JPEG/etc.)
      2. Google Drive sharing links found in the text layer

    Returns a list of image-metadata dicts:
      {
        "type":        "embedded" | "gdrive",
        "url":         "/static/..." or "https://drive.google.com/uc?...",
        "page":        1-based page number,
        "width":       px (embedded only),
        "height":      px (embedded only),
        "caption":     nearby text snippet used as alt text / search signal,
        "keywords":    list[str] — searchable keywords for question matching,
      }

    Strategy notes
    ──────────────
    • Tiny images (< IMAGE_MIN_WIDTH × IMAGE_MIN_HEIGHT) are skipped because
      they are almost always bullets, icons, or PDF chrome.
    • Google Drive links are extracted from the text layer; their link format is
      normalised to the direct-embed URL so an <img> tag can render them.
    • A "caption" is synthesised from the 300 chars of text immediately above
      the image position on the page, giving the LLM / viz-scorer a semantic
      handle on what each image represents.
    """
    images = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  ⚠️ fitz could not open {pdf_name}: {e}")
        return images

    for page_num, page in enumerate(doc):
        page_text = page.get_text()

        # ── 1. Directly embedded raster images ─────────────────────────────
        img_index = 0
        for img_info in page.get_images(full=True):
            xref, _, w, h = img_info[0], img_info[1], img_info[2], img_info[3]
            if w < IMAGE_MIN_WIDTH or h < IMAGE_MIN_HEIGHT:
                print(f"    ⏭  Skipping tiny image xref={xref} ({w}×{h})")
                continue
            try:
                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                ext       = base_img.get("ext", "png")
                if len(img_bytes) > IMAGE_MAX_BYTES:
                    print(f"    ⏭  Skipping oversized image xref={xref} ({len(img_bytes)//1_000_000} MB)")
                    continue

                url = _save_image_bytes(img_bytes, ext, pdf_name, page_num, img_index)
                caption  = _extract_image_caption(page_text, page_num, pdf_name)
                keywords = _build_image_keywords(caption, pdf_name, page_num)
                images.append({
                    "type":     "embedded",
                    "url":      url,
                    "page":     page_num + 1,
                    "width":    w,
                    "height":   h,
                    "caption":  caption,
                    "keywords": keywords,
                    "source_pdf": pdf_name,
                })
                print(f"    🖼  Embedded image p{page_num+1} {w}×{h} → {url}")
                img_index += 1
            except Exception as e:
                print(f"    ⚠️  Error extracting xref={xref}: {e}")

        # ── 2. Google Drive links in text layer ─────────────────────────────
        for pattern in (_GDRIVE_SHARE_RE, _GDRIVE_OPEN_RE, _GDRIVE_EXPORT_RE):
            for m in pattern.finditer(page_text):
                file_id = m.group(1)
                embed_url = _gdrive_id_to_embed_url(file_id)
                caption   = _extract_image_caption(page_text, page_num, pdf_name)
                keywords  = _build_image_keywords(caption, pdf_name, page_num)
                images.append({
                    "type":     "gdrive",
                    "url":      embed_url,
                    "file_id":  file_id,
                    "page":     page_num + 1,
                    "caption":  caption,
                    "keywords": keywords,
                    "source_pdf": pdf_name,
                })
                print(f"    🔗  Google Drive image p{page_num+1} id={file_id}")

    doc.close()
    return images




def _extract_image_caption(page_text: str, page_num: int, pdf_name: str, max_chars: int = 300) -> str:
    """
    Build a caption string by taking the last ~300 chars of meaningful text
    from the page (before the image area). This gives the semantic matcher
    enough context to associate images with questions about specific topics.
    """
    # Remove JSON blobs from caption to keep it human-readable
    text = re.sub(r'\{[^}]{20,}\}', '', page_text)
    text = re.sub(r'\[[^\]]{20,}\]', '', text)
    text = text.strip()
    if not text:
        return f"Figure from {pdf_name} page {page_num+1}"
    # Take the last meaningful 300 chars
    snippet = text[-max_chars:].strip()
    # Collapse whitespace
    snippet = re.sub(r'\s+', ' ', snippet)
    return snippet


def _build_image_keywords(caption: str, pdf_name: str, page_num: int) -> list[str]:
    """
    Extract keywords from the caption for the image-question matching scorer.
    Combines: caption words (filtered), PDF name parts, and page number.
    """
    stop_words = {
        'the','a','an','is','are','was','were','and','or','in','on','at','to',
        'of','for','with','this','that','these','those','it','its','by','from',
        'has','have','been','be','as','their','which','also',
    }
    words = re.findall(r'[a-zA-Z]{3,}', caption.lower())
    keywords = [w for w in words if w not in stop_words]
    # Deduplicate preserving order
    seen = set()
    unique_kw = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique_kw.append(w)
    # Add PDF name tokens
    pdf_tokens = re.findall(r'[a-zA-Z]{3,}', pdf_name.lower())
    seen_tokens = set(seen)
    unique_kw.extend([t for t in pdf_tokens if t not in seen_tokens])
    return unique_kw[:30]


def find_matching_images(question: str, image_store: list[dict]) -> list[dict]:
    """
    Semantic matching between a user question and extracted images.

    Two-pass scoring:
      Pass 1 — keyword overlap score  (fast, exact)
      Pass 2 — embedding cosine score (slow, semantic)

    An image is included when:
      keyword_score >= 0.10  OR  embedding_score >= 0.50

    Result is sorted by embedding_score desc, capped at 3 images per response
    to keep the UI clean.
    """
    if not image_store:
        return []

    # Quick pre-filter: only attempt if question contains image-trigger words
    q_lower = question.lower()
    q_words = set(re.findall(r'[a-zA-Z]+', q_lower))
    has_trigger = bool(IMAGE_TRIGGER_WORDS & q_words)
    if not has_trigger:
        return []

    q_vec = embed(question)
    results = []

    for img in image_store:
        keywords = img.get("keywords", [])
        caption  = img.get("caption", "")

        # Pass 1 — keyword overlap
        kw_set     = set(keywords)
        overlap    = len(q_words & kw_set)
        kw_score   = overlap / max(len(kw_set), 1) if kw_set else 0.0

        # Pass 2 — embedding cosine on caption
        cap_vec    = embed(caption) if caption else None
        emb_score  = _cosine(q_vec, cap_vec) if cap_vec else 0.0

        combined   = emb_score * 0.70 + kw_score * 0.30

        print(
            f"  [IMG] p{img['page']} kw={kw_score:.2f} emb={emb_score:.3f} "
            f"combined={combined:.3f} | {img['url'][:60]}"
        )

        if emb_score >= 0.50 or kw_score >= 0.10:
            results.append({**img, "_score": combined})

    results.sort(key=lambda x: x["_score"], reverse=True)
    return results[:3]   # cap at 3 images per response
