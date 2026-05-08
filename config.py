# ========= CONFIGURATION ===========
import torch
import re
from pathlib import Path

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
LLM_MODEL       = "llama3.2"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"   # upgraded from all-MiniLM-L6-v2

# ─── Hybrid retrieval weights ─────────────────────────────────────────────────
HYBRID_DENSE_WEIGHT  = 0.65   # FAISS dense score weight
HYBRID_SPARSE_WEIGHT = 0.35   # BM25 sparse score weight

# ─── Metadata-aware re-ranking weights ────────────────────────────────────────
META_RERANK_SECTION_W = 0.50
META_RERANK_TAG_W     = 0.30
META_RERANK_DENSE_W   = 0.20

# ─── Item-level retrieval thresholds ──────────────────────────────────────────
ITEM_COSINE_THRESHOLD   = 0.50   # minimum combined score to show item table

# ─── Condition-based filtering ────────────────────────────────────────────────
# Semantic similarity threshold for field-name detection in condition queries
CONDITION_FILTER_THRESHOLD = 0.42
# Max rows to include verbatim in LLM context (avoids token overflow)
CONDITION_CONTEXT_MAX_ROWS = 30

# ─── Image extraction settings ────────────────────────────────────────────────
IMAGE_OUTPUT_DIR = Path("static/images/extracted")    # served at /static/images/extracted/
IMAGE_MIN_WIDTH  = 80      # skip tiny icons / decorations
IMAGE_MIN_HEIGHT = 80
IMAGE_MAX_BYTES  = 50_000_000   # skip absurdly large embedded objects

# ─── Google Drive URL patterns ─────────────────────────────────────────────────
# Matches sharing links like:
#   https://drive.google.com/file/d/<ID>/view?usp=sharing
#   https://drive.google.com/open?id=<ID>
_GDRIVE_SHARE_RE  = re.compile(
    r'https?://drive\.google\.com/file/d/([A-Za-z0-9_\-]+)(?:/[^\s]*)?'
)
_GDRIVE_OPEN_RE   = re.compile(
    r'https?://drive\.google\.com/open\?id=([A-Za-z0-9_\-]+)'
)
_GDRIVE_EXPORT_RE = re.compile(
    r'https?://drive\.google\.com/uc\?(?:[^&\s]*&)*id=([A-Za-z0-9_\-]+)'
)

print(f"🚀 Using device: {DEVICE}")
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Viz similarity thresholds ────────────────────────────────────────────────
SECTION_THRESHOLD  = 0.60
COMBINED_THRESHOLD = 0.70

# ─── Topic-shift detection ────────────────────────────────────────────────────
TOPIC_SHIFT_THRESHOLD = 0.45

# ─── Retrieval k values ───────────────────────────────────────────────────────
RETRIEVAL_K = 10

# ─── Text chunking ────────────────────────────────────────────────────────────
TEXT_CHUNK_SIZE    = 1500
TEXT_CHUNK_OVERLAP = 300

# ─── Ambiguous pronoun set ────────────────────────────────────────────────────
AMBIGUOUS_TOKENS = {
    "it", "its", "they", "them", "their", "those", "that", "these",
    "he", "she", "his", "her", "same", "this", "one", "ones",
    "both", "each", "either", "neither",
}

# ─── Image keyword triggers ───────────────────────────────────────────────────
# If a question contains any of these words, attempt to render matching images.
IMAGE_TRIGGER_WORDS = {
    "image", "images", "chart", "charts", "graph", "graphs", "diagram",
    "diagrams", "picture", "pictures", "photo", "photos", "figure",
    "figures", "visual", "visuals", "overview", "screenshot", "map",
    "infographic", "show", "display", "see", "look",
}

# ─── Trigger tokens that signal a "list/filter" query ────────────────────────
_CONDITION_TRIGGER_TOKENS = {
    "list", "show", "all", "filter", "only", "which", "find", "get",
    "display", "give", "tell", "what", "where", "who", "under", "above",
    "between", "greater", "less", "more", "fewer", "details", "info",
    "details" ,"how" ,"many"
}

# ─── Keyword → condition mapping ─────────────────────────────────────────────
# Each entry: (regex_pattern, condition_type, value)

_CONDITION_KEYWORD_RULES: list[tuple[re.Pattern, str, object]] = [
    # 100% / "fully finished" / "fully completed" (multi-word forms first)
    (re.compile(r'\b(100\s*%?|fully?\s*(?:finished|completed|done))\b', re.I), "gte_pct", 100),
    # standalone "finished" / "completed" / "done" as a predicate
    # (word-boundary; must not be part of a field name like "nt_finished")
    (re.compile(r'\b(finished|completed|done)\b', re.I),                        "gte_pct", 100),
    # "not started" / "0%" / "zero"
    (re.compile(r'\b(0\s*%?|not\s+started|not\s+begun|zero|none)\b', re.I),    "eq_pct",  0),
    # "in progress" / "partial" / "ongoing" / "incomplete"
    (re.compile(r'\b(in[\s_-]?progress|partial(?:ly)?|ongoing|incomplete|started)\b', re.I), "nonzero_pct", None),
    # status: active
    (re.compile(r'\bactive\b', re.I),    "eq_str",  "Active Project"),
    # status: inactive
    (re.compile(r'\binactive\b', re.I),  "eq_str",  "Inactive Project"),
    # ">X%" / "more than X%" / "above X%" / "over X%"
    (re.compile(r'(?:more\s+than|above|over|greater\s+than|>)\s*(\d+)\s*%?', re.I), "gt_pct", "__CAPTURE__"),
    # "<X%" / "less than X%" / "below X%" / "under X%"
    (re.compile(r'(?:less\s+than|below|under|<)\s*(\d+)\s*%?', re.I),               "lt_pct", "__CAPTURE__"),
    # "at least X%"
    (re.compile(r'at\s+least\s+(\d+)\s*%?', re.I),  "gte_pct", "__CAPTURE__"),
    # "at most X%"
    (re.compile(r'at\s+most\s+(\d+)\s*%?', re.I),   "lte_pct", "__CAPTURE__"),
]

# ─── Field alias map: query keywords → JSON field name substrings ─────────────
# Keyed by lowercase alias, value is a list of substrings to look for in field names.
#
# DESIGN RULES:
#   1. Specific multi-word aliases first (they shadow their substrings in iteration).
#   2. Generic aliases like "finished"/"completed" are intentionally OMITTED because
#      they appear in almost every condition query and would match ALL percentage fields,
#      making field disambiguation impossible.  The specific aliases (nt, ot, full bible)
#      already encode the "finished" field suffix in their hint list.
_FIELD_ALIASES: dict[str, list[str]] = {
    "full stack":       ["full_stack_finished", "fullstack"],
    "fulll stack":      ["full_stack_finished", "fullstack"],   # typo tolerance
    "stack":            ["full_stack_finished", "fullstack"],
    "fullstack":        ["full_stack_finished", "fullstack"],
    "full-stack":       ["full_stack_finished", "fullstack"],
    "end to end":       ["full_stack_finished", "fullstack"],
    "ios":              ["ios_finished", "ios"],
    "ios app":          ["ios_finished", "ios"],
    "iphone":           ["ios_finished", "ios"],
    "ipad":             ["ios_finished", "ios"],
    "mobile":           ["ios_finished", "ios"],
    "frontend":         ["ios_finished", "ios"],
    "front end":        ["ios_finished", "ios"],
    "backend":          ["backend_finished", "backend"],
    "back end":         ["backend_finished", "backend"],
    "api":              ["backend_finished", "backend"],
    "server":           ["backend_finished", "backend"],
    "database":         ["backend_finished", "backend"],
    "rest api":         ["backend_finished", "backend"],
    "status":           ["officialstatus", "status"],
    "official status":  ["officialstatus", "status"],
    "active":           ["officialstatus", "status"],
    "inactive":         ["officialstatus", "status"],
    "region":           ["region", "nation"],
    "nation":           ["nation", "region"],
    "mvp":              ["ios_finished", "ios"],           # maps to iOS completion
    "minimum viable product": ["ios_finished", "ios"],
    "production":       ["full_stack_finished", "fullstack"],
    "prod":             ["full_stack_finished", "fullstack"],
    "live":             ["full_stack_finished", "fullstack"],
    "deployed":         ["full_stack_finished", "fullstack"],
    "qa":               ["ios_finished", "ios"],
    "quality assurance": ["ios_finished", "ios"],
    "testing":          ["ios_finished", "ios"],
    "sprint":           ["officialstatus", "status"],
    "agile":            ["officialstatus", "status"],
    "scrum":            ["officialstatus", "status"],
    "velocity":         ["officialstatus", "status"],
    "story points":     ["officialstatus", "status"],
    "commit":           ["ios_finished", "ios"],
    "code review":      ["ios_finished", "ios"],
    "pr":               ["ios_finished", "ios"],          # pull request
    "pull request":     ["ios_finished", "ios"],
    "bug":              ["ios_finished", "ios"],
    "issue":            ["officialstatus", "status"],
    "ticket":           ["officialstatus", "status"],
    "jira":             ["officialstatus", "status"],
    "github":           ["ios_finished", "ios"],
    "gitlab":           ["ios_finished", "ios"],
    "cicd":             ["full_stack_finished", "fullstack"],
    "ci/cd":            ["full_stack_finished", "fullstack"],
    "devops":           ["full_stack_finished", "fullstack"],
    "release":          ["full_stack_finished", "fullstack"],
    "deployment":       ["full_stack_finished", "fullstack"],
    "rollout":          ["full_stack_finished", "fullstack"],
    "ios dev":          ["ios_finished", "ios"],
    "backend dev":      ["backend_finished", "backend"],
    "fullstack dev":    ["full_stack_finished", "fullstack"],
    # NOTE: "finished" and "completed" removed — too generic, causes cross-field
    # contamination. Field specificity is carried by the aliases above.
}

# ─── Trigger vocabulary for the LLM-driven detection gate ────────────────────
# This set intentionally mirrors _CONDITION_TRIGGER_TOKENS so the gate is
# consistent — but it lives separately so either path can be tuned independently.
_LLM_FILTER_TRIGGER_TOKENS = {
    "list", "show", "all", "filter", "only", "which", "find", "get",
    "display", "give", "tell", "what", "where", "who", "highest", "lowest",
    "most", "least", "max", "min", "top", "bottom", "under", "above",
    "between", "greater", "less", "more", "fewer", "details", "info",
    "information", "status", "nation", "region", "project", "projects",
    "how", "many",
}

# ─── OCR repair map ────────────────────────────────────────────────────────────
_OCR_REPAIR_MAP = {
    r"T otal":      "Total",
    r"T icket":     "Ticket",
    r"NOR THERN":   "NORTHERN",
    r"CENTR AL":    "CENTRAL",
    r"SOU THERN":   "SOUTHERN",
    r"EUR ASIA":    "EURASIA",
    r"EUR OPE":     "EUROPE",
    r"Ac tive":     "Active",
    r"Pro ject":    "Project",
    r"Mis sing":    "Missing",
    r"Fin ished":   "Finished",
    r"Sta tus":     "Status",
    r"Re gion":     "Region",
    r"Na tion":     "Nation",
    r"Of ficial":   "Official",
}

# ─── QA prompt template ────────────────────────────────────────────────────────
from langchain.prompts import PromptTemplate

qa_template = """You are a helpful assistant. Use the following context to answer the question.
Answer ONLY what is asked. Do not output raw JSON or code blocks.
Summarize structured data briefly in plain English.

Context:
{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate.from_template(qa_template)
