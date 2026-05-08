import json
import re
from collections import Counter
from typing import List, Union

from pydantic import BaseModel, Field
import numpy as np
import jmespath

# ── LlamaIndex imports ──
from llama_index.core import (
    SummaryIndex,
    Document,
    QueryBundle,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ── LangChain PydanticOutputParser imports ──
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM

# --- CONFIGURATION ---
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# ── Pydantic models for structured LLM output ──
class FilterCondition(BaseModel):
    column: str = Field(..., description="Column name from the schema")
    operator: str = Field(..., description="One of: ==, !=, contains, in, >, <, >=, <=")
    value: Union[str, List[str]] = Field(..., description="Value to compare against as a string or list of strings for 'in' operator")

class FilterLogic(BaseModel):
    conditions: List[FilterCondition] = Field(
        default_factory=list, description="Array of conditions. Empty [] = full view."
    )
    logic: str = Field(
        default="AND", description="AND = all must match; OR = any can match"
    )

class ContextAwareJSONFilter:
    def __init__(self):
        print(f"Initializing models...")

        # ── LlamaIndex Settings (global config) ──
        print(f"Loading Embedding Model via LlamaIndex: {EMBEDDING_MODEL}...")
        self.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

        print(f"Connecting LlamaIndex to Local Ollama Model: {LLM_MODEL}...")
        self.llama_llm = Ollama(model=LLM_MODEL, temperature=0.01, request_timeout=120.0)

        Settings.llm = self.llama_llm
        Settings.embed_model = self.embed_model

        # ── LangChain LLM + PydanticOutputParser for structured output ──
        print(f"Connecting LangChain to Local Ollama Model: {LLM_MODEL}...")
        self.langchain_llm = OllamaLLM(model=LLM_MODEL, temperature=0.01)
        self.pydantic_parser = PydanticOutputParser(pydantic_object=FilterLogic)

        # ── LlamaIndex SummaryIndex/VectorStoreIndex for schema-aware retrieval ──
        self.schema_index = None
        self.schema_retriever = None
        
        # ── Cache for column embeddings to avoid redundant calls ──
        self.col_embeddings_cache = {}

        print("Models ready.\n")

    def _extract_context(self, input_json):
        """Dynamically extracts wrapper, metadata, and smart schema for long JSONs.
        Also builds a LlamaIndex SummaryIndex from schema documents for retrieval-augmented
        filter generation."""
        original_type = input_json.get("type", "table")
        original_meta = input_json.get("meta", {})
        data = input_json.get("data", [])
        
        if not data:
            return original_type, original_meta, data, {}, ""
            
        columns = list(data[0].keys())
        schema_info = {}
        
        # Extract unique values safely for long JSONs
        for col in columns:
            unique_vals = set()
            for row in data[:2000]: 
                val = str(row[col])
                if len(val) < 60: 
                    unique_vals.add(val)
            # Increase unique value sample size for better LLM context
            schema_info[col] = sorted(list(unique_vals))[:25]
            
        # --- DYNAMIC STRUCTURE ANALYSIS ---
        # Detect if the data is a "Key-Value Summary" (e.g., Donut charts, Count tables)
        dynamic_hint = ""
        if len(columns) == 2:
            col1, col2 = columns[0], columns[1]
            
            # Check if Column 1 is strings and Column 2 is numbers
            is_col1_str = all(isinstance(row[col1], str) for row in data[:10])
            is_col2_num = all(isinstance(row[col2], (int, float)) for row in data[:10])
            
            # Check the reverse
            is_col2_str = all(isinstance(row[col2], str) for row in data[:10])
            is_col1_num = all(isinstance(row[col1], (int, float)) for row in data[:10])

            if is_col1_str and is_col2_num:
                dynamic_hint = f"STRUCTURAL HINT: This data is a Key-Value summary. The column '{col1}' contains category identifiers. If the user asks for a specific category, you MUST use the 'contains' operator on the '{col1}' column."
            elif is_col2_str and is_col1_num:
                dynamic_hint = f"STRUCTURAL HINT: This data is a Key-Value summary. The column '{col2}' contains category identifiers. If the user asks for a specific category, you MUST use the 'contains' operator on the '{col2}' column."

        # ── Build LlamaIndex VectorStoreIndex from schema documents ──
        # Use VectorStoreIndex for semantic retrieval if there are many columns
        from llama_index.core import VectorStoreIndex
        
        schema_docs = []
        for col, vals in schema_info.items():
            col_desc = f"Column: {col}. Sample values: {', '.join(str(v) for v in vals)}"
            schema_docs.append(Document(text=col_desc, metadata={"column_name": col}))

        if schema_docs:
            self.schema_index = VectorStoreIndex.from_documents(schema_docs)
            # Use top 5 most relevant columns (or all if fewer than 5)
            self.schema_retriever = self.schema_index.as_retriever(similarity_top_k=min(5, len(columns)))
        else:
            self.schema_index = None
            self.schema_retriever = None

        return original_type, original_meta, data, columns, schema_info, dynamic_hint

    def _ground_query(self, query, columns):
        """Uses LlamaIndex QueryBundle + HuggingFaceEmbedding to map the user query
        to the correct columns via cosine similarity."""
        query_bundle = QueryBundle(query_str=query)
        query_emb = np.array(query_bundle.embedding or self.embed_model.get_text_embedding(query))
        
        # Use cached column embeddings
        col_embs_list = []
        for col in columns:
            if col not in self.col_embeddings_cache:
                self.col_embeddings_cache[col] = self.embed_model.get_text_embedding(col)
            col_embs_list.append(self.col_embeddings_cache[col])
        col_embs = np.array(col_embs_list)
        
        # Normalize
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        col_embs = col_embs / (np.linalg.norm(col_embs, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(col_embs, query_emb)
        relevant_indices = np.argsort(similarities)[::-1]
        return [columns[i] for i in relevant_indices]

    @staticmethod
    def _is_numeric(val):
        try:
            float(str(val).replace('%', '').replace(',', '').strip())
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _to_float(val):
        try:
            return float(str(val).replace('%', '').replace(',', '').strip())
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _text_tokens(text):
        """Normalize text into lightweight comparable tokens."""
        split_camel = re.sub(r'([a-z])([A-Z])', r'\1 \2', str(text))
        raw_tokens = re.findall(r'[a-z0-9]+', split_camel.lower().replace('_', ' '))
        tokens = []
        for token in raw_tokens:
            if len(token) > 4 and token.endswith("ing"):
                token = token[:-3]
            elif len(token) > 3 and token.endswith("ed"):
                token = token[:-2]
            elif len(token) > 3 and token.endswith("s"):
                token = token[:-1]
            tokens.append(token)
        return tokens

    @staticmethod
    def _is_numeric_column(col, data, sample_size=50):
        sample_vals = [row.get(col) for row in data[:sample_size] if row.get(col) is not None]
        if not sample_vals:
            return False
        numeric_count = sum(1 for val in sample_vals if ContextAwareJSONFilter._is_numeric(val))
        return numeric_count > len(sample_vals) * 0.5

    def _numeric_columns(self, columns, data):
        return [col for col in columns if self._is_numeric_column(col, data)]

    def _column_text_score(self, text, col):
        """Score direct token evidence between query/context text and a column name."""
        query_tokens = set(self._text_tokens(text))
        col_tokens = set(self._text_tokens(col))
        if not query_tokens or not col_tokens:
            return 0.0

        overlap = query_tokens & col_tokens
        score = float(len(overlap))

        # Exact short-token matches such as OT/NT should dominate generic words.
        short_matches = [tok for tok in overlap if len(tok) <= 3]
        score += 2.0 * len(short_matches)

        col_phrase = " ".join(self._text_tokens(col))
        text_phrase = " ".join(self._text_tokens(text))
        if col_phrase and col_phrase in text_phrase:
            score += 4.0

        return score

    def _find_numeric_column(self, query, columns, data, context_hint=None):
        """Find the best numeric column using token evidence first, then embeddings.

        If the query does not name a numeric field, the first numeric column in the
        table acts as the default measure. This avoids letting embeddings pick a
        component column for generic words like "completion" or "progress".
        """
        numeric_cols = self._numeric_columns(columns, data)
        if not numeric_cols:
            return None

        for text in (context_hint, query):
            if not text:
                continue
            lexical_scores = [(self._column_text_score(text, col), col) for col in numeric_cols]
            best_score, best_col = max(lexical_scores, key=lambda item: item[0])
            if best_score > 0:
                return best_col

        # Embeddings are useful only after restricting candidates to numeric columns.
        for text in (context_hint, query):
            if not text:
                continue
            try:
                sims = self._embedding_similarity(text, numeric_cols)
                best_idx = int(np.argmax(sims))
                sorted_sims = np.sort(sims)
                margin = float(sorted_sims[-1] - sorted_sims[-2]) if len(sorted_sims) > 1 else float(sorted_sims[-1])
                if sims[best_idx] > 0.42 and margin > 0.04:
                    return numeric_cols[best_idx]
            except Exception:
                pass

        return numeric_cols[0]

    def _embedding_similarity(self, query_text, columns):
        """Compute cosine similarity between a query text and column names
        using LlamaIndex HuggingFaceEmbedding."""
        query_emb = np.array(self.embed_model.get_text_embedding(query_text))
        
        col_embs_list = []
        for col in columns:
            if col not in self.col_embeddings_cache:
                self.col_embeddings_cache[col] = self.embed_model.get_text_embedding(col)
            col_embs_list.append(self.col_embeddings_cache[col])
        col_embs = np.array(col_embs_list)
        
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        col_embs = col_embs / (np.linalg.norm(col_embs, axis=1, keepdims=True) + 1e-10)
        return np.dot(col_embs, query_emb)

    def _dynamic_ground_query(self, query, columns, data):
        """Purely data-driven mapping of query terms to columns and values. No hardcoded patterns."""
        words = query.lower().split()
        query_str = query.lower()
        
        # 1. Column name grounding (abbreviations)
        for i, word in enumerate(words):
            if len(word) <= 3:
                for col in columns:
                    if word in col.lower().replace('_', ' ').split():
                        words[i] = col
                        break
        grounded_query = " ".join(words)
        
        # 2. Value-to-Column mapping (Instructional) + OR Logic Detection (combined pass)
        value_hints = []
        col_value_map = {}  # Tracks which values map to which column for OR detection
        
        # Pre-scan ALL potential values in the query to find the best grounded columns
        # Priority: Longer strings first to avoid sub-phrase overlap (e.g. "ASIA R1" before "ASIA")
        potential_matches = []
        for col in columns:
            samples = set()
            # Increase scan range to 2000 rows to find items not in the first 50
            for row in data[:2000]:
                val = str(row.get(col, '')).strip()
                if val and len(val) < 60:
                    samples.add(val)
            
            for s_val in samples:
                s_val_lower = s_val.lower()
                if re.match(r'^[\d.]+%?$', s_val_lower):
                    continue 
                
                matched = False
                if len(s_val_lower) > 2:
                    if re.search(r'\b' + re.escape(s_val_lower) + r'\b', query_str):
                        matched = True
                    elif re.search(r'\b' + re.escape(s_val_lower) + r's?\b', query_str):
                        matched = True
                
                if matched:
                    potential_matches.append((len(s_val_lower), s_val, col))

        # Sort by length descending to match longest phrases first
        potential_matches.sort(key=lambda x: x[0], reverse=True)
        
        matched_query_spans = []
        for _, s_val, col in potential_matches:
            s_val_lower = s_val.lower()
            # Find all occurrences in query
            for match in re.finditer(r'\b' + re.escape(s_val_lower) + r'\b', query_str):
                span = match.span()
                # Check if this span overlaps with an already matched (longer) span
                if not any(span[0] < ms[1] and span[1] > ms[0] for ms in matched_query_spans):
                    matched_query_spans.append(span)
                    value_hints.append(f"Query term '{s_val}' refers to Column '{col}'")
                    if col not in col_value_map:
                        col_value_map[col] = []
                    if s_val not in col_value_map[col]:
                        col_value_map[col].append(s_val)
        
        hint_str = "\n".join(value_hints[:10]) if value_hints else "None"
        
        # --- NEW: OR Logic Detection ---
        # 1. Explicit "OR" word in query
        explicit_or = bool(re.search(r'\b(or|either|any of)\b', query_str))
        
        # 2. Multiple values in same column (already handles internal OR)
        or_hint = "None"
        for col, vals in col_value_map.items():
            if len(vals) >= 2:
                or_hint = f"OR_LOGIC_DETECTED: The user listed multiple specific items: {json.dumps(vals)}. Create a separate condition for EACH value on column '{col}' and set logic to 'OR'."
                break
        
        # If explicit OR was found and items are in different columns, trigger OR logic
        if explicit_or and len(col_value_map) >= 2:
             or_hint = f"OR_LOGIC_DETECTED: The user explicitly used 'OR' logic. Set the global logic field to 'OR'."
        
        # --- NEW: Explicit Range Detection ---
        range_hint = "None"
        # Pattern 1: "between X and Y"
        range_match = re.search(r'between\s+([\d.]+)%?\s+and\s+([\d.]+)%?', query_str)
        if range_match:
            low, high = range_match.group(1), range_match.group(2)
            range_hint = f"RANGE_DETECTED: User specified a range between {low} and {high}. Create TWO conditions with AND logic: >= {low} AND <= {high} on the relevant numeric column."
        else:
            # Pattern 2: "X-Y%" or "X - Y%" (e.g., "0-10%", "40 - 60%")
            range_match2 = re.search(r'([\d.]+)\s*%?\s*[-–]\s*([\d.]+)\s*%?', query_str)
            if range_match2:
                low, high = range_match2.group(1), range_match2.group(2)
                range_hint = f"RANGE_DETECTED: User specified a range from {low} to {high}. Create TWO conditions with AND logic: >= {low} AND <= {high} on the relevant numeric column."
        
        # --- Structured range info for validation layer ---
        range_info_struct = None
        if range_match:
            range_info_struct = {"low": float(range_match.group(1)), "high": float(range_match.group(2)), "column": None}
        elif range_match2:
            range_info_struct = {"low": float(range_match2.group(1)), "high": float(range_match2.group(2)), "column": None}

        # Determine the target column for range (best numeric column from query)
        if range_info_struct:
            range_info_struct["column"] = self._find_numeric_column(query, columns, data)

        return grounded_query, hint_str, or_hint, range_hint, col_value_map, range_info_struct

    def _retrieve_schema_context(self, query):
        """Uses LlamaIndex SummaryIndex retriever to fetch the most relevant
        schema column documents for the given query."""
        if not self.schema_retriever:
            return ""
        try:
            query_bundle = QueryBundle(query_str=query)
            nodes = self.schema_retriever.retrieve(query_bundle)
            retrieved = []
            for node in nodes:
                col_name = node.node.metadata.get("column_name", "")
                text = node.node.text
                retrieved.append(f"[{col_name}] {text}")
            return "\n".join(retrieved)
        except Exception as e:
            print(f"[WARN] LlamaIndex schema retrieval failed: {e}")
            return ""

    def _generate_filter_json(self, query, meta, schema_info, relevant_columns, dynamic_hint, or_hint="None", range_hint="None"):
        """Uses LlamaIndex schema retrieval + LangChain PydanticOutputParser for
        robust structured filter generation with multi-layer fallback."""

        # ── Step 1: LlamaIndex schema retrieval for context enrichment ──
        retrieved_schema = self._retrieve_schema_context(query)
        schema_section = json.dumps(schema_info, indent=2)
        if retrieved_schema:
            schema_section += f"\n\nRetrieved relevant columns:\n{retrieved_schema}"

        system_prompt = """You are a JSON filter generator. Output a JSON object with "conditions" array and "logic" field.

GROUNDING INSTRUCTIONS (MANDATORY):
{grounding_hints}

STRUCTURAL RULES:
- "conditions" is an array of objects with "column", "operator", "value" fields. Empty [] for full view.
- "logic" must be "AND" (all conditions must match) or "OR" (any condition can match).
- Valid operators: "==", "!=", "contains", "in", ">", "<", ">=", "<="
- Use "in" ONLY when the user explicitly lists multiple items by name.
- NEVER list items from the sample data that the user didn't ask for.

HALLUCINATION WARNING:
- You are a FILTER, not a search result.
- DO NOT pick specific project names from sample data unless they are mentioned in the user query.
- Use the 'OfficialStatus' column for categories like "active" or "inactive".

MULTI-CONDITION RULES:
- Every distinct requirement in the query (status, location, number) MUST be its own condition.
- DO NOT combine two numeric values into one operator list.
- If the user specifies a range (e.g. 30-70%), create TWO conditions: >= 30 AND <= 70.

SEMANTIC RULES:
- "Incomplete" -> operator "<", value "100" on a percentage column.
- "Moderate progress" -> >= 30 AND <= 70.
- "Half completed" or "half done" -> >= 40 AND <= 60 on the relevant percentage column.
- "Nearly complete" or "almost done" -> >= 90 on the relevant percentage column.
- "Early stage" or "just started" -> <= 10 on the relevant percentage column.
- "Good progress" -> >= 50 on the relevant percentage column.
- "Partially completed" -> > 0 AND < 100.
- "Barely started" -> <= 10.
- "Substantial completion" -> >= 50.
- "Stalled" -> == 0 on the relevant percentage column.
- When a parenthetical range or comparison follows a term, the range OVERRIDES any default.

OR LOGIC RULES:
- If 'OR_LOGIC_DETECTED' instruction is present, use "logic": "OR" and include all requested items.

RANGE RULES:
- If 'RANGE_DETECTED' instruction is present, create TWO conditions (>= and <=) and use "logic": "AND".

AVAILABLE SCHEMA & SAMPLE DATA:
{schema}

STRUCTURAL HINTS:
{dynamic_hint}

OR LOGIC INSTRUCTION:
{or_hint}

RANGE INSTRUCTION:
{range_hint}

CRITICAL: If GROUNDING INSTRUCTIONS say a query term refers to a column, you MUST use that column.

{format_instructions}"""

        format_instructions = self.pydantic_parser.get_format_instructions()

        formatted_prompt = system_prompt.format(
            schema=schema_section,
            dynamic_hint=dynamic_hint if dynamic_hint else "Standard Table Data: Use exact string matching for categories.",
            grounding_hints=meta.get("grounding_hints", "None"),
            or_hint=or_hint,
            range_hint=range_hint,
            format_instructions=format_instructions
        )

        final_prompt = formatted_prompt + f"\n\nUser Request: {query}\n\nJSON Filter:"

        # ── Step 2: LangChain LLM + PydanticOutputParser with retry ──
        for attempt in range(3):
            try:
                raw_output = self.langchain_llm.invoke(final_prompt)

                # ── Layer 1: PydanticOutputParser (strictest) ──
                try:
                    result = self.pydantic_parser.parse(raw_output)
                    return json.loads(result.model_dump_json())
                except Exception as parse_err:
                    if attempt == 0:
                        print(f"[WARN] PydanticOutputParser attempt {attempt+1} failed: {parse_err}")

                # ── Layer 2: Manual JSON extraction with brace-matching ──
                try:
                    json_str = raw_output.strip()
                    if "```json" in json_str: json_str = json_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_str: json_str = json_str.split("```")[1].split("```")[0].strip()

                    start_idx = json_str.find('{')
                    if start_idx != -1:
                        depth = 0
                        for i in range(start_idx, len(json_str)):
                            if json_str[i] == '{': depth += 1
                            elif json_str[i] == '}':
                                depth -= 1
                                if depth == 0:
                                    json_str = json_str[start_idx:i+1]
                                    break

                    parsed_json = json.loads(json_str)
                    if "conditions" in parsed_json:
                        valid_conds = []
                        for c in parsed_json["conditions"]:
                            if isinstance(c, dict) and "column" in c and "operator" in c:
                                valid_conds.append(c)
                        parsed_json["conditions"] = valid_conds
                    return parsed_json
                except Exception:
                    pass

                # ── Layer 3: Try Pydantic model direct construction ──
                try:
                    json_str = raw_output.strip()
                    if "```json" in json_str: json_str = json_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_str: json_str = json_str.split("```")[1].split("```")[0].strip()
                    start_idx = json_str.find('{')
                    if start_idx != -1:
                        depth = 0
                        for i in range(start_idx, len(json_str)):
                            if json_str[i] == '{': depth += 1
                            elif json_str[i] == '}':
                                depth -= 1
                                if depth == 0:
                                    json_str = json_str[start_idx:i+1]
                                    break
                    parsed = json.loads(json_str)
                    # Construct Pydantic model to validate and normalize
                    filter_logic = FilterLogic(**parsed)
                    return json.loads(filter_logic.model_dump_json())
                except Exception:
                    continue

            except Exception as e:
                print(f"[WARN] LLM invocation attempt {attempt+1} failed: {e}")
                continue

        # ── Layer 4 (last resort): regex extraction ──
        try:
            cond_matches = re.findall(r'\{"column":\s*"([^"]+)",\s*"operator":\s*"([^"]+)",\s*"value":\s*"([^"]+)"\}', raw_output)
            if cond_matches:
                conds = [{"column": c, "operator": o, "value": v} for c, o, v in cond_matches]
                return {"conditions": conds, "logic": "AND"}
        except Exception:
            pass

        print(f"[ERROR] Failed to parse LLM JSON after 3 attempts with all fallback layers.")
        return None

    def _apply_filter(self, data, filter_json):
        """Dynamically executes the filter, handling strings, ints, and percentages safely."""
        if not filter_json or "conditions" not in filter_json or not data:
            return data

        conditions = filter_json["conditions"]
        logic = filter_json.get("logic", "AND").upper()
        
        # Empty conditions = full view
        if not conditions:
            return data
        
        filtered_data = []

        for row in data:
            row_matches = []
            for cond in conditions:
                col = cond.get("column")
                op = cond.get("operator")
                val = cond.get("value")

                if col not in row:
                    row_matches.append(False)
                    continue

                row_val = row[col]

                if op in ['>', '<', '>=', '<=', '==', '!=']:
                    row_val_num = self._to_float(row_val)
                    val_num = self._to_float(val)
                    if row_val_num is not None and val_num is not None:
                        if op == ">": row_matches.append(row_val_num > val_num)
                        elif op == "<": row_matches.append(row_val_num < val_num)
                        elif op == ">=": row_matches.append(row_val_num >= val_num)
                        elif op == "<=": row_matches.append(row_val_num <= val_num)
                        elif op == "==": row_matches.append(row_val_num == val_num)
                        elif op == "!=": row_matches.append(row_val_num != val_num)
                        continue
                    if op in ['>', '<', '>=', '<=']:
                        row_matches.append(False)
                        continue

                # Support for 'in' operator with list or string value
                if op.lower() == "in":
                    str_row = str(row_val).lower().strip()
                    if isinstance(val, list):
                        row_matches.append(any(str(v).lower().strip() == str_row for v in val))
                    else:
                        row_matches.append(str(val).lower().strip() == str_row)
                    continue

                str_row = str(row_val).lower().strip()
                str_val = str(val).lower().strip()
                
                if op == "==": row_matches.append(str_row == str_val)
                elif op == "!=": row_matches.append(str_row != str_val)
                elif op == "contains": row_matches.append(str_val in str_row)
                else: row_matches.append(False)

            if logic == "AND" and all(row_matches): filtered_data.append(row)
            elif logic == "OR" and any(row_matches): filtered_data.append(row)

        return filtered_data

    def _apply_jmespath_filter(self, data, filter_json):
        """Uses JMESPath declarative JSON query language for filtering.
        Converts our filter_json conditions into JMESPath expressions.
        Falls back to _apply_filter if JMESPath can't express the query.
        """
        if not filter_json or "conditions" not in filter_json or not data:
            return data

        conditions = filter_json.get("conditions", [])
        logic = filter_json.get("logic", "AND").upper()

        if not conditions:
            return data

        # Build JMESPath expression from conditions
        # JMESPath: [?col=='val'], [?col>'num'], etc.
        jmespath_exprs = []
        has_complex_op = False

        for cond in conditions:
            col = cond.get("column", "")
            op = cond.get("operator", "")
            val = cond.get("value", "")

            if not col or not op:
                continue

            # Escape column names with special chars for JMESPath
            col_escaped = f'"{col}"' if any(c in col for c in [' ', '-', '.']) else col

            if op == "==":
                if self._is_numeric_column(col, data) and self._is_numeric(val):
                    has_complex_op = True
                    continue
                if isinstance(val, (int, float)):
                    jmespath_exprs.append(f"[?{col_escaped}==`{val}`]")
                else:
                    jmespath_exprs.append(f"[?{col_escaped}=='{val}']")
            elif op == "!=":
                if self._is_numeric_column(col, data) and self._is_numeric(val):
                    has_complex_op = True
                    continue
                if isinstance(val, (int, float)):
                    jmespath_exprs.append(f"[?{col_escaped}!=`{val}`]")
                else:
                    jmespath_exprs.append(f"[?{col_escaped}!='{val}']")
            elif op == "contains":
                jmespath_exprs.append(f"[?contains({col_escaped}, '{val}')]")
            elif op == "in":
                # "in" operator: value is a list, match any
                if isinstance(val, list):
                    # JMESPath requires backtick-wrapped JSON literals for arrays
                    vals_json = json.dumps(val)
                    jmespath_exprs.append(f"[?contains(`{vals_json}`, {col_escaped})]")
                else:
                    has_complex_op = True
            elif op in ('>', '<', '>=', '<='):
                # Numeric comparison — need to handle percentage strings
                # JMESPath can't parse "53%" as a number, so we need Python filtering
                has_complex_op = True
            else:
                has_complex_op = True

        # If we have numeric or complex operators, fall back to Python filtering
        if has_complex_op:
            return self._apply_filter(data, filter_json)

        # Combine JMESPath expressions
        if not jmespath_exprs:
            return data

        if logic == "AND":
            # Chain filters: data | [?cond1] | [?cond2]
            expr = " | ".join(jmespath_exprs)
        else:
            # OR logic: harder in JMESPath — use Python fallback
            return self._apply_filter(data, filter_json)

        try:
            result = jmespath.search(expr, data)
            return result if result is not None else []
        except Exception:
            # Fall back to Python filtering
            return self._apply_filter(data, filter_json)

    def _extract_numeric_conditions(self, query, columns, data):
        """Extract explicit numeric comparisons from query text (e.g., '>50%', '<10%', '>=30').
        Also handles parenthetical patterns like '(0-10%)', '(>90%)'.
        Returns list of (column, operator, value) tuples."""
        q_lower = query.lower()
        numeric_conds = []

        consumed_spans = []

        def normalize_op(op):
            op = op.strip()
            if op == "=":
                return "=="
            if op == "=>":
                return ">="
            if op == "=<":
                return "<="
            return op

        def context_around(start, end, before=60, after=60):
            left = q_lower[max(0, start - before):start]
            right = q_lower[end:min(len(q_lower), end + after)]
            return f"{left} {right}".strip()

        def add_condition(start, end, op, num_val, context=None):
            op = normalize_op(op)
            if op not in ('>', '<', '>=', '<=', '==', '!='):
                return
            target_col = self._find_numeric_column(query, columns, data, context_hint=context)
            if target_col:
                numeric_conds.append((target_col, op, num_val))
                consumed_spans.append((start, end))

        def span_consumed(start, end):
            return any(start < used_end and end > used_start for used_start, used_end in consumed_spans)

        # Mark inline ranges so their endpoint percentages are not treated as equality filters.
        for match in re.finditer(r'between\s+[\d.]+\s*%?\s+and\s+[\d.]+\s*%?', q_lower):
            consumed_spans.append(match.span())
        for match in re.finditer(r'[\d.]+\s*%?\s*[-–]\s*[\d.]+\s*%?', q_lower):
            consumed_spans.append(match.span())

        # Parenthetical ranges like "(30-70%)" or "(0-10%)".
        paren_range = re.compile(r'\(\s*([\d.]+)\s*%?\s*[-–]\s*([\d.]+)\s*%?\s*\)')
        for match in paren_range.finditer(q_lower):
            low, high = match.group(1), match.group(2)
            context = context_around(match.start(), match.end())
            add_condition(match.start(), match.end(), ">=", low, context)
            add_condition(match.start(), match.end(), "<=", high, context)

        # Parenthetical comparisons like "(>90%)" or "(<20%)".
        paren_comp = re.compile(r'\(\s*(>=|<=|==|!=|=>|=<|>|<|=)\s*([\d.]+)\s*%?\s*\)')
        for match in paren_comp.finditer(q_lower):
            op, num_val = match.group(1), match.group(2)
            context = context_around(match.start(), match.end())
            add_condition(match.start(), match.end(), op, num_val, context)

        # Explicit symbolic comparisons: "> 50%", "<10%", ">=30".
        num_pattern = re.compile(r'(>=|<=|==|!=|=>|=<|>|<|=)\s*([\d.]+)\s*%?')
        for match in num_pattern.finditer(q_lower):
            if span_consumed(match.start(), match.end()):
                continue
            op, num_val = match.group(1), match.group(2)
            context = context_around(match.start(), match.end())
            add_condition(match.start(), match.end(), op, num_val, context)

        # Word-based comparisons: "over 50", "under 20", "greater than 30".
        word_ops = re.compile(r'\b(greater than|less than|more than|over|under|above|below)\s+([\d.]+)\s*%?')
        op_map = {"over": ">=", "under": "<=", "above": ">", "below": "<",
                  "greater than": ">", "less than": "<", "more than": ">"}
        for match in word_ops.finditer(q_lower):
            if span_consumed(match.start(), match.end()):
                continue
            word_op, num_val = match.group(1), match.group(2)
            context = context_around(match.start(), match.end())
            add_condition(match.start(), match.end(), op_map.get(word_op, ">="), num_val, context)

        # Bare percentages with clear field context mean equality, e.g. "NT finished 100%".
        bare_percent = re.compile(r'(?<![-–\d.])([\d.]+)\s*%')
        for match in bare_percent.finditer(q_lower):
            if span_consumed(match.start(), match.end()):
                continue
            before = q_lower[max(0, match.start() - 12):match.start()]
            after = q_lower[match.end():min(len(q_lower), match.end() + 12)]
            if re.search(r'(between|over|under|above|below|than|[><=]|[-–])\s*$', before):
                continue
            if re.search(r'^\s*(and|[-–])', after):
                continue
            context = context_around(match.start(), match.end())
            add_condition(match.start(), match.end(), "==", match.group(1), context)

        # Deduplicate
        seen = set()
        unique_conds = []
        for item in numeric_conds:
            key = (item[0], item[1], item[2])
            if key not in seen:
                seen.add(key)
                unique_conds.append(item)

        return unique_conds

    @staticmethod
    def _condition_key(cond):
        value = cond.get("value")
        if isinstance(value, list):
            value = tuple(str(v).lower().strip() for v in value)
        else:
            value_num = ContextAwareJSONFilter._to_float(value)
            value = value_num if value_num is not None else str(value).lower().strip()
        return (cond.get("column"), cond.get("operator"), value)

    def _add_condition_once(self, conditions, condition):
        key = self._condition_key(condition)
        if key not in {self._condition_key(c) for c in conditions}:
            conditions.append(condition)

    def _extract_semantic_conditions(self, query, columns, data, numeric_conds, range_info):
        """Translate broad completion/progress language into numeric bounds.

        Explicit numbers always win; this only runs when the query does not already
        provide a numeric comparison or range.
        """
        if numeric_conds or range_info:
            return []

        target_col = self._find_numeric_column(query, columns, data)
        if not target_col:
            return []

        q_norm = " ".join(self._text_tokens(query))
        semantic_rules = [
            (("half completed", "half complete", "half done", "halfway", "half way"), [(">=", "40"), ("<=", "60")]),
            (("moderate progress", "moderate completion", "medium progress"), [(">=", "30"), ("<=", "70")]),
            (("partially completed", "partially complete", "partly completed", "partial completion"), [(">", "0"), ("<", "100")]),
            (("incomplete", "not complete", "not completed", "unfinished", "not finished"), [("<", "100")]),
            (("very little progress", "little progress", "low progress", "minimal progress"), [("<=", "20")]),
            (("barely started", "just started", "early stage", "early stages"), [("<=", "10")]),
            (("nearly finished", "nearly complete", "almost done", "almost complete"), [(">=", "90")]),
            (("substantial completion", "substantial progress", "good progress"), [(">=", "50")]),
            (("stalled", "no progress", "zero progress"), [("==", "0")]),
        ]

        for phrases, bounds in semantic_rules:
            for phrase in phrases:
                phrase_norm = " ".join(self._text_tokens(phrase))
                if re.search(r'\b' + re.escape(phrase_norm) + r'\b', q_norm):
                    return [(target_col, op, value) for op, value in bounds]

        return []

    def _build_deterministic_filter(self, col_value_map, range_info, numeric_conds, semantic_conds, explicit_or=False):
        """Build a filter from high-confidence query-grounded signals."""
        conditions = []

        for col, vals in col_value_map.items():
            if len(vals) >= 2:
                self._add_condition_once(
                    conditions,
                    {"column": col, "operator": "in", "value": vals}
                )
            elif len(vals) == 1:
                self._add_condition_once(
                    conditions,
                    {"column": col, "operator": "==", "value": vals[0]}
                )

        if range_info and range_info.get("column"):
            self._add_condition_once(
                conditions,
                {"column": range_info["column"], "operator": ">=", "value": str(range_info["low"])}
            )
            self._add_condition_once(
                conditions,
                {"column": range_info["column"], "operator": "<=", "value": str(range_info["high"])}
            )

        for col, op, value in numeric_conds:
            self._add_condition_once(
                conditions,
                {"column": col, "operator": op, "value": str(value)}
            )

        for col, op, value in semantic_conds:
            self._add_condition_once(
                conditions,
                {"column": col, "operator": op, "value": str(value)}
            )

        if not conditions:
            return None
            
        # Use OR logic if explicitly detected and items are from different columns
        logic = "OR" if explicit_or and len(col_value_map) >= 2 else "AND"
        
        return {"conditions": conditions, "logic": logic}

    def _value_present_in_column(self, data, col, value, operator="=="):
        value_norm = str(value).lower().strip()
        for row in data[:2000]:
            row_norm = str(row.get(col, "")).lower().strip()
            if operator == "contains":
                if value_norm in row_norm:
                    return True
            elif row_norm == value_norm:
                return True
        return False

    def _sanitize_generated_filter(self, filter_json, query, columns, data, col_value_map):
        """Remove LLM conditions that are not grounded in the query or data."""
        if not filter_json or "conditions" not in filter_json:
            return filter_json

        query_norm = query.lower()
        grounded_values = {
            col: {str(v).lower().strip() for v in vals}
            for col, vals in col_value_map.items()
        }

        sanitized = []
        changed = False
        for cond in filter_json.get("conditions", []):
            col = cond.get("column")
            op = cond.get("operator")
            value = cond.get("value")

            if col not in columns or not op:
                changed = True
                print(f"  [REPAIR] Removed invalid condition: {cond}")
                continue

            if op in ('>', '<', '>=', '<='):
                if not self._is_numeric_column(col, data) or not self._is_numeric(value):
                    changed = True
                    print(f"  [REPAIR] Removed non-numeric comparison: {cond}")
                    continue
                sanitized.append(cond)
                continue

            if op in ("==", "!=") and self._is_numeric_column(col, data) and self._is_numeric(value):
                sanitized.append(cond)
                continue

            if op not in ("==", "!=", "contains", "in"):
                changed = True
                print(f"  [REPAIR] Removed unsupported operator condition: {cond}")
                continue

            values = value if isinstance(value, list) else [value]
            kept_values = []
            for raw_val in values:
                val_norm = str(raw_val).lower().strip()
                directly_mentioned = bool(re.search(r'\b' + re.escape(val_norm) + r'\b', query_norm))
                grounded_for_col = val_norm in grounded_values.get(col, set())
                present = self._value_present_in_column(data, col, raw_val, operator=op if op == "contains" else "==")
                if present and (directly_mentioned or grounded_for_col):
                    kept_values.append(raw_val)

            if op == "in":
                if kept_values:
                    sanitized.append({**cond, "value": kept_values})
                    if len(kept_values) != len(values):
                        changed = True
                        print(f"  [REPAIR] Trimmed ungrounded values from condition: {cond}")
                else:
                    changed = True
                    print(f"  [REPAIR] Removed ungrounded list condition: {cond}")
            else:
                if kept_values:
                    sanitized.append(cond)
                else:
                    changed = True
                    print(f"  [REPAIR] Removed ungrounded string condition: {cond}")

        if changed:
            filter_json["conditions"] = sanitized
        return filter_json

    def _validate_and_repair_filter(self, filter_json, query, columns, data,
                                     col_value_map, range_info, numeric_conds, semantic_conds=None, explicit_or=False):
        """Post-LLM validation: repair filter JSON when the LLM drops conditions."""
        if not filter_json or "conditions" not in filter_json:
            return filter_json

        semantic_conds = semantic_conds or []
        filter_json = self._sanitize_generated_filter(filter_json, query, columns, data, col_value_map)
        conditions = filter_json["conditions"]
        logic = filter_json.get("logic", "AND").upper()
        repaired = False

        # --- FIX 0: Impossible AND on same column ---
        # If logic is AND and there are multiple == conditions on the same column,
        # it's impossible (e.g., Projects=='Aunga' AND Projects=='Hado').
        # Convert to "in" operator with list of values, keeping AND for other conditions.
        if logic == "AND":
            from collections import Counter
            eq_cols = [c.get("column") for c in conditions
                       if c.get("operator") in ("==", "contains") and c.get("column")]
            col_counts = Counter(eq_cols)
            for col, count in col_counts.items():
                if count >= 2:
                    # Check if these are all different values (impossible AND)
                    vals = [c.get("value") for c in conditions
                            if c.get("column") == col and c.get("operator") in ("==", "contains")]
                    if len(set(str(v) for v in vals)) >= 2:
                        # Replace multiple == conditions with a single "in" condition
                        conditions = [c for c in conditions if not (
                            c.get("column") == col and c.get("operator") in ("==", "contains")
                        )]
                        conditions.append({"column": col, "operator": "in", "value": vals})
                        # Keep logic as AND (the "in" handles the OR internally)
                        repaired = True
                        print(f"  [REPAIR] Impossible AND on '{col}': converted to 'in' with {len(vals)} values")
                        break

        # --- FIX 1: OR logic with multiple items ---
        # If col_value_map has 2+ values on same column but LLM didn't include them all,
        # add a single "in" condition with all values (works with AND logic)
        for col, vals in col_value_map.items():
            if len(vals) < 2:
                continue
            # Check if the LLM already handled all values (via "in" or multiple "==")
            existing_vals_for_col = set()
            for c in conditions:
                if c.get("column") == col and c.get("operator") in ("==", "contains", "in"):
                    v = c.get("value", "")
                    if isinstance(v, list):
                        existing_vals_for_col.update(str(x).lower().strip() for x in v)
                    else:
                        existing_vals_for_col.add(str(v).lower().strip())

            requested_vals = set(v.lower().strip() for v in vals)
            if requested_vals.issubset(existing_vals_for_col):
                continue  # LLM already handled all values

            # Remove any existing single-value conditions for this column
            conditions = [c for c in conditions if not (
                c.get("column") == col and c.get("operator") in ("==", "contains")
            )]

            # Add a single "in" condition with all values
            conditions.append({"column": col, "operator": "in", "value": vals})
            logic = "AND"  # "in" handles OR internally, so AND is correct for other conditions

            repaired = True
            print(f"  [REPAIR] OR logic: added 'in' with {len(vals)} values for column '{col}'")

        # --- FIX 2: Range conditions missing upper bound ---
        if range_info and range_info != "None":
            range_low = range_info.get("low")
            range_high = range_info.get("high")
            range_col = range_info.get("column")

            if range_low is not None and range_high is not None and range_col:
                # Check if both bounds exist in conditions
                has_low = any(
                    c.get("column") == range_col and c.get("operator") in (">=", ">")
                    for c in conditions
                )
                has_high = any(
                    c.get("column") == range_col and c.get("operator") in ("<=", "<")
                    for c in conditions
                )
                if not has_low:
                    conditions.append({"column": range_col, "operator": ">=", "value": str(range_low)})
                    repaired = True
                    print(f"  [REPAIR] Range: added >= {range_low} on '{range_col}'")
                if not has_high:
                    conditions.append({"column": range_col, "operator": "<=", "value": str(range_high)})
                    repaired = True
                    print(f"  [REPAIR] Range: added <= {range_high} on '{range_col}'")
                # Ensure logic is AND for range
                if logic == "OR" and len(conditions) > 2:
                    # Keep OR only if there are multi-value conditions
                    logic = "AND"

        # --- FIX 3: Numeric conditions from query text ---
        # If the query explicitly says ">50%" or "<10%" and the LLM dropped it or put it on wrong column, add/fix it
        for target_col, op, num_val in numeric_conds:
            # Check if this condition already exists (approximately)
            exists = False
            for c in conditions:
                if c.get("column") == target_col and c.get("operator") in ('>', '<', '>=', '<='):
                    try:
                        existing_val = float(str(c.get("value", "")).replace('%', '').strip())
                        new_val = float(num_val)
                        # If the LLM already generated a numeric condition on this column
                        # with a similar value, trust it
                        if abs(existing_val - new_val) < 5:
                            exists = True
                            break
                    except ValueError:
                        pass

            if not exists:
                # Also check if LLM put a conflicting condition on a DIFFERENT numeric column
                # e.g., LLM generated OT_Finished > 90 but should be Full_Bible_Finished > 90
                # Remove conflicting conditions from other numeric columns
                conflicting = []
                for c in conditions:
                    if c.get("column") != target_col and c.get("operator") in ('>', '<', '>=', '<='):
                        # Check if this is a similar comparison that might be on wrong column
                        try:
                            existing_val = float(str(c.get("value", "")).replace('%', '').strip())
                            new_val = float(num_val)
                            if abs(existing_val - new_val) < 5:
                                conflicting.append(c)
                        except ValueError:
                            pass
                for c in conflicting:
                    conditions.remove(c)
                    repaired = True
                    print(f"  [REPAIR] Removed wrong-column condition: {c}")

                conditions.append({"column": target_col, "operator": op, "value": num_val})
                logic = "AND"
                repaired = True
                print(f"  [REPAIR] Numeric: added {op} {num_val} on '{target_col}'")

        # --- FIX 4: Semantic conditions from fuzzy completion/progress language ---
        for target_col, op, num_val in semantic_conds:
            exists = False
            for c in conditions:
                if c.get("column") == target_col and c.get("operator") == op:
                    try:
                        existing_val = float(str(c.get("value", "")).replace('%', '').strip())
                        new_val = float(num_val)
                        if abs(existing_val - new_val) < 1e-9:
                            exists = True
                            break
                    except ValueError:
                        pass
            if not exists:
                conditions.append({"column": target_col, "operator": op, "value": str(num_val)})
                logic = "AND"
                repaired = True
                print(f"  [REPAIR] Semantic: added {op} {num_val} on '{target_col}'")

        # --- FIX 5: Single-value string conditions from col_value_map ---
        # If col_value_map has a single value for a column but LLM didn't include it
        # (e.g., "Active projects with full bible > 50%" — LLM only added numeric condition)
        for col, vals in col_value_map.items():
            if len(vals) != 1:
                continue  # Multi-value handled by FIX 1
            val = vals[0]
            # Check if this value is already in conditions
            already_exists = False
            for c in conditions:
                if c.get("column") == col:
                    c_val = c.get("value", "")
                    if isinstance(c_val, list):
                        if val.lower().strip() in [str(v).lower().strip() for v in c_val]:
                            already_exists = True
                            break
                    elif str(c_val).lower().strip() == val.lower().strip():
                        already_exists = True
                        break
            if not already_exists:
                conditions.append({"column": col, "operator": "==", "value": val})
                logic = "AND"
                repaired = True
                print(f"  [REPAIR] String value: added {col} == '{val}'")

        if repaired:
            filter_json["conditions"] = conditions
            # If explicit OR was detected across columns, force logic to OR
            if explicit_or and len(col_value_map) >= 2:
                filter_json["logic"] = "OR"
            else:
                filter_json["logic"] = logic

        return filter_json

    def _apply_advanced_logic(self, data, query, columns):
        """Handles highest, lowest, top N, bottom N, and full view logic in Python for robustness."""
        q_lower = query.lower()
        
        # 1. Full View Trigger
        full_view_words = ["full", "all", "complete", "everything", "whole", "dataset", "table"]
        # Use more restrictive check: only trigger if the query is a short broad request
        # like "show all", "full table" — NOT "full bible finished > 50%"
        full_view_trigger = False
        if any(f" {w} " in f" {q_lower} " for w in full_view_words) and len(q_lower.split()) < 5:
            # Additional check: the word should be the main intent, not a column name reference
            # e.g., "full" in "full table" is intent, but "full" in "full bible" is a column reference
            has_comparison = any(op in q_lower for op in ['>', '<', '>=', '<=', '=', 'between', 'less', 'more', 'greater', 'above', 'below', 'under', 'over'])
            has_specific_filter = any(w in q_lower for w in ['inactive', 'active', 'status', 'region', 'nation', 'finished', 'completed', 'percent'])
            if not has_comparison and not has_specific_filter:
                full_view_trigger = True
        if full_view_trigger:
             print(f"  [PYTHON-LOGIC] Full view detected")
             return data

        # 2. Top N / Bottom N / Highest / Lowest Detection
        top_n_match = re.search(r'(top|bottom)\s+(\d+)', q_lower)
        has_highest_keyword = any(w in q_lower for w in ["highest", "most", "max", "maximum", "top"])
        has_lowest_keyword = any(w in q_lower for w in ["lowest", "least", "min", "minimum", "bottom"])
        
        if top_n_match or has_highest_keyword or has_lowest_keyword:
            direction = "top"
            n = None
            
            # Determine direction and N from regex if present (takes priority)
            if top_n_match:
                direction = top_n_match.group(1)  # "top" or "bottom"
                n = int(top_n_match.group(2))
            elif has_highest_keyword:
                direction = "top"
            elif has_lowest_keyword:
                direction = "bottom"
            
            is_highest = (direction == "top")
            
            # Find the best numeric column via embedding similarity
            relevant_cols = self._ground_query(query, columns)
            target_col = None
            for col in relevant_cols:
                sample_vals = [row.get(col) for row in data[:20] if row.get(col) is not None]
                if not sample_vals: continue
                numeric_count = 0
                for v in sample_vals:
                    try:
                        float(str(v).replace('%', '').replace(',', '').strip())
                        numeric_count += 1
                    except ValueError: pass
                if numeric_count > len(sample_vals) * 0.5:
                    target_col = col
                    break
            
            if target_col:
                # --- NEW: Pre-filter data if query contains explicit data values ---
                pre_filtered_data = data
                _, val_hints, _, _, _, _ = self._dynamic_ground_query(query, columns, data)
                
                if val_hints and val_hints != "None":
                    pre_filters = []
                    for hint in val_hints.split("\n"):
                        if "Query term '" in hint and "Column '" in hint:
                            val = hint.split("Query term '")[1].split("'")[0]
                            col = hint.split("Column '")[1].split("'")[0]
                            # Skip the target_col to avoid filtering on the sort column
                            if col != target_col:
                                pre_filters.append((col, val))
                    
                    if pre_filters:
                        for col, val in pre_filters:
                            pre_filtered_data = [
                                row for row in pre_filtered_data 
                                if val.lower().strip() in str(row.get(col, '')).lower().strip()
                            ]
                
                if n is not None:
                    print(f"  [PYTHON-LOGIC] Top/Bottom {n} detection on '{target_col}' (pre-filtered to {len(pre_filtered_data)} rows)")
                else:
                    print(f"  [PYTHON-LOGIC] {'Highest' if is_highest else 'Lowest'} detection on '{target_col}' (pre-filtered to {len(pre_filtered_data)} rows)")
                    
                def sort_key(row):
                    try:
                        # Clean percentage and comma before conversion
                        val_str = str(row.get(target_col, 0)).replace('%', '').replace(',', '').strip()
                        return float(val_str)
                    except ValueError:
                        return -1.0 if is_highest else float('inf') # Push errors to end
                
                sorted_data = sorted(pre_filtered_data, key=sort_key, reverse=is_highest)
                
                if n is not None:
                    # Top N / Bottom N: return exactly N items
                    return sorted_data[:n]
                else:
                    # Highest/Lowest without N: return all rows that share the top/bottom value
                    if sorted_data:
                        top_val = sort_key(sorted_data[0])
                        results = [row for row in sorted_data if sort_key(row) == top_val]
                        return results

        return None

    def process_query(self, input_json, query):
        """Main entry point: Handles advanced Python logic first, then falls back to LLM."""
        print(f"{'='*50}\nQuery: '{query}'")
        
        orig_type, orig_meta, data, columns, schema_info, dynamic_hint = self._extract_context(input_json)
        
        if not data:
            return input_json

        # --- STEP 1: ADVANCED PYTHON LOGIC (Highest/Lowest/TopN/BottomN/Full) ---
        advanced_result = self._apply_advanced_logic(data, query, columns)
        if advanced_result is not None:
            print(f"[RESULT] Advanced Python logic matched {len(advanced_result)} records.")
            return {
                "type": orig_type,
                "meta": orig_meta,
                "data": advanced_result
            }

        # --- STEP 2: STANDARD LLM FILTERING ---
        relevant_cols = self._ground_query(query, columns)
        print(f"[INFO] Target Columns: {relevant_cols[:3]}")
        
        grounded_query, val_hints, or_hint, range_hint, col_value_map, range_info_struct = self._dynamic_ground_query(query, columns, data)
        
        # Detect explicit OR logic for the final construction
        explicit_or = bool(re.search(r'\b(or|either|any of)\b', query.lower()))
        
        # If value hints identified specific columns, ensure they are in relevant_cols
        if val_hints and val_hints != "None":
            print(f"[INFO] Value Hints: {val_hints}")
            for hint in val_hints.split("\n"):
                if "Column '" in hint:
                    col_name = hint.split("Column '")[1].split("'")[0]
                    if col_name in columns and col_name not in relevant_cols:
                        relevant_cols.insert(0, col_name) # High priority
        
        if grounded_query != query.lower():
            print(f"[INFO] Grounded Query: '{grounded_query}'")
        
        if or_hint != "None":
            print(f"[INFO] OR Hint: {or_hint}")
        if range_hint != "None":
            print(f"[INFO] Range Hint: {range_hint}")
        
        orig_meta["grounding_hints"] = val_hints if val_hints else "None"
        
        # Extract explicit numeric conditions from query text (e.g., ">50%", "<10%")
        numeric_conds = self._extract_numeric_conditions(query, columns, data)
        if numeric_conds:
            print(f"[INFO] Numeric Conds: {numeric_conds}")

        semantic_conds = self._extract_semantic_conditions(
            query, columns, data, numeric_conds, range_info_struct
        )
        if semantic_conds:
            print(f"[INFO] Semantic Conds: {semantic_conds}")

        deterministic_filter = self._build_deterministic_filter(
            col_value_map, range_info_struct, numeric_conds, semantic_conds, explicit_or=explicit_or
        )
        if deterministic_filter:
            print(f"[INFO] Deterministic Logic: {json.dumps(deterministic_filter)}")
            filtered_data = self._apply_jmespath_filter(data, deterministic_filter)
            print(f"[RESULT] Matched {len(filtered_data)} records.")
            return {
                "type": orig_type,
                "meta": orig_meta,
                "data": filtered_data
            }

        # Pass or_hint and range_hint to the LLM
        filter_json = self._generate_filter_json(
            grounded_query, orig_meta, schema_info, relevant_cols, dynamic_hint,
            or_hint=or_hint, range_hint=range_hint
        )
        print(f"[INFO] Generated Logic: {json.dumps(filter_json)}")

        # Post-LLM validation: repair filter if LLM dropped conditions
        filter_json = self._validate_and_repair_filter(
            filter_json, query, columns, data,
            col_value_map, range_info_struct, numeric_conds, semantic_conds, explicit_or=explicit_or
        )
        if filter_json:
            print(f"[INFO] Final Logic: {json.dumps(filter_json)}")
        
        filtered_data = self._apply_jmespath_filter(data, filter_json)
        print(f"[RESULT] Matched {len(filtered_data)} records.")
        
        return {
            "type": orig_type,
            "meta": orig_meta,
            "data": filtered_data
        }


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    engine = ContextAwareJSONFilter()