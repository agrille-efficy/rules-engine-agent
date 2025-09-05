from typing import List, Dict, Any  
import pandas as pd
from langchain_core.documents import Document
import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(r'C:\Users\axel.grille\Documents\rules-engine-agent\Agent\.env')

NAME_KEYS = ["name", "fullname", "title"]
EMAIL_KEYS = ["email", "mail"]
PHONE_KEYS = ["phone", "mobile", "tel", "telephone"]
DATE_KEYS = ["date", "created", "updated"]
STATUS_KEYS = ["status", "state"]

class TableIngestionChunkBuilder:
    def __init__(self, dico_data: Dict[str, Any], enable_llm_summary: bool = True):  
        self.dico_data = dico_data
        # Always derive DataFrames from dico_data
        tables_raw = dico_data['data']['tables']
        fields_raw = dico_data['data']['fields']
        tables_df = pd.DataFrame.from_dict(tables_raw.values(), orient='columns')
        fields_df = pd.DataFrame.from_dict(fields_raw.values(), orient='columns')
        self.tables_df = tables_df[tables_df['stblKTable'].notna()] if 'stblKTable' in tables_df.columns else tables_df
        self.fields_df = fields_df[fields_df['sfldKTable'].notna()] if 'sfldKTable' in fields_df.columns else fields_df
        self.cleaned_tables = [t for t in tables_raw.values() if t['stblKind'] in ['E','R']]
        self.fields_index = self.fields_df.groupby('sfldKTable') if 'sfldKTable' in self.fields_df.columns else {}
        self._table_by_k = {t['stblKTable']: t for t in self.cleaned_tables}
        self.enable_llm_summary = enable_llm_summary and bool(os.getenv("OPENAI_API_KEY"))
        self.client = None
        if self.enable_llm_summary:
            try:
                self.client = OpenAI()
            except Exception:
                self.enable_llm_summary = False
        # Precompute relationship graph from linkedBeans
        self.relationship_graph = self._build_relationship_graph()
        # Precompute FK parent candidates per table from linkedBeans
        self.parent_candidates = {tbl['stblName']: self.relationship_graph.get(tbl['stblName'], []) for tbl in self.cleaned_tables}

    def generate(self) -> List[Document]:
        docs: List[Document] = []
        for table in self.cleaned_tables:
            ktable = table['stblKTable']
            if pd.isna(ktable):
                continue
            try:
                tbl_fields = self.fields_index.get_group(ktable)
            except KeyError:
                continue
            content = self._build_table_chunk(table, tbl_fields)
            metadata = {
                "chunk_type": "table_ingestion_profile",
                "primary_table": table.get('stblName'),
                "table_code": table.get('stblCode'),
                "table_kind": ('Relation' if table.get('stblKind') == 'R' else 'Entity'),
                "field_count": len(tbl_fields),
            }
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def generate_relationship_path_chunks(self, max_paths: int = 50) -> List[Document]:
        """Generate relationship path chunks capturing multi-table insertion order."""
        paths = []
        visited_pairs = set()
        for parent, children in self.relationship_graph.items():
            for child in children:
                pair_key = (parent, child)
                if pair_key in visited_pairs:
                    continue
                visited_pairs.add(pair_key)
                content = self._build_relationship_path_content(parent, child)
                metadata = {
                    "chunk_type": "relationship_path",
                    "primary_table": parent,
                    "related_tables": [child],
                    "path_length": 2,
                }
                paths.append(Document(page_content=content, metadata=metadata))
                if len(paths) >= max_paths:
                    return paths
        return paths

    def _build_table_chunk(self, table: Dict[str, Any], tbl_fields: pd.DataFrame) -> str:
        name = table.get('stblName')
        code = table.get('stblCode')
        kind_flag = table.get('stblKind')
        kind = 'Relation' if kind_flag == 'R' else 'Entity'
        linked = table.get('linkedBeans', {}) or {}

        required, optional, foreign_keys, all_defs = self._classify_fields(tbl_fields)
        parent_dependencies = self._infer_parents(foreign_keys)

        purpose_line = self._infer_purpose(table, tbl_fields)

        parts = []
        parts.append(f"=== TABLE INGESTION PROFILE: {name} ({code}) ===")
        parts.append(f"Type: {kind} Table")
        parts.append(f"Purpose: {purpose_line}")
        parts.append("")
        parts.append("# SCHEMA & CONSTRAINTS")
        parts.append(f"Total fields: {len(tbl_fields)}")
        if required:
            parts.append(f"Required fields ({len(required)}): " + ", ".join(f[0] for f in required))
        else:
            parts.append("Required fields: (none detected)")
        if foreign_keys:
            parts.append(f"Foreign keys ({len(foreign_keys)}): " + ", ".join(f[0] for f in foreign_keys))
        if parent_dependencies:
            parts.append("Depends on parent tables: " + ", ".join(sorted(parent_dependencies)))
        if kind == 'Relation':
            parts.append("Insertion rule: parent entity rows must exist before inserting this relation row.")
        
        primary_key_info = self._detect_primary_key(tbl_fields)
        if primary_key_info:
            parts.append(f"Primary Key: {primary_key_info['field_name']} ({primary_key_info['code']})" + (" [AUTO_INCREMENT]" if primary_key_info.get('auto_increment') else ""))
        parts.append("")

        parts.append("# FIELD DEFINITIONS")
        for f in all_defs:
            extra = []
            if primary_key_info and f['code'] == primary_key_info['code']:
                extra.append('PK')
            if f['fk_flag']:
                extra.append('FK')
            marker = (' [' + ','.join(extra) + ']') if extra else ''
            parts.append(f"- {f['name']} ({f['code']}): {f['type']}{f['size']} | {'REQUIRED' if not f['nullable'] else 'NULLABLE'}{marker}")
        parts.append("")

        parts.append("# VALIDATION RULES")
        parts.append("- All REQUIRED fields must be provided.")
        if foreign_keys:
            parts.append("- Foreign key fields must reference existing primary keys in their parent tables.")
        parts.append("- Respect maximum sizes for VARCHAR-like fields.")
        parts.append("- Ensure data types (numeric/date/text) align with field expectations.")
        parts.append("")

        parts.append("# FOREIGN KEY RELATIONSHIPS")
        if foreign_keys:
            for fk_name, fk_code, fk_target in foreign_keys:
                parts.append(f"- {fk_name} ({fk_code}) -> references parent table '{fk_target or 'UNKNOWN'}'")
        else:
            parts.append("(No foreign keys detected)")
        parts.append("")

        parts.append("# COMMON DATA MAPPING EXAMPLES")
        mapping_examples = self._mapping_examples(name, required, optional)
        for ex in mapping_examples:
            parts.append(ex)
        parts.append("")

        parts.append("# INSERTION ORDER GUIDANCE")
        if kind == 'Entity' and foreign_keys:
            parts.append("Insert parent tables first, then this entity (due to FK dependencies).")
        elif kind == 'Entity':
            parts.append("This entity can be inserted independently (no FK dependencies detected).")
        else:
            parts.append("Insert all referenced parent entity rows before relation rows.")
        parts.append("")

        parts.append("# RISK / AMBIGUITY FLAGS")
        if not required:
            parts.append("- No required fields detected: verify if primary key is auto-generated.")
        if len(required) > 15:
            parts.append("- High number of required fields: consider staged ingestion.")
        if any('date' in f[1].lower() for f in required):
            parts.append("- Required date fields: ensure proper date formatting (YYYY-MM-DD).")
        parts.append("")

        return "\n".join(parts)

    def _classify_fields(self, tbl_fields: pd.DataFrame):
        required = []  
        optional = []
        foreign_keys = []  
        all_defs = []
        parent_name_lower = {n.lower() for n in self.parent_candidates.get(self._current_table_name(tbl_fields), [])}
        for _, row in tbl_fields.iterrows():
            name = row.get('sfldName')
            code = row.get('sfldCode')
            ftype = row.get('sfldType')
            size_val = row.get('sfldSize')
            size = f"({int(size_val)})" if isinstance(size_val, (int,float)) and size_val else ""
            nullable = bool(row.get('sfldNullable', True))
            # Heuristic PK detection: exact 'Key' code or endswith 'Id'
            is_pk = code in ('Key', 'KEY') or (code and code.lower().endswith('id') and not nullable)
            # Heuristic FK detection:
            is_fk = False
            target_guess = None
            if code and code.startswith('K') and code.lower() != 'key':
                is_fk = True
                core = code[1:].lower()
                target_guess = self._match_table_by_code_fragment(core)
            elif code and any(parent in code.lower() for parent in parent_name_lower):
                is_fk = True
                target_guess = self._match_table_by_code_fragment(code.lower())
            if is_fk and not target_guess:
                target_guess = None
            if is_fk:
                foreign_keys.append((name, code, target_guess))
            elif not nullable and not is_pk:
                required.append((name, code, None))
            else:
                optional.append((name, code, None))
            all_defs.append({
                'name': name,
                'code': code,
                'type': ftype,
                'size': size,
                'nullable': nullable,
                'fk_flag': ' [FK]' if is_fk else ''
            })
        return required, optional, foreign_keys, all_defs

    def _current_table_name(self, tbl_fields: pd.DataFrame) -> str:
        try:
            k = tbl_fields.iloc[0]['sfldKTable']
            tbl = self._table_by_k.get(k)
            return tbl.get('stblName') if tbl else ''
        except Exception:
            return ''

    def _detect_primary_key(self, tbl_fields: pd.DataFrame):
        for _, row in tbl_fields.iterrows():
            code = row.get('sfldCode')
            name = row.get('sfldName')
            if code in ('Key', 'KEY') or (code and code.lower().endswith('id')):
                auto_inc = self._infer_auto_increment(row)
                return {"field_name": name, "code": code, "auto_increment": auto_inc}
        return None

    def _infer_auto_increment(self, row) -> bool:
        # Heuristic: numeric type + non-null + code equals 'Key'
        ftype = str(row.get('sfldType', '')).lower()
        code = str(row.get('sfldCode', ''))
        nullable = bool(row.get('sfldNullable', True))
        if code.lower() == 'key' and (('int' in ftype) or ('number' in ftype)) and not nullable:
            return True
        return False

    def _infer_purpose(self, table: Dict[str, Any], tbl_fields: pd.DataFrame) -> str:
        if not self.enable_llm_summary or not self.client:
            return '<ADD BUSINESS PURPOSE HERE>'
        field_names = [str(n) for n in tbl_fields['sfldName'].head(12)]
        prompt = (
            "You are analyzing a CRM/ERP database table. "
            "Given table name and sample field names, produce a concise business purpose (max 18 words).\n" \
            f"Table: {table.get('stblName')}\nFields: {', '.join(field_names)}\nPurpose:"
        )
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=50
            )
            text = resp.choices[0].message.content.strip()
            return text
        except Exception:
            return '<ADD BUSINESS PURPOSE HERE>'

    def _build_relationship_graph(self) -> Dict[str, List[str]]:
        graph = {}
        for t in self.cleaned_tables:
            name = t.get('stblName')
            beans = t.get('linkedBeans', {}) or {}
            rels = []
            for _, arr in beans.items():
                for bean in arr:
                    target = bean.get('detailEntity')
                    if target and target != name:
                        rels.append(target)
            graph[name] = sorted(set(rels))
        return graph

    def _build_relationship_path_content(self, parent: str, child: str) -> str:
        parts = []
        parts.append(f"=== RELATIONSHIP PATH: {parent} -> {child} ===")
        parts.append("Purpose: Multi-table ingestion ordering and foreign key dependency guidance.")
        parts.append("")
        parts.append("# WORKFLOW")
        parts.append(f"1. Insert parent record into {parent} (capture primary key).")
        parts.append(f"2. Insert dependent record into {child} referencing {parent}'s primary key.")
        parts.append("")
        parts.append("# VALIDATION")
        parts.append("- Ensure parent row exists before child insert.")
        parts.append("- Use transaction or rollback strategy if child insert fails.")
        parts.append("- Enforce referential integrity constraints.")
        parts.append("")
        parts.append("# SAMPLE MAPPING LOGIC")
        parts.append(f"If source dataset has columns for both {parent} and {child}, partition records by parent uniqueness, insert parents first, then children.")
        return "\n".join(parts)

    def _infer_parents(self, foreign_keys):
        return {fk[2] for fk in foreign_keys if fk[2]}

    def _mapping_examples(self, table_name: str, required, optional):
        examples = []
        take_optional = optional[:5]
        header_codes = [r[1] for r in required] + [o[1] for o in take_optional]
        header_codes = [c for c in header_codes if c]
        if header_codes:
            examples.append("Example CSV headers: " + ",".join(header_codes))
        # Heuristic source mapping lines
        examples.append("User source column 'Full Name' -> map to field containing keyword: name")
        examples.append("User source column 'EmailAddress' -> map to field containing: email")
        examples.append("User source column 'Phone' -> map to field containing: phone")
        examples.append("If ambiguity (e.g., multiple phone fields), flag for human review instead of guessing.")
        # Sample INSERT pattern (simplified)
        req_codes = [r[1] for r in required if r[1]]
        if req_codes:
            placeholders = ", ".join([f":{c.lower()}" for c in req_codes])
            examples.append("Minimal INSERT pattern (required fields only):")
            examples.append(f"INSERT INTO {table_name} (" + ", ".join(req_codes) + f") VALUES ({placeholders});")
        return examples


def generate_table_ingestion_chunks(dico_data: Dict[str, Any]) -> List[Document]:  
    """Generate per-table ingestion profile chunks from raw dico_data only."""
    builder = TableIngestionChunkBuilder(dico_data)
    return builder.generate()

def generate_all_ingestion_chunks(dico_data: Dict[str, Any]) -> List[Document]:
    """Generate both table ingestion profiles and relationship path chunks."""
    builder = TableIngestionChunkBuilder(dico_data)
    table_docs = builder.generate()
    rel_docs = builder.generate_relationship_path_chunks()
    return table_docs + rel_docs

def export_chunks_to_json(docs: List[Document]) -> List[Dict[str, Any]]:
    """Convert Document chunks to JSON-serializable dicts for UI rendering."""
    out = []
    for d in docs:
        out.append({
            "content": d.page_content,
            "metadata": d.metadata
        })
    return out

def save_chunks_json(docs: List[Document], path: str):
    data = export_chunks_to_json(docs)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
