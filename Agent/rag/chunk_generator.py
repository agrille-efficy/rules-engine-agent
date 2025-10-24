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

        tables_raw = dico_data['data']['tables']
        fields_raw = dico_data['data']['fields']

        tables_df = pd.DataFrame.from_dict(tables_raw.values(), orient='columns')
        fields_df = pd.DataFrame.from_dict(fields_raw.values(), orient='columns')

        # Fix: Assign to self first, then filter
        self.tables_df = tables_df
        self.fields_df = fields_df
        
        # Now filter based on the assigned dataframes
        if 'stblKTable' in self.tables_df.columns:
            self.tables_df = self.tables_df[self.tables_df['stblKTable'].notna()]
        
        if 'sfldKTable' in self.fields_df.columns:
            self.fields_df = self.fields_df[self.fields_df['sfldKTable'].notna()]
        
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
        self.relationship_graph = self._build_relationship_graph()

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
            field_list = [
                {"name": str(f.get('sfldName')), "type": str(f.get('sfldType'))}
                for _, f in tbl_fields.iterrows()
            ]
            metadata = {
                "chunk_type": "table_ingestion_profile",
                "primary_table": table.get('stblName'),
                "table_code": table.get('stblCode'),
                "table_kind": ('Relation' if table.get('stblKind') == 'R' else 'Entity'),
                "field_count": len(tbl_fields),
                "fields": field_list,
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

        parts.append("# MAPPING ORDER GUIDANCE")
        if kind == 'Entity' and foreign_keys:
            parts.append("Map parent entities first, then this entity (due to FK dependencies).")
        elif kind == 'Entity':
            parts.append("This entity can be mapped independently (no FK dependencies detected).")
        else:
            parts.append("Map all referenced parent entity fields before relation fields.")
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
            if code in ('Key', 'KEY') or (code and code.lower().endswith('id')) or (code and code.startswith('K')):
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
            relatives = []
            for _, arr in beans.items():
                for bean in arr:
                    target = bean.get('detailEntity')
                    if target and target != name:
                        relatives.append(target)
            graph[name] = sorted(set(relatives))
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
            examples.append("Example CSV headers that could map: " + ",".join(header_codes))
        # Heuristic source mapping lines
        examples.append("User source column 'Full Name' -> map to field containing keyword: name")
        examples.append("User source column 'EmailAddress' -> map to field containing: email")
        examples.append("User source column 'Phone' -> map to field containing: phone")
        examples.append("If ambiguity (e.g., multiple phone fields), flag for human review instead of guessing.")
        return examples

    def _match_table_by_code_fragment(self, fragment):
        """Match a table name by code fragment for foreign key target inference"""
        if not fragment:
            return None
            
        # First, try exact matches with table codes
        for table in self.cleaned_tables:
            table_code = str(table.get('stblCode', '')).lower()
            if table_code and fragment == table_code:
                return table.get('stblName')
        
        # Then try partial matches with table codes
        for table in self.cleaned_tables:
            table_code = str(table.get('stblCode', '')).lower()
            if table_code and fragment in table_code:
                return table.get('stblName')
        
        # Try matches with table names
        for table in self.cleaned_tables:
            table_name = str(table.get('stblName', '')).lower()
            if table_name and fragment in table_name:
                return table.get('stblName')
        
        # Try reverse match - if table name contains the fragment
        for table in self.cleaned_tables:
            table_name = str(table.get('stblName', '')).lower()
            if table_name and table_name in fragment:
                return table.get('stblName')
        
        return None

    
    def generate_multi_level_chunks(self) -> List[Document]:
        """Generate multiple chunk types with varying granularity for better semantic matching"""
        all_chunks = []
        
        for table in self.cleaned_tables:
            ktable = table['stblKTable']
            if pd.isna(ktable):
                continue
            try:
                tbl_fields = self.fields_index.get_group(ktable)
            except KeyError:
                continue
                
            # Generate different chunk types for this table
            table_chunks = self._generate_table_multi_chunks(table, tbl_fields)
            all_chunks.extend(table_chunks)
        
        return all_chunks

    def _generate_table_multi_chunks(self, table: Dict[str, Any], tbl_fields: pd.DataFrame) -> List[Document]:
        """Generate multiple chunk types for a single table"""
        chunks = []
        table_name = table.get('stblName')
        table_code = table.get('stblCode')
        field_names = [str(f.get('sfldName')) for _, f in tbl_fields.iterrows()]
        # 1. High-level table summary chunk
        summary_chunk = self._build_table_summary_chunk(table, tbl_fields)
        field_list = [
            {"name": str(f.get('sfldName')), "type": str(f.get('sfldType'))}
            for _, f in tbl_fields.iterrows()
        ]
        chunks.append(Document(
            page_content=summary_chunk,
            metadata={
                "chunk_type": "table_summary",
                "primary_table": table_name,
                "table_code": table_code,
                "table_kind": ('Relation' if table.get('stblKind') == 'R' else 'Entity'),
                "field_count": len(tbl_fields),
                "semantic_focus": "table_purpose_domain",
                "fields": field_list,
            }
        ))
        # 2. Field-focused chunks
        field_chunks = self._build_field_semantic_chunks(table, tbl_fields)
        chunks.extend(field_chunks)
        # 3. Relationship-focused chunk
        if self._has_relationships(table, tbl_fields):
            rel_chunk = self._build_relationship_chunk(table, tbl_fields)
            if rel_chunk:
                chunks.append(rel_chunk)
        # 4. Keep original comprehensive chunk for complete context
        original_chunk = self._build_table_chunk(table, tbl_fields)
        field_list = [
            {"name": str(f.get('sfldName')), "type": str(f.get('sfldType'))}
            for _, f in tbl_fields.iterrows()
        ]
        chunks.append(Document(
            page_content=original_chunk,
            metadata={
                "chunk_type": "table_ingestion_profile_complete",
                "primary_table": table_name,
                "table_code": table_code,
                "table_kind": ('Relation' if table.get('stblKind') == 'R' else 'Entity'),
                "field_count": len(tbl_fields),
                "semantic_focus": "complete_schema",
                "fields": field_list,
            }
        ))
        return chunks

    def _build_table_summary_chunk(self, table: Dict[str, Any], tbl_fields: pd.DataFrame) -> str:
        """Build a concise table summary chunk focused on purpose and domain"""
        name = table.get('stblName')
        code = table.get('stblCode')
        kind_flag = table.get('stblKind')
        kind = 'Relation' if kind_flag == 'R' else 'Entity'
        
        purpose_line = self._infer_purpose(table, tbl_fields)
        required, optional, foreign_keys, _ = self._classify_fields(tbl_fields)
        
        # Extract key semantic indicators
        key_fields = []
        for _, row in tbl_fields.iterrows():
            field_name = str(row.get('sfldName', '')).lower()
            if any(key in field_name for key in ['name', 'title', 'email', 'phone', 'date', 'status']):
                key_fields.append(row.get('sfldName'))
        
        parts = []
        parts.append(f"=== {name} ({code}) - {kind.upper()} TABLE ===")
        parts.append(f"Business Purpose: {purpose_line}")
        parts.append(f"Table Type: {kind}")
        parts.append(f"Total Fields: {len(tbl_fields)}")
        parts.append("")
        
        if key_fields:
            parts.append(f"Key Business Fields: {', '.join(key_fields[:6])}")
        
        if required:
            parts.append(f"Required Fields: {len(required)} fields must be provided")
        
        if foreign_keys:
            parent_tables = [fk[2] for fk in foreign_keys if fk[2]]
            if parent_tables:
                parts.append(f"Dependencies: Requires data from {', '.join(set(parent_tables))}")
        
        parts.append("")
        parts.append("# INGESTION SUITABILITY")
        
        # Domain-specific hints
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in ['contact', 'person', 'individual']):
            parts.append("- Ideal for: Contact/Person data with names, emails, phone numbers")
        elif any(keyword in name_lower for keyword in ['company', 'organization', 'enterprise']):
            parts.append("- Ideal for: Company/Organization data with business information")
        elif any(keyword in name_lower for keyword in ['mail', 'email', 'message']):
            parts.append("- Ideal for: Email/Message data with sender, recipient, subject, content")
        elif any(keyword in name_lower for keyword in ['activity', 'visit', 'event']):
            parts.append("- Ideal for: Activity/Event tracking with dates, types, participants")
        elif any(keyword in name_lower for keyword in ['opportunity', 'deal', 'sales']):
            parts.append("- Ideal for: Sales opportunity data with stages, amounts, probabilities")
        else:
            parts.append("- General purpose table for structured business data")
        
        return "\n".join(parts)

    def _build_field_semantic_chunks(self, table: Dict[str, Any], tbl_fields: pd.DataFrame) -> List[Document]:
        """Create semantic field groups for better matching"""
        chunks = []
        table_name = table.get('stblName')
        table_code = table.get('stblCode')
        
        # Group fields by semantic categories
        field_groups = self._group_fields_semantically(tbl_fields)
        
        for group_name, fields in field_groups.items():
            if not fields:  # Skip empty groups
                continue
                
            chunk_content = self._build_field_group_content(table_name, group_name, fields, table)
            group_field_list = [
                {"name": str(f.get('sfldName')), "type": str(f.get('sfldType'))}
                for f in fields
            ]
            chunks.append(Document(
                page_content=chunk_content,
                metadata={
                    "chunk_type": "field_group",
                    "primary_table": table_name,
                    "table_code": table_code,
                    "field_group": group_name,
                    "field_count": len(fields),
                    "semantic_focus": f"fields_{group_name}",
                    "fields": group_field_list,
                }
            ))
        
        return chunks

    def _group_fields_semantically(self, tbl_fields: pd.DataFrame) -> Dict[str, List]:
        """Group fields by semantic meaning"""
        groups = {
            "identification": [],
            "contact_info": [], 
            "dates_times": [],
            "business_data": [],
            "relationships": [],
            "metadata": []
        }
        
        for _, row in tbl_fields.iterrows():
            field_name = str(row.get('sfldName', '')).lower()
            field_code = str(row.get('sfldCode', '')).lower()
            
            # Categorize based on field patterns
            if any(pattern in field_name for pattern in ['id', 'key', 'number', 'code']) or field_code in ['key', 'KEY']:
                groups["identification"].append(row)
            elif any(pattern in field_name for pattern in ['email', 'mail', 'phone', 'mobile', 'tel', 'telephone', 'address', 'contact']):
                groups["contact_info"].append(row)
            elif any(pattern in field_name for pattern in ['date', 'time', 'created', 'updated', 'modified', 'timestamp']):
                groups["dates_times"].append(row)
            elif field_code.startswith('K') and field_code.lower() != 'key':
                groups["relationships"].append(row)
            elif any(pattern in field_name for pattern in ['status', 'flag', 'active', 'deleted', 'state', 'type']):
                groups["metadata"].append(row)
            else:
                groups["business_data"].append(row)
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _build_field_group_content(self, table_name: str, group_name: str, fields: List, table: Dict[str, Any]) -> str:
        """Build field group content with context overlap"""
        parts = []
        parts.append(f"=== {table_name} - {group_name.upper()} FIELDS ===")
        parts.append(f"Table: {table_name}")
        parts.append(f"Field Category: {group_name}")
        parts.append("")
        
        # Add context about the table's purpose (overlap from summary)
        purpose_context = self._get_table_purpose_context(table_name, table)
        parts.append(f"Table Purpose: {purpose_context}")
        parts.append("")
        
        parts.append("# FIELD DEFINITIONS")
        for field in fields:
            field_name = field.get('sfldName')
            field_code = field.get('sfldCode')
            field_type = field.get('sfldType')
            size_val = field.get('sfldSize')
            size = f"({int(size_val)})" if isinstance(size_val, (int,float)) and size_val else ""
            nullable = 'NULLABLE' if field.get('sfldNullable') else 'REQUIRED'
            
            parts.append(f"- {field_name} ({field_code}): {field_type}{size} | {nullable}")
        
        parts.append("")
        
        # Add group-specific guidance
        if group_name == "relationships":
            parts.append("# RELATIONSHIP GUIDANCE")
            parts.append("These fields reference other tables - ensure parent records exist first")
            parts.append("Foreign key constraints must be satisfied during data ingestion")
        elif group_name == "contact_info":
            parts.append("# CONTACT DATA GUIDANCE")
            parts.append("Validate email formats, phone number patterns, and address structures")
            parts.append("Consider data privacy and GDPR compliance for personal information")
        elif group_name == "dates_times":
            parts.append("# DATE/TIME GUIDANCE")
            parts.append("Use ISO format (YYYY-MM-DD HH:MM:SS) for timestamps")
            parts.append("Handle timezone conversions appropriately")
        elif group_name == "identification":
            parts.append("# IDENTIFICATION GUIDANCE")
            parts.append("Primary keys may be auto-generated - check if manual assignment is needed")
            parts.append("Ensure uniqueness constraints are respected")
        
        return "\n".join(parts)

    def _get_table_purpose_context(self, table_name: str, table: Dict[str, Any] = None) -> str:
        """Get table purpose context for overlap in chunks"""
        if table and self.enable_llm_summary:
            # Try to get from existing table if available
            try:
                ktable = table['stblKTable']
                tbl_fields = self.fields_index.get_group(ktable)
                return self._infer_purpose(table, tbl_fields)
            except:
                pass
        
        # Fallback to heuristic-based purpose inference
        name_lower = table_name.lower()
        if any(keyword in name_lower for keyword in ['contact', 'person']):
            return "manages contact and person information"
        elif any(keyword in name_lower for keyword in ['company', 'organization']):
            return "stores company and organization data"
        elif any(keyword in name_lower for keyword in ['mail', 'email']):
            return "handles email and message communication"
        elif any(keyword in name_lower for keyword in ['activity', 'visit']):
            return "tracks activities and interactions"
        elif any(keyword in name_lower for keyword in ['opportunity', 'deal']):
            return "manages sales opportunities and deals"
        else:
            return "general business data management"

    def _has_relationships(self, table: Dict[str, Any], tbl_fields: pd.DataFrame) -> bool:
        """Check if table has meaningful relationships to warrant a relationship chunk"""
        table_name = table.get('stblName')
        
        # Check if table has foreign keys
        _, _, foreign_keys, _ = self._classify_fields(tbl_fields)
        if foreign_keys:
            return True
            
        # Check if table appears in relationship graph
        if table_name in self.relationship_graph and self.relationship_graph[table_name]:
            return True
            
        return False

    def _build_relationship_chunk(self, table: Dict[str, Any], tbl_fields: pd.DataFrame) -> Document:
        """Build a relationship-focused chunk for tables with significant relationships"""
        table_name = table.get('stblName')
        table_code = table.get('stblCode')
        _, _, foreign_keys, _ = self._classify_fields(tbl_fields)
        related_tables = self.relationship_graph.get(table_name, [])
        field_list = [
            {"name": str(f.get('sfldName')), "type": str(f.get('sfldType'))}
            for _, f in tbl_fields.iterrows()
        ]
        parts = []
        parts.append(f"=== {table_name} - RELATIONSHIPS & DEPENDENCIES ===")
        parts.append(f"Table: {table_name} ({table_code})")
        parts.append("")
        
        parts.append("# FOREIGN KEY DEPENDENCIES")
        if foreign_keys:
            for fk_name, fk_code, fk_target in foreign_keys:
                target_info = f" -> {fk_target}" if fk_target else " -> [UNKNOWN TABLE]"
                parts.append(f"- {fk_name} ({fk_code}){target_info}")
            parts.append("")
            parts.append("INSERTION ORDER: Parent tables must be populated before this table")
        else:
            parts.append("- No foreign key dependencies detected")
        
        parts.append("")
        parts.append("# CHILD RELATIONSHIPS")
        if related_tables:
            parts.append("This table is referenced by:")
            for child_table in related_tables:
                parts.append(f"- {child_table}")
            parts.append("")
            parts.append("DELETION IMPACT: Removing records may affect child table data")
        else:
            parts.append("- No child table dependencies detected")
        
        parts.append("")
        parts.append("# DATA INGESTION WORKFLOW")
        if foreign_keys and related_tables:
            parts.append("1. Ensure all parent tables have required data")
            parts.append("2. Insert records into this table")
            parts.append("3. Child tables can then reference these records")
        elif foreign_keys:
            parts.append("1. Ensure all parent tables have required data")
            parts.append("2. Insert records into this table")
        elif related_tables:
            parts.append("1. This table can be populated independently")
            parts.append("2. Child tables will then reference these records")
        else:
            parts.append("1. This table can be populated independently")
        
        return Document(
            page_content="\n".join(parts),
            metadata={
                "chunk_type": "relationship_focus",
                "primary_table": table_name,
                "table_code": table_code,
                "related_tables": related_tables,
                "foreign_key_count": len(foreign_keys),
                "semantic_focus": "relationships_dependencies",
                "fields": field_list,
            }
        )

def generate_table_ingestion_chunks(dico_data: Dict[str, Any]) -> List[Document]:  
    """Generate per-table ingestion profile chunks from raw dico_data only."""
    builder = TableIngestionChunkBuilder(dico_data)
    return builder.generate()

def generate_multi_level_ingestion_chunks(dico_data: Dict[str, Any]) -> List[Document]:
    """Generate multi-level chunks with improved semantic matching."""
    builder = TableIngestionChunkBuilder(dico_data)
    return builder.generate_multi_level_chunks()

def generate_all_ingestion_chunks(dico_data: Dict[str, Any]) -> List[Document]:
    """Generate both table ingestion profiles and relationship path chunks."""
    builder = TableIngestionChunkBuilder(dico_data)
    table_docs = builder.generate()
    rel_docs = builder.generate_relationship_path_chunks()
    return table_docs + rel_docs

def generate_all_multi_level_chunks(dico_data: Dict[str, Any]) -> List[Document]:
    """Generate multi-level chunks plus relationship path chunks."""
    builder = TableIngestionChunkBuilder(dico_data)
    table_docs = builder.generate_multi_level_chunks()
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
