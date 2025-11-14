"""
Mapper Service - Unified multi-table mapping using exact, fuzzy, semantic, and LLM batch calls.
Replaces llm_field_matcher, field_mapper, and multi_table_mapper.
Includes: semantic grouping, relationship detection, refinement, and judge.
"""
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from fuzzywuzzy import fuzz
from dataclasses import dataclass

from .clients.openai_client import ResilientOpenAIClient
from ..config import get_settings
from ..models.file_analysis_model import FileAnalysisResult, ColumnMetadata
from ..models.rag_match_model import (
    FieldMapping, 
    MappingValidationResult,
    TableFieldMapping,
    MultiTableMappingResult
)

settings = get_settings()

@dataclass
class EnhancedColumnProfile:
    """Enhanced column profile with semantic analysis for LLM mapping."""
    name: str
    english_name: str
    data_type: str
    semantic_category: str
    content_pattern: str
    business_meaning: str
    sample_analysis: Dict[str, Any]
    normalized_tokens: List[str]
    confidence_indicators: List[str]

class Mapper:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self.client = ResilientOpenAIClient(
            api_key=settings.openai_api_key,
            model=model,
            temperature=temperature,
            max_retries=3
        )
        
        # Initialize semantic embeddings model (lazy loading)
        self._embedding_model = None
        self._field_embeddings_cache = {}
        
        # Matching thresholds
        self.exact_match_threshold = 1.0
        self.fuzzy_match_threshold = 0.69
        self.semantic_match_threshold = 0.6
        self.embedding_match_threshold = 0.7  # New: for embedding-based matching
        self.llm_match_threshold = 0.5
        
        # Generic field detection
        self.generic_field_terms = [
            'metadata', 'memo', 'data', 'info', 'information',
            'ai', 'scoring', 'generic', 'misc', 'miscellaneous',
            'other', 'extra', 'additional', 'temp', 'temporary',
            'custom', 'field', 'column', 'value', 'blob', 'archived'
        ]
        
        # Penalties
        self.generic_field_penalty = 0.5
        self.type_mismatch_penalty = 0.6
        
        # Refinement settings
        self.max_same_field_mappings = 2
        self.refinement_confidence_threshold = 0.72
        
        # Multi-table settings
        self.min_confidence_threshold = 0.5
        self.min_columns_per_table = 1

    def map_to_multiple_tables(
        self,
        file_analysis: FileAnalysisResult,
        candidate_tables: List[Dict],
        primary_table: str = None,
        max_tables: int = 10
    ) -> MultiTableMappingResult:
        """
        Map CSV columns to multiple database tables with intelligent assignment.
        
        Args:
            file_analysis: Result from file analysis
            candidate_tables: List of candidate tables from RAG
            primary_table: Optional primary table to prioritize
            max_tables: Maximum number of tables to consider
            
        Returns:
            MultiTableMappingResult with mappings per table
        """
        logging.info(f"Starting multi-table mapping for {file_analysis.structure.file_name}")
        logging.info(f"Evaluating {len(candidate_tables)} candidate tables (max: {max_tables})")
        if primary_table:
            logging.info(f"Primary table: {primary_table}")
        
        # Step 1: Prepare table schemas
        prioritized_tables = candidate_tables[:max_tables]
        table_schemas = self._prepare_table_schemas(prioritized_tables)
        
        # Step 2: Semantic grouping of columns
        column_groups = self._group_columns_semantically(file_analysis)
        
        # Step 3: Map columns to ALL tables (multi-strategy)
        column_to_table_mappings = self._map_columns_to_all_tables(
            file_analysis,
            table_schemas
        )
        
        # Step 4: Smart multi-table assignment
        table_assignments = self._smart_multi_table_assignment(
            column_to_table_mappings,
            column_groups,
            table_schemas,
            candidate_tables,
            primary_table
        )
        
        # Step 5: Judge - analyze if refinement is needed
        needs_refinement = self._judge_needs_refinement(table_assignments, file_analysis)
        
        # Step 6: Refine if needed
        if needs_refinement:
            logging.warning("Judge recommends refinement")
            table_assignments = self._refine_all_table_mappings(table_assignments, table_schemas)
        else:
            logging.info("Judge approved mappings - no refinement needed")
        
        # Step 7: Filter and validate
        valid_table_mappings = self._filter_and_validate_table_mappings(
            table_assignments,
            file_analysis,
            candidate_tables
        )
        
        # Step 8: Determine insertion order
        valid_table_mappings = self._determine_insertion_order(valid_table_mappings, table_schemas)
        
        # Step 9: Create final result
        result = self._create_final_result(file_analysis, valid_table_mappings, needs_refinement)
        
        # Step 10: Log summary
        self._log_summary(result)
        
        return result
    
    def _group_columns_semantically(self, file_analysis: FileAnalysisResult) -> Dict[str, List[str]]:
        """
        Group CSV columns by semantic domain.
        Detects: entity core, foreign keys, user/company/contact references, metadata, financial data.
        """
        groups = {
            'entity_core': [],
            'entity_metadata': [],
            'entity_financial': [],
            'foreign_keys': [],
            'user_references': [],
            'company_references': [],
            'contact_references': [],
            'relationship_data': [],
            'other': []
        }
        
        for column in file_analysis.columns:
            col_name = column.name
            col_lower = (column.english_name or column.name).lower()
            
            # Priority 1: Foreign keys
            if (col_lower.endswith('uniqueid') or col_lower.endswith('_id') or 
                (col_lower != 'id' and '_id' in col_lower)):
                groups['foreign_keys'].append(col_name)
            
            # Priority 2: User references
            elif any(p in col_lower for p in ['responsible', 'creator', 'owner', 'personne', 'charge', 'assignee']):
                groups['user_references'].append(col_name)
            
            # Priority 3: Company references
            elif any(p in col_lower for p in ['company', 'comp', 'compte', 'société', 'organization']):
                groups['company_references'].append(col_name)
            
            # Priority 4: Contact references
            elif any(p in col_lower for p in ['contact', 'cont']) and 'id' not in col_lower:
                groups['contact_references'].append(col_name)
            
            # Priority 5: Relationship data
            elif any(p in col_lower for p in ['intr_', 'vat', 'intervention', 'interest']):
                groups['relationship_data'].append(col_name)
            
            # Priority 6: Core entity fields
            elif any(p in col_lower for p in ['name', 'status', 'type', 'nature', 'reason', 'nom', 'statut']):
                groups['entity_core'].append(col_name)
            
            # Priority 7: Metadata
            elif any(p in col_lower for p in ['date', 'time', 'created', 'updated', 'sys', 'archived', 'banner']):
                groups['entity_metadata'].append(col_name)
            
            # Priority 8: Financial
            elif any(p in col_lower for p in ['amount', 'price', 'cost', 'revenue', 'budget', 'montant', 'loyer', 'capex', 'vente']):
                groups['entity_financial'].append(col_name)
            
            else:
                groups['other'].append(col_name)
        
        filtered_groups = {k: v for k, v in groups.items() if v}
        logging.info(f"Semantic grouping: {', '.join([f'{k}({len(v)})' for k, v in filtered_groups.items()])}")
        return filtered_groups
    
    def _map_columns_to_all_tables(
        self,
        file_analysis: FileAnalysisResult,
        table_schemas: Dict
    ) -> Dict[str, List[Dict]]:
        """
        Map each column to all tables using multiple strategies.
        Returns: {column_name: [{table, mapping, confidence, reasoning}, ...]}
        """
        columns_mappings = {}
        source_columns = file_analysis.columns
        
        for column in source_columns:
            col_name = column.name
            columns_mappings[col_name] = []
            
            # Try exact + fuzzy + semantic for each table
            for table_name, schema_info in table_schemas.items():
                target_fields = schema_info['fields']
                
                # Strategy: Exact, Fuzzy, Semantic
                best_match = self._find_best_non_llm_match(column, target_fields)
                
                if best_match:
                    columns_mappings[col_name].append({
                        'table': table_name,
                        'table_type': schema_info['table_type'],
                        'mapping': FieldMapping(
                            source_column=column.name,
                            source_column_english=column.english_name or column.name,
                            target_column=best_match['target_field'],
                            confidence_score=best_match['confidence'],
                            match_type=best_match['match_type'],
                            data_type_compatible=True,
                            source_data_type=column.data_type,
                            sample_source_values=column.sample_values[:3],
                            score_detail=best_match['score_detail']
                        ),
                        'confidence': best_match['confidence'],
                        'reasoning': best_match.get('reasoning', '')
                    })
        
        # Strategy 4: Enhanced Batch LLM for unmapped or low-confidence columns
        unmapped_columns = []
        for col_name, mappings in columns_mappings.items():
            if not mappings or max([m['confidence'] for m in mappings]) < 0.7:
                col = next(c for c in source_columns if c.name == col_name)
                unmapped_columns.append(col)
        
        if unmapped_columns:
            logging.info(f"Using Enhanced LLM for {len(unmapped_columns)} columns with low/no matches")
            llm_results = self._batch_llm_mapping(unmapped_columns, table_schemas, file_analysis)
            
            for col_name, llm_matches in llm_results.items():
                for match in llm_matches:
                    # Add or replace existing mapping if LLM confidence is higher
                    existing = columns_mappings.get(col_name, [])
                    existing.append({
                        'table': match['table'],
                        'table_type': match['table_type'],
                        'mapping': match['mapping'],
                        'confidence': match['confidence'],
                        'reasoning': match['reasoning']
                    })
                    columns_mappings[col_name] = existing
        
        # Sort by confidence
        for col_name in columns_mappings:
            columns_mappings[col_name].sort(key=lambda x: x['confidence'], reverse=True)
        
        return columns_mappings
    
    def _batch_llm_mapping(
        self,
        columns: List[ColumnMetadata],
        table_schemas: Dict,
        file_analysis: FileAnalysisResult
    ) -> Dict[str, List[Dict]]:
        """Enhanced batch LLM mapping using semantic column profiles for better understanding."""
        results = {}
        
        # Create enhanced profiles for all columns
        enhanced_profiles = self._create_enhanced_column_profiles(columns)
        
        for table_name, schema_info in table_schemas.items():
            target_fields = schema_info['fields']
            table_type = schema_info['table_type']
            
            # Build enhanced prompt with semantic analysis
            batch_prompt = self._build_enhanced_batch_prompt(
                enhanced_profiles,
                target_fields,
                table_name,
                schema_info,
                file_analysis
            )
            
            llm_response = self._batch_llm_call(batch_prompt)
            
            for match in llm_response.get('matches', []):
                source_col = match.get('source_column')
                target_field = match.get('target_field')
                confidence = match.get('confidence', 0.0)
                reasoning = match.get('reasoning', '')
                
                if target_field and target_field != 'NO_MATCH' and confidence >= self.llm_match_threshold:
                    col = next((c for c in columns if c.name == source_col), None)
                    if col:
                        if source_col not in results:
                            results[source_col] = []
                        
                        results[source_col].append({
                            'table': table_name,
                            'table_type': table_type,
                            'mapping': FieldMapping(
                                source_column=col.name,
                                source_column_english=col.english_name or col.name,
                                target_column=target_field,
                                confidence_score=confidence,
                                match_type='llm_enhanced',
                                data_type_compatible=True,
                                source_data_type=col.data_type,
                                sample_source_values=col.sample_values[:3]
                            ),
                            'confidence': confidence,
                            'reasoning': reasoning
                        })
        
        return results

    def _build_enhanced_batch_prompt(
        self,
        enhanced_profiles: List[EnhancedColumnProfile],
        target_fields: List[str],
        table_name: str,
        schema_info: Dict,
        file_analysis: FileAnalysisResult
    ) -> str:
        """
        Build enhanced prompt with semantic analysis and pattern recognition.
        This provides much richer context for LLM decision making.
        """
        # Build detailed column descriptions with semantic analysis
        column_descriptions = []
        for profile in enhanced_profiles:
            # Format sample analysis
            patterns = ", ".join(profile.sample_analysis.get('patterns_detected', []))
            patterns_text = f"Patterns: {patterns}" if patterns else "Patterns: None detected"
            
            # Format confidence indicators
            confidence_text = " | ".join(profile.confidence_indicators[:3])  # Top 3 indicators
            
            # Create rich description
            col_desc = f"""
Column: "{profile.name}" (English: "{profile.english_name}")
- Data Type: {profile.data_type}
- Semantic Category: {profile.semantic_category}
- Business Meaning: {profile.business_meaning}
- Content Pattern: {profile.content_pattern}
- {patterns_text}
- Semantic Tokens: {', '.join(profile.normalized_tokens)}
- Confidence Indicators: {confidence_text}
- Sample Values: {', '.join([f'"{v}"' for v in profile.sample_analysis.get('clean_sample_count', 0) and profile.name in [c.name for c in file_analysis.columns] and next(c for c in file_analysis.columns if c.name == profile.name).sample_values[:3] or []])}
""".strip()
            
            column_descriptions.append(col_desc)
        
        # Build table context
        table_context = f"""
Table: {table_name}
Type: {schema_info.get('table_type', 'Unknown')}
Available Fields: {', '.join(target_fields)}
RAG Score: {schema_info.get('rag_score', 0.0):.2f}
"""
        
        # Build the enhanced prompt
        prompt = f"""
You are analyzing {len(enhanced_profiles)} CSV columns for mapping to database table '{table_name}'.

ENHANCED COLUMN ANALYSIS:
{'=' * 50}
{chr(10).join(column_descriptions)}

TARGET TABLE CONTEXT:
{'=' * 50}
{table_context}

MAPPING INSTRUCTIONS:
1. Use the semantic analysis to understand the true meaning of each column
2. Consider the detected patterns (phone, email, VAT, etc.) for precise matching
3. Match semantic categories to appropriate database field types
4. Prioritize business meaning over simple name similarity  
5. Use confidence indicators to assess mapping quality
6. For foreign keys, look for ID patterns and reference semantics
7. For contact info, match based on detected patterns (phone/email/address)
8. For financial data, consider currency patterns and amount semantics
9. Avoid mapping structured data to generic memo/blob fields
10. Be conservative - return NO_MATCH if semantic alignment is poor

Return JSON with detailed reasoning based on semantic analysis:
{{
  "matches": [
    {{
      "source_column": "column_name",
      "target_field": "field_name" or "NO_MATCH",
      "confidence": 0.0-1.0,
      "reasoning": "Detailed explanation based on semantic category, patterns, and business meaning"
    }}
  ]
}}
"""
        
        return prompt

    def _get_enhanced_system_prompt(self) -> str:
        """Enhanced system prompt for better semantic understanding."""
        return """
You are an expert data mapping specialist with deep understanding of:
- Database schema design and field semantics
- Business data patterns and structures  
- ETL best practices and data quality
- Cross-language field naming conventions
- Pattern recognition in sample data

Your expertise includes:
1. SEMANTIC ANALYSIS: Understanding the business meaning behind field names
2. PATTERN RECOGNITION: Identifying data formats (phone, email, VAT, dates, etc.)
3. BUSINESS CONTEXT: Mapping fields based on their functional purpose
4. DATA QUALITY: Ensuring type compatibility and structural alignment
5. RELATIONSHIP MODELING: Understanding foreign keys and entity relationships

Key Principles:
- Semantic meaning trumps lexical similarity
- Pattern recognition validates field purpose
- Business context guides mapping decisions
- Confidence should reflect semantic alignment quality
- Conservative approach: NO_MATCH is better than wrong mapping

Focus on the enhanced column analysis provided, which includes:
- Semantic categories (foreign_key, contact_phone, person_name, etc.)
- Detected patterns from sample data analysis
- Business meaning extraction from field names
- Content pattern identification
- Confidence indicators for mapping quality

Use this rich context to make informed, high-quality mapping decisions.
"""

    def _batch_llm_call(self, prompt: str) -> Dict[str, Any]:
        """Call LLM using resilient client with enhanced system prompt."""
        try:
            response_text = self.client.generate_completion(
                messages=[
                    {"role": "system", "content": self._get_enhanced_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response_text)
            return result if isinstance(result, dict) and "matches" in result else {"matches": []}
        except Exception as e:
            logging.error(f"Enhanced batch LLM call failed: {e}")
            return {"matches": []}

    def _build_batch_prompt_legacy(self, columns, target_fields, table_name, table_metadata):
        """Legacy prompt builder - kept as fallback."""
        col_descriptions = []
        for col in columns:
            samples = getattr(col, 'sample_values', [])
            sample_text = ', '.join([f'"{v}"' for v in samples[:5]]) if samples else 'No samples'
            col_descriptions.append(
                f"- Name: {col.name}, English: {getattr(col, 'english_name', col.name)}, Type: {getattr(col, 'data_type', 'string')}, Samples: {sample_text}"
            )
        prompt = f"Match these source columns to the best target fields in table '{table_name}':\n"
        prompt += '\n'.join(col_descriptions)
        prompt += f"\nTarget fields: {', '.join(target_fields)}\nTable metadata: {table_metadata}\n"
        prompt += "Return JSON: {matches: [{source_column, target_field, confidence, reasoning}]}"
        return prompt

    def _get_system_prompt_legacy(self) -> str:
        """Legacy system prompt - kept as fallback."""
        return (
            "You are an expert data engineer specializing in database schema mapping and ETL processes.\n"
            "Your task is to match CSV column names to database table fields by analyzing:\n"
            "- Column names and their semantic meaning\n"
            "- Sample data values and patterns\n"
            "- Data types and formats\n"
            "- Business domain context\n"
            "- Common database naming conventions\n"
            "Guidelines:\n"
            "1. Consider the SEMANTIC MEANING, not just string similarity\n"
            "2. Avoid mapping structured data (dates, numbers) to generic memo/blob fields\n"
            "3. Penalize matches to overly generic fields (metadata, memo, misc, data, blob, etc.)\n"
            "4. Consider data type compatibility\n"
            "5. Use sample values to validate the match makes sense\n"
            "6. Be conservative - if uncertain, return NO_MATCH rather than guessing\n"
            "7. Provide clear reasoning for your decision\n"
            "8. Avoid at all costs mapping to or from fields with capital F in their name.\n"
            "Return JSON: {matches: [{source_column, target_field, confidence, reasoning}]}"
        )

    def _create_enhanced_column_profiles(
        self,
        columns: List[ColumnMetadata]
    ) -> List[EnhancedColumnProfile]:
        """
        Create enhanced column profiles with semantic analysis and pattern recognition.
        This provides much richer context for LLM mapping decisions.
        """
        enhanced_profiles = []
        
        for col in columns:
            # Extract semantic tokens from column name
            normalized_tokens = self._extract_semantic_tokens(col.name, col.english_name)
            
            # Analyze sample values for patterns
            sample_analysis = self._analyze_column_samples(col.sample_values, col.data_type)
            
            # Determine semantic category
            semantic_category = self._determine_semantic_category(col, sample_analysis)
            
            # Extract business meaning from name patterns
            business_meaning = self._extract_business_meaning(col.name, col.english_name, semantic_category)
            
            # Identify content pattern
            content_pattern = self._identify_content_pattern(col, sample_analysis)
            
            # Generate confidence indicators
            confidence_indicators = self._generate_confidence_indicators(col, sample_analysis, semantic_category)
            
            profile = EnhancedColumnProfile(
                name=col.name,
                english_name=col.english_name or col.name,
                data_type=col.data_type,
                semantic_category=semantic_category,
                content_pattern=content_pattern,
                business_meaning=business_meaning,
                sample_analysis=sample_analysis,
                normalized_tokens=normalized_tokens,
                confidence_indicators=confidence_indicators
            )
            
            enhanced_profiles.append(profile)
            
        return enhanced_profiles
    
    def _extract_semantic_tokens(self, name: str, english_name: str = None) -> List[str]:
        """
        Extract meaningful semantic tokens from column names.
        Better than simple word splitting - understands business context.
        """
        # Use english name if available, otherwise original name
        text = english_name or name
        
        # Split into words using existing method
        words = self._split_words(text)
        
        # Normalize and filter
        tokens = []
        for word in words:
            word_lower = word.lower()
            
            # Skip very short words and common noise
            if len(word_lower) < 2 or word_lower in ['id', 'cd', 'no', 'nb', 'dt', 'tm']:
                continue
                
            # Expand common abbreviations
            expansions = {
                'soc': 'societe',
                'comp': 'company', 
                'cont': 'contact',
                'tel': 'telephone',
                'addr': 'address',
                'desc': 'description',
                'ref': 'reference',
                'num': 'number',
                'amt': 'amount',
                'qty': 'quantity',
                'pct': 'percent',
                'std': 'standard',
                'cp': 'custom',
                'rem': 'remarks'
            }
            
            expanded = expansions.get(word_lower, word_lower)
            tokens.append(expanded)
            
        return tokens
    
    def _analyze_column_samples(self, samples: List[str], data_type: str) -> Dict[str, Any]:
        """
        Analyze sample values to understand content patterns and structure.
        This is crucial for semantic understanding.
        """
        if not samples:
            return {'pattern_type': 'unknown', 'confidence': 0.0}
            
        analysis = {
            'sample_count': len(samples),
            'pattern_type': 'unknown',
            'confidence': 0.0,
            'patterns_detected': [],
            'value_characteristics': {},
            'business_indicators': []
        }
        
        # Clean samples (remove None, empty)
        clean_samples = [str(s).strip() for s in samples if s is not None and str(s).strip()]
        if not clean_samples:
            return analysis
            
        analysis['clean_sample_count'] = len(clean_samples)
        
        # Pattern detection
        patterns = []
        
        # Phone numbers
        phone_patterns = [
            r'^\+?[\d\s\-\(\)\.]{8,}$',  # International/formatted
            r'^\d{10}$',  # 10 digits
            r'^\d{3}[\-\.\s]\d{3}[\-\.\s]\d{4}$'  # US format
        ]
        
        if any(re.match(p, sample) for p in phone_patterns for sample in clean_samples[:3]):
            patterns.append('phone_number')
            
        # Email addresses
        if any(re.match(r'^[^@]+@[^@]+\.[^@]+$', sample) for sample in clean_samples[:3]):
            patterns.append('email')
            
        # URLs
        if any(re.match(r'^https?://', sample) for sample in clean_samples[:3]):
            patterns.append('url')
            
        # Postal codes
        postal_patterns = [
            r'^\d{5}(-\d{4})?$',  # US ZIP
            r'^[A-Z]\d[A-Z] \d[A-Z]\d$',  # Canadian
            r'^\d{4,5}$'  # European
        ]
        if any(re.match(p, sample) for p in postal_patterns for sample in clean_samples[:3]):
            patterns.append('postal_code')
            
        # VAT/Tax numbers
        if any(re.match(r'^[A-Z]{2}[\d\w]{8,}$', sample) for sample in clean_samples[:3]):
            patterns.append('vat_number')
            
        # Currency amounts
        if any(re.match(r'^[\$€£]?[\d,]+\.?\d*$', sample) for sample in clean_samples[:3]):
            patterns.append('currency')
            
        # Dates
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}',  # ISO date
            r'^\d{1,2}/\d{1,2}/\d{4}',  # US date
            r'^\d{1,2}\.\d{1,2}\.\d{4}'  # European date
        ]
        if any(re.match(p, sample) for p in date_patterns for sample in clean_samples[:3]):
            patterns.append('date')
            
        # Person names (heuristic)
        name_indicators = 0
        for sample in clean_samples[:5]:
            words = sample.split()
            if (len(words) >= 2 and 
                all(w[0].isupper() and w[1:].islower() for w in words if w) and
                len(sample) > 5):
                name_indicators += 1
                
        if name_indicators >= 2:
            patterns.append('person_name')
            
        # Company names (heuristic)
        company_indicators = ['ltd', 'inc', 'corp', 'llc', 'sa', 'sas', 'sarl', 'gmbh']
        if any(any(indicator in sample.lower() for indicator in company_indicators) 
               for sample in clean_samples[:3]):
            patterns.append('company_name')
            
        # Boolean/status values
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 'oui', 'non'}
        if all(str(sample).lower().strip() in boolean_values for sample in clean_samples[:5]):
            patterns.append('boolean')
            
        # Codes/IDs (alphanumeric patterns)
        if all(re.match(r'^[A-Z0-9\-_]{3,}$', str(sample).upper()) for sample in clean_samples[:3]):
            patterns.append('code_identifier')
            
        # Text/memo content
        avg_length = sum(len(str(sample)) for sample in clean_samples) / len(clean_samples)
        if avg_length > 50 and any(len(str(sample)) > 100 for sample in clean_samples[:3]):
            patterns.append('long_text')
        elif avg_length > 20:
            patterns.append('short_text')
            
        analysis['patterns_detected'] = patterns
        analysis['pattern_type'] = patterns[0] if patterns else 'generic_text'
        analysis['confidence'] = min(1.0, len(patterns) * 0.3 + 0.1)
        
        # Value characteristics
        analysis['value_characteristics'] = {
            'avg_length': avg_length,
            'max_length': max(len(str(sample)) for sample in clean_samples),
            'has_special_chars': any(re.search(r'[^\w\s]', str(sample)) for sample in clean_samples[:3]),
            'numeric_content': any(re.search(r'\d', str(sample)) for sample in clean_samples[:3]),
            'all_caps': any(str(sample).isupper() for sample in clean_samples[:3]),
            'mixed_case': any(re.search(r'[a-z].*[A-Z]|[A-Z].*[a-z]', str(sample)) for sample in clean_samples[:3])
        }
        
        return analysis
    
    def _determine_semantic_category(self, col: ColumnMetadata, sample_analysis: Dict) -> str:
        """
        Determine high-level semantic category based on column name and content.
        """
        name_lower = (col.english_name or col.name).lower()
        patterns = sample_analysis.get('patterns_detected', [])
        
        # Identity/Reference fields
        if ('id' in name_lower and name_lower != 'id') or 'uniqueid' in name_lower:
            return 'foreign_key'
        elif name_lower in ['id', 'key', 'pk']:
            return 'primary_key'
            
        # Contact information
        elif 'phone' in patterns or any(term in name_lower for term in ['tel', 'phone', 'mobile']):
            return 'contact_phone'
        elif 'email' in patterns or 'email' in name_lower:
            return 'contact_email'
        elif 'postal_code' in patterns or any(term in name_lower for term in ['zip', 'postal', 'cp']):
            return 'address_postal'
        elif any(term in name_lower for term in ['address', 'addr', 'street', 'adresse']):
            return 'address_street'
            
        # Person/Entity names
        elif 'person_name' in patterns or any(term in name_lower for term in ['nom', 'name', 'prenom', 'firstname', 'lastname']):
            return 'person_name'
        elif 'company_name' in patterns or any(term in name_lower for term in ['company', 'societe', 'raison', 'sociale']):
            return 'company_name'
            
        # Financial
        elif 'currency' in patterns or any(term in name_lower for term in ['amount', 'price', 'cost', 'montant', 'prix']):
            return 'financial_amount'
        elif 'vat' in name_lower or 'tva' in name_lower:
            return 'tax_identifier'
            
        # Temporal
        elif 'date' in patterns or any(term in name_lower for term in ['date', 'time', 'created', 'updated']):
            return 'temporal'
            
        # Status/Classification
        elif 'boolean' in patterns or any(term in name_lower for term in ['status', 'active', 'enabled', 'statut']):
            return 'status_flag'
        elif any(term in name_lower for term in ['type', 'category', 'classe', 'genre']):
            return 'classification'
            
        # Content/Description
        elif 'long_text' in patterns or any(term in name_lower for term in ['description', 'memo', 'comment', 'note', 'remarque']):
            return 'descriptive_text'
        elif 'code_identifier' in patterns:
            return 'business_code'
            
        return 'generic_attribute'
    
    def _extract_business_meaning(self, name: str, english_name: str, semantic_category: str) -> str:
        """
        Extract business meaning from column name and semantic context.
        """
        text = english_name or name
        
        # Business domain mappings
        meanings = {
            'foreign_key': f"References another entity via {text}",
            'contact_phone': f"Phone/telephone number for {text.replace('tel', '').replace('phone', '').strip() or 'contact'}",
            'contact_email': f"Email address for communication",
            'person_name': f"Person's name field: {text}",
            'company_name': f"Business/company name or identifier",
            'financial_amount': f"Monetary value: {text}",
            'address_street': f"Physical address component: {text}",
            'address_postal': f"Postal/ZIP code for location",
            'temporal': f"Date/time field: {text}",
            'status_flag': f"Status or boolean indicator: {text}",
            'classification': f"Categorization field: {text}",
            'descriptive_text': f"Textual description or notes: {text}",
            'business_code': f"Business identifier or code: {text}",
            'tax_identifier': f"Tax/VAT identification number"
        }
        
        return meanings.get(semantic_category, f"Business attribute: {text}")
    
    def _identify_content_pattern(self, col: ColumnMetadata, sample_analysis: Dict) -> str:
        """
        Identify specific content pattern for better matching.
        """
        patterns = sample_analysis.get('patterns_detected', [])
        characteristics = sample_analysis.get('value_characteristics', {})
        
        if 'phone_number' in patterns:
            return "Phone number format (various international formats)"
        elif 'email' in patterns:
            return "Email address format (user@domain.com)"
        elif 'vat_number' in patterns:
            return "VAT/Tax number format (country code + digits)"
        elif 'postal_code' in patterns:
            return "Postal code format (numeric or alphanumeric)"
        elif 'person_name' in patterns:
            return "Person name format (First Last or similar)"
        elif 'company_name' in patterns:
            return "Company name with legal suffixes"
        elif 'currency' in patterns:
            return "Currency amount with symbols/decimals"
        elif 'date' in patterns:
            return "Date format (ISO, US, or European)"
        elif 'boolean' in patterns:
            return "Boolean/flag values (true/false, 1/0, yes/no)"
        elif 'code_identifier' in patterns:
            return "Alphanumeric codes/identifiers"
        elif 'long_text' in patterns:
            return f"Long text content (avg {characteristics.get('avg_length', 0):.0f} chars)"
        elif 'short_text' in patterns:
            return f"Short text content (avg {characteristics.get('avg_length', 0):.0f} chars)"
        else:
            return f"Generic content ({col.data_type})"
    
    def _generate_confidence_indicators(
        self,
        col: ColumnMetadata, 
        sample_analysis: Dict, 
        semantic_category: str
    ) -> List[str]:
        """
        Generate confidence indicators for mapping quality assessment.
        """
        indicators = []
        
        # Sample quality indicators
        sample_count = sample_analysis.get('clean_sample_count', 0)
        if sample_count >= 3:
            indicators.append(f"Good sample coverage ({sample_count} samples)")
        elif sample_count > 0:
            indicators.append(f"Limited samples ({sample_count})")
        else:
            indicators.append("No sample data available")
            
        # Pattern recognition confidence
        pattern_confidence = sample_analysis.get('confidence', 0.0)
        if pattern_confidence > 0.7:
            indicators.append("High pattern recognition confidence")
        elif pattern_confidence > 0.4:
            indicators.append("Medium pattern recognition confidence")
        else:
            indicators.append("Low pattern recognition confidence")
            
        # Semantic category confidence
        if semantic_category not in ['generic_attribute']:
            indicators.append(f"Clear semantic category: {semantic_category}")
        else:
            indicators.append("Generic semantic category")
            
        # Name clarity
        english_name = col.english_name
        if english_name and english_name != col.name:
            indicators.append("English translation available")
            
        tokens = self._extract_semantic_tokens(col.name, english_name)
        if len(tokens) >= 2:
            indicators.append("Multi-token descriptive name")
        elif len(tokens) == 1:
            indicators.append("Single-token name")
        else:
            indicators.append("Unclear column name")
            
        return indicators

    def _prepare_table_schemas(self, candidate_tables: List[Dict]) -> Dict:
        """Prepare table schemas from candidate tables."""
        table_schemas = {}
        for table in candidate_tables:
            table_name = table.get('table_name')
            if table_name:
                # DEBUG: Log the raw table data
                logging.debug(f"Processing table: {table_name}")
                logging.debug(f"  Raw table data keys: {list(table.keys())}")
                
                # Extract field names from metadata
                fields_raw = table.get('fields', []) or table.get('metadata', {}).get('table_fields', [])
                
                # DEBUG: Log what we found
                logging.debug(f"  fields_raw type: {type(fields_raw)}")
                logging.debug(f"  fields_raw length: {len(fields_raw) if fields_raw else 0}")
                logging.debug(f"  fields_raw sample: {fields_raw[:3] if fields_raw else 'None'}")
                
                if fields_raw and isinstance(fields_raw[0], dict):
                    fields = [f.get('field_name', f.get('name', '')) for f in fields_raw if isinstance(f, dict)]
                    logging.debug(f"  Processed as dict list: {fields[:5]}")
                elif fields_raw and isinstance(fields_raw[0], str):
                    fields = fields_raw
                    logging.debug(f"  Processed as string list: {fields[:5]}")
                else:
                    fields = []
                    logging.debug(f"  No fields found - empty list")
                
                # DEBUG: Log final result
                logging.debug(f"  Final fields for {table_name}: {len(fields)} fields")
                if fields:
                    logging.debug(f"    Sample fields: {fields[:5]}")
                else:
                    logging.warning(f"    WARNING: No fields found for table {table_name}")
                    # Try alternative extraction methods
                    metadata = table.get('metadata', {})
                    logging.debug(f"    Metadata keys: {list(metadata.keys())}")
                    content = metadata.get('content', '')
                    if content:
                        logging.debug(f"    Content sample: {content[:200]}...")
                
                table_schemas[table_name] = {
                    'fields': fields,
                    'table_type': table.get('table_kind', 'Entity'),
                    'rag_score': table.get('composite_score', 0.0)
                }
        
        # DEBUG: Summary of all table schemas
        logging.info(f"Prepared {len(table_schemas)} table schemas:")
        for name, schema in table_schemas.items():
            logging.info(f"  {name}: {len(schema['fields'])} fields ({schema['table_type']})")
        
        return table_schemas

    def _find_best_non_llm_match(
        self,
        source_col: ColumnMetadata,
        target_fields: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Find best match using weighted combination of exact, fuzzy, and semantic strategies."""
        col_name = source_col.english_name or source_col.name
        col_clean = self._normalize_name(col_name)
        
        # Weights for different matching strategies to finetune
        exact_weight = 0.5
        fuzzy_weight = 0.3
        semantic_weight = 0.2
        
        best_match = None
        best_score = 0.0
        best_score_detail = None
        
        # DEBUG: Log the source column being matched
        logging.debug(f"Matching column: '{col_name}' (normalized: '{col_clean}') against {len(target_fields)} targets")
        
        for target_field in target_fields:
            target_clean = self._normalize_name(target_field)
            
            exact_score = 1.0 if col_clean == target_clean else 0.0
            fuzzy_score = self._fuzzy_similarity(col_clean, target_clean) / 100.0  # Normalized to 0-1
            semantic_score = self._semantic_similarity(col_clean, target_clean)
            
            # DEBUG: Log individual scores before penalties
            logging.debug(f"  vs '{target_field}' (normalized: '{target_clean}')")
            logging.debug(f"    Raw scores - Exact: {exact_score:.3f}, Fuzzy: {fuzzy_score:.3f}, Semantic: {semantic_score:.3f}")
            
            # Track penalties applied
            penalties_applied = []
            original_semantic = semantic_score
            
            if self._is_generic_field(target_field):
                semantic_score *= self.generic_field_penalty
                penalties_applied.append(f"generic_field_penalty({self.generic_field_penalty})")
                logging.debug(f"    Applied generic field penalty: {original_semantic:.3f} -> {semantic_score:.3f}")
            if self._has_type_mismatch(source_col, target_field):
                semantic_score *= self.type_mismatch_penalty
                penalties_applied.append(f"type_mismatch_penalty({self.type_mismatch_penalty})")
                logging.debug(f"    Applied type mismatch penalty: semantic -> {semantic_score:.3f}")
            
            # Calculate weighted combination
            weighted_score = (
                exact_score * exact_weight +
                fuzzy_score * fuzzy_weight +
                semantic_score * semantic_weight
            )
            
            # DEBUG: Log weighted calculation
            logging.debug(f"    Weighted: {exact_score:.3f}×{exact_weight} + {fuzzy_score:.3f}×{fuzzy_weight} + {semantic_score:.3f}×{semantic_weight} = {weighted_score:.3f}")
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_match = target_field
                
                # Determine primary method
                raw_scores = {'exact': exact_score, 'fuzzy': fuzzy_score, 'semantic': original_semantic}
                primary_method = max(raw_scores, key=raw_scores.get)
                
                logging.debug(f"    NEW BEST MATCH: {target_field} (score: {weighted_score:.3f}, primary: {primary_method})")
                
                # Create detailed score breakdown
                from ..models.rag_match_model import MatchScoreDetail
                best_score_detail = MatchScoreDetail(
                    exact_score=exact_score,
                    fuzzy_score=fuzzy_score,
                    semantic_score=semantic_score,  # After penalties
                    exact_weight=exact_weight,
                    fuzzy_weight=fuzzy_weight,
                    semantic_weight=semantic_weight,
                    weighted_score=weighted_score,
                    penalties_applied=penalties_applied,
                    primary_method=primary_method
                )
        
        min_threshold = 0.4  
        
        logging.info(f"=== WEIGHTED MATCHING DEBUG for '{col_name}' ===")
        logging.info(f"  Best match found: {best_match}")
        logging.info(f"  Best score: {best_score:.3f}")
        logging.info(f"  Threshold: {min_threshold}")
        logging.info(f"  Target fields count: {len(target_fields)}")
        
        if best_match and best_score >= min_threshold:
            # Determine match type based on primary method
            if best_score_detail.exact_score == 1.0:
                match_type = 'exact'
            elif best_score_detail.fuzzy_score > best_score_detail.semantic_score:
                match_type = 'fuzzy'
            else:
                match_type = 'semantic'
            
            logging.info(f"MATCHED: {col_name} → {best_match} (type: {match_type}, score: {best_score:.3f})")
            
            return {
                'target_field': best_match,
                'confidence': best_score,
                'match_type': match_type,
                'score_detail': best_score_detail,
                'reasoning': best_score_detail.get_score_breakdown()
            }
        else:
            logging.debug(f"  No match found for '{col_name}' - best score {best_score:.3f} below threshold {min_threshold}")
        
        return None
    
    def _fuzzy_similarity(self, str1: str, str2: str) -> float:
        """Calculate fuzzy string similarity using Levenshtein distance."""
        return fuzz.ratio(str1, str2)
    
    def _semantic_similarity(self, str1: str, str2: str) -> float:
        """Calculate semantic similarity (substring, prefix, word overlap)."""
        if str1 == str2:
            return 1.0
        
        scores = []
        
        if str1 in str2 or str2 in str1:
            shorter = min(len(str1), len(str2))
            longer = max(len(str1), len(str2))
            scores.append(0.65 + (0.15 * (shorter / longer)))
        
        common_prefix_len = len(self._common_prefix(str1, str2))
        if common_prefix_len >= 3:
            scores.append(min(0.80, 0.50 + (common_prefix_len / max(len(str1), len(str2)) * 0.30)))
        
        words1 = set(self._split_words(str1))
        words2 = set(self._split_words(str2))
        if words1 and words2:
            common_words = words1 & words2
            if common_words:
                jaccard = len(common_words) / len(words1 | words2)
                important_matches = any(len(w) >= 4 for w in common_words)
                if important_matches:
                    scores.append(0.55 + (jaccard * 0.25))
                else:
                    scores.append(0.40 + (jaccard * 0.20))
        
        return max(scores) if scores else 0.0
    
    def _common_prefix(self, str1: str, str2: str) -> str:
        """Find common prefix of two strings."""
        prefix = []
        for c1, c2 in zip(str1, str2):
            if c1 == c2:
                prefix.append(c1)
            else:
                break
        return ''.join(prefix)
    
    def _split_words(self, text: str) -> List[str]:
        """
        Split camelCase, snake_case, or mixed formats into words.
        Treats 3+ consecutive capitals as acronyms.
        """
        if not text:
            return []
        
        text = re.sub(r'[_\-\.\s]+', ' ', text)
        
        words = []
        for part in text.split():
            if not part:
                continue
            
            camel_split = re.sub(r'(?<=[a-z])(?=[A-Z]{3,})|(?<=[a-z])(?=[A-Z][a-z])|(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', part)
            
            part_words = re.findall(r'[A-Z]{3,}|[A-Z]?[a-z]+|[A-Z](?=[A-Z][a-z]|\d|\W|$)|\d+', camel_split)
            
            words.extend(part_words)
        
        return [w.lower() for w in words if w and len(w) > 0]
    
    def _smart_multi_table_assignment(
        self,
        column_to_table_mappings: Dict[str, List[Dict]],
        column_groups: Dict[str, List[str]],
        table_schemas: Dict,
        candidate_tables: List[Dict],
        primary_table: str = None
    ) -> Dict[str, List[FieldMapping]]:
        """
        Intelligently assign columns to tables based on semantic groups and priorities.
        """
        table_assignments = {table: [] for table in table_schemas.keys()}
        assigned_columns = set()
        table_metadata_lookup = {t['table_name']: t for t in candidate_tables}
        
        # Step 1: Primary table entity fields (exclude FKs/relationships)
        if primary_table and primary_table in table_schemas:
            logging.info(f"Mapping entity fields to primary table: {primary_table}")
            exclude_groups = {'foreign_keys', 'user_references', 'company_references', 
                            'contact_references', 'relationship_data'}
            
            for col_name, candidates in column_to_table_mappings.items():
                # Skip if in excluded groups
                if any(col_name in column_groups.get(g, []) for g in exclude_groups):
                    continue
                
                primary_mapping = next((c['mapping'] for c in candidates if c['table'] == primary_table), None)
                if primary_mapping:
                    table_assignments[primary_table].append(primary_mapping)
                    assigned_columns.add(col_name)
        
        # Step 2: Relationship table mapping (FKs and references)
        logging.info(f"Mapping relationship fields to relationship tables")
        relation_tables = [t for t in table_schemas.keys() 
                          if table_metadata_lookup.get(t, {}).get('table_kind') == 'Relation']
        
        for semantic_group in ['foreign_keys', 'user_references', 'company_references', 
                               'contact_references', 'relationship_data']:
            group_cols = column_groups.get(semantic_group, [])
            for col_name in group_cols:
                if col_name in assigned_columns:
                    continue
                
                candidates = column_to_table_mappings.get(col_name, [])
                # Prefer relationship tables
                best_rel_match = next(
                    (c for c in candidates if c['table'] in relation_tables and c['confidence'] >= 0.3),
                    None
                )
                
                if best_rel_match:
                    table_assignments[best_rel_match['table']].append(best_rel_match['mapping'])
                    assigned_columns.add(col_name)
        
        # Step 3: Remaining columns to best matching table
        for col_name, candidates in column_to_table_mappings.items():
            if col_name in assigned_columns or not candidates:
                continue
            
            best_candidate = candidates[0]  # Already sorted by confidence
            if best_candidate['confidence'] >= self.min_confidence_threshold:
                table_assignments[best_candidate['table']].append(best_candidate['mapping'])
                assigned_columns.add(col_name)
        
        return table_assignments
    
    def _judge_needs_refinement(
        self,
        table_assignments: Dict[str, List[FieldMapping]],
        file_analysis: FileAnalysisResult
    ) -> bool:
        """
        Judge analyzes mapping quality and decides if refinement is needed.
        
        Criteria for refinement:
        1. Overloaded fields (>2 columns to same field)
        2. Many identical confidence scores (algorithmic issue)
        3. Too many generic field mappings
        4. Low overall confidence
        """
        logging.info("Judge analyzing mapping quality...")
        
        reasons = []
        
        for table_name, mappings in table_assignments.items():
            if not mappings:
                continue
            
            # Check 1: Overloaded fields
            target_field_counts = {}
            for mapping in mappings:
                if mapping.target_column:
                    target_field_counts[mapping.target_column] = target_field_counts.get(mapping.target_column, 0) + 1
            
            overloaded = {f: c for f, c in target_field_counts.items() if c > self.max_same_field_mappings}
            if overloaded:
                reasons.append(f"Table {table_name}: {len(overloaded)} overloaded fields")
            
            # Check 2: Identical confidence scores
            confidences = [m.confidence_score for m in mappings if m.confidence_score > 0]
            if len(confidences) > 3 and len(set(confidences)) == 1:
                reasons.append(f"Table {table_name}: identical confidence scores detected")
            
            # Check 3: Generic field mappings
            generic_mappings = [m for m in mappings if m.target_column and self._is_generic_field(m.target_column)]
            if len(generic_mappings) > len(mappings) * 0.3:
                reasons.append(f"Table {table_name}: {len(generic_mappings)} generic field mappings")
            
            # Check 4: Low average confidence
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            if avg_conf < 0.65:
                reasons.append(f"Table {table_name}: low avg confidence ({avg_conf:.2f})")
        
        if reasons:
            logging.warning(f"Judge found {len(reasons)} issues:")
            for reason in reasons:
                logging.warning(f"   • {reason}")
            return True
        
        return False
    
    def _refine_all_table_mappings(
        self,
        table_assignments: Dict[str, List[FieldMapping]],
        table_schemas: Dict
    ) -> Dict[str, List[FieldMapping]]:
        """Refine mappings for all tables."""
        refined_assignments = {}
        
        for table_name, mappings in table_assignments.items():
            refined_mappings = self._refine_table_mappings(table_name, mappings)
            refined_assignments[table_name] = refined_mappings
        
        return refined_assignments
    
    def _refine_table_mappings(
        self,
        table_name: str,
        mappings: List[FieldMapping]
    ) -> List[FieldMapping]:
        """
        Refine mappings for a single table by detecting overloaded fields.
        Removes low-confidence mappings when too many columns map to same field.
        """
        logging.info(f"Refining mappings for table: {table_name} ({len(mappings)} mappings)")
        
        # Count mappings per target field
        target_field_analysis = {}
        for mapping in mappings:
            if mapping.target_column:
                if mapping.target_column not in target_field_analysis:
                    target_field_analysis[mapping.target_column] = []
                target_field_analysis[mapping.target_column].append(mapping)
        
        # Detect overloaded fields
        overloaded_fields = {}
        for field, field_mappings in target_field_analysis.items():
            count = len(field_mappings)
            if count <= self.max_same_field_mappings:
                continue
            
            is_generic = self._is_generic_field(field)
            confidence_scores = [m.confidence_score for m in field_mappings]
            has_identical_scores = len(set(confidence_scores)) == 1
            
            if is_generic or (has_identical_scores and count > self.max_same_field_mappings):
                reason = 'generic_field' if is_generic else 'identical_scores'
                overloaded_fields[field] = {'count': count, 'reason': reason, 'mappings': field_mappings}
                logging.warning(f"Overloaded: '{field}' ({count} mappings, reason: {reason})")
        
        if not overloaded_fields:
            logging.info("No overloaded fields detected")
            return mappings
        
        # Remove low-confidence mappings to overloaded fields
        refined_mappings = []
        removed_count = 0
        
        for mapping in mappings:
            if mapping.target_column in overloaded_fields:
                if mapping.confidence_score < self.refinement_confidence_threshold:
                    logging.info(f"Removed: {mapping.source_column} → {mapping.target_column} "
                               f"(score: {mapping.confidence_score:.2f})")
                    removed_count += 1
                else:
                    refined_mappings.append(mapping)
            else:
                refined_mappings.append(mapping)
        
        if removed_count > 0:
            logging.warning(f"Refinement: Removed {removed_count} suspicious mappings")
            logging.info(f"Mappings reduced: {len(mappings)} → {len(refined_mappings)}")
        
        return refined_mappings

    def _filter_and_validate_table_mappings(
        self,
        table_assignments: Dict[str, List[FieldMapping]],
        file_analysis: FileAnalysisResult,
        candidate_tables: List[Dict]
    ) -> List[TableFieldMapping]:
        """Filter tables with too few mappings and validate."""
        table_metadata_lookup = {t['table_name']: t for t in candidate_tables}
        valid_mappings = []
        
        for table_name, mappings in table_assignments.items():
            if len(mappings) < self.min_columns_per_table:
                logging.debug(f"Skipping {table_name}: only {len(mappings)} columns mapped")
                continue
            
            
            all_target_fields = []
            for t in candidate_tables:
                if t['table_name'] == table_name:
                    fields_raw = t.get('fields', [])
                    # Handle both list of strings and list of dicts
                    if fields_raw and isinstance(fields_raw[0], dict):
                        all_target_fields = [f.get('field_name', f.get('name', '')) for f in fields_raw if isinstance(f, dict)]
                    elif fields_raw and isinstance(fields_raw[0], str):
                        all_target_fields = fields_raw
                    break
            
            validation = self._validate_mappings(mappings, file_analysis.columns, all_target_fields)
            
            avg_confidence = sum(m.confidence_score for m in mappings) / len(mappings) if mappings else 0.0
            table_type = table_metadata_lookup.get(table_name, {}).get('table_kind', 'Entity')
            
            valid_mappings.append(TableFieldMapping(
                table_name=table_name,
                table_type=table_type,
                mappings=mappings,
                validation=validation,
                confidence=avg_confidence
            ))
        
        valid_mappings.sort(key=lambda x: len(x.mappings), reverse=True)
        return valid_mappings
    
    def _determine_insertion_order(
        self,
        table_mappings: List[TableFieldMapping],
        table_schemas: Dict
    ) -> List[TableFieldMapping]:
        """Determine insertion order (Entity tables first, then Relation tables)."""
        for i, tm in enumerate(table_mappings):
            tm.insertion_order = 1 if tm.table_type == "Entity" else 2
        
        table_mappings.sort(key=lambda x: x.insertion_order)
        return table_mappings
    
    def _create_final_result(
        self,
        file_analysis: FileAnalysisResult,
        valid_table_mappings: List[TableFieldMapping],
        needs_refinement: bool
    ) -> MultiTableMappingResult:
        """Create final result object."""
        total_mapped = sum(len(tm.mappings) for tm in valid_table_mappings)
        total_source = file_analysis.structure.total_columns
        overall_coverage = (total_mapped / total_source * 100) if total_source > 0 else 0.0
        
        mapped_columns = set()
        for tm in valid_table_mappings:
            mapped_columns.update(m.source_column for m in tm.mappings if m.target_column)
        
        all_columns = {col.name for col in file_analysis.columns}
        unmapped = list(all_columns - mapped_columns)
        
        avg_confidence = sum(tm.confidence for tm in valid_table_mappings) / len(valid_table_mappings) if valid_table_mappings else 0.0
        overall_confidence = self._calculate_confidence_level(avg_confidence)
        
        is_valid = overall_coverage >= 50.0 and avg_confidence >= 0.6
        requires_review = not is_valid or overall_coverage < 70.0
        
        return MultiTableMappingResult(
            source_file=file_analysis.structure.file_name,
            total_source_columns=total_source,
            table_mappings=valid_table_mappings,
            overall_coverage=overall_coverage,
            overall_confidence=overall_confidence,
            unmapped_columns=unmapped,
            is_valid=is_valid,
            requires_review=requires_review,
            requires_refinement=needs_refinement
        )
    
    def _calculate_confidence_level(self, score: float) -> str:
        """Calculate confidence level from score."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _log_summary(self, result: MultiTableMappingResult):
        """Log detailed summary of mapping results with weighted scoring details."""
        logging.info("=" * 80)
        logging.info("MULTI-TABLE MAPPING SUMMARY")
        logging.info("=" * 80)
        logging.info(f"Source: {result.source_file}")
        logging.info(f"Total columns: {result.total_source_columns}")
        logging.info(f"Tables mapped: {len(result.table_mappings)}")
        logging.info(f"Overall coverage: {result.overall_coverage:.1f}%")
        logging.info(f"Overall confidence: {result.overall_confidence}")
        logging.info(f"Refinement applied: {'Yes' if result.requires_refinement else 'No'}")
        logging.info(f"Requires review: {'Yes' if result.requires_review else 'No'}")
        
        for tm in result.table_mappings:
            logging.info(f"\n  {tm.table_name} ({tm.table_type}) [Order: {tm.insertion_order}]:")
            logging.info(f"     • Columns: {len(tm.mappings)}")
            logging.info(f"     • Coverage: {tm.validation.mapping_coverage_percent:.1f}%")
            logging.info(f"     • Confidence: {tm.validation.confidence_level}")
            
            logging.info(f"     • Detailed Mappings:")
            for mapping in tm.mappings:
                if mapping.score_detail:
                    score_breakdown = mapping.score_detail.get_score_breakdown()
                    penalties = ", ".join(mapping.score_detail.penalties_applied) if mapping.score_detail.penalties_applied else "None"
                    logging.info(f"       - {mapping.source_column} → {mapping.target_column}")
                    logging.info(f"         Score: {score_breakdown}")
                    logging.info(f"         Method: {mapping.match_type} (Primary: {mapping.score_detail.primary_method})")
                    logging.info(f"         Penalties: {penalties}")
                else:
                    logging.info(f"       - {mapping.source_column} → {mapping.target_column}")
                    logging.info(f"         Score: {mapping.confidence_score:.2f} ({mapping.match_type})")
        
        if result.unmapped_columns:
            logging.warning(f"\n  Unmapped columns ({len(result.unmapped_columns)}):")
            logging.warning(f"     {', '.join(result.unmapped_columns[:10])}")
        
        logging.info("=" * 80)

    def _normalize_name(self, name: str) -> str:
        """Normalize field name for comparison."""
        return re.sub(r'[^a-z0-9]', '', name.lower())
    
    def _is_generic_field(self, field_name: str) -> bool:
        """Check if field is generic/catch-all."""
        field_lower = field_name.lower()
        return any(term in field_lower for term in self.generic_field_terms)
    
    def _has_type_mismatch(self, source_col: ColumnMetadata, target_field: str) -> bool:
        """Detect structured data → unstructured field mismatch."""
        target_lower = target_field.lower()
        is_unstructured = any(term in target_lower for term in ['memo', 'blob', 'metadata', 'data'])
        if not is_unstructured:
            return False
        
        source_type_lower = source_col.data_type.lower()
        is_structured = any(term in source_type_lower for term in ['date', 'number', 'integer', 'decimal', 'numeric'])
        return is_structured

    def _validate_mappings(self, mappings: List[FieldMapping], source_columns: List[ColumnMetadata], target_fields: List[str]) -> MappingValidationResult:
        """Validate mapping quality with coverage, confidence, and unmapped tracking."""
        total_source = len(source_columns)
        mapped_count = len([m for m in mappings if m.target_column])
        coverage_percent = (mapped_count / total_source * 100) if total_source > 0 else 0
        
        mapped_source = {m.source_column for m in mappings if m.target_column}
        unmapped_source = [col.name for col in source_columns if col.name not in mapped_source]
        
        mapped_targets = {m.target_column for m in mappings if m.target_column}
        unmapped_targets = [f for f in target_fields if f not in mapped_targets]
        
        avg_confidence = sum(m.confidence_score for m in mappings if m.target_column) / mapped_count if mapped_count else 0
        
        if avg_confidence >= 0.8 and coverage_percent >= 80:
            confidence_level = "high"
        elif avg_confidence >= 0.6 and coverage_percent >= 60:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        issues = []
        warnings = []
        
        if coverage_percent < 50:
            issues.append(f"Low mapping coverage: {coverage_percent:.1f}%")
        if avg_confidence < 0.6:
            issues.append(f"Low average confidence: {avg_confidence:.2f}")
        if unmapped_source:
            warnings.append(f"{len(unmapped_source)} source columns unmapped")
        
        low_confidence = [m for m in mappings if m.target_column and m.confidence_score < 0.7]
        if low_confidence:
            warnings.append(f"{len(low_confidence)} mappings have low confidence")
        
        is_valid = len(issues) == 0 and coverage_percent >= 50
        requires_review = confidence_level == "low" or len(issues) > 0
        
        return MappingValidationResult(
            is_valid=is_valid,
            confidence_level=confidence_level,
            total_mappings=total_source,
            mapped_count=mapped_count,
            unmapped_source_columns=unmapped_source,
            unmapped_target_columns=unmapped_targets,
            issues=issues,
            warnings=warnings,
            mapping_coverage_percent=coverage_percent,
            requires_review=requires_review
        )