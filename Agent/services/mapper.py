"""
Mapper Service - Unified multi-table mapping using exact and LLM batch calls.
Replaces llm_field_matcher, field_mapper, and multi_table_mapper.
"""
import logging
import json
from typing import List, Dict, Any
from openai import OpenAI
from ..models.file_analysis_model import FileAnalysisResult, ColumnMetadata
from ..models.rag_match_model import FieldMapping, FieldMappingResult, MappingValidationResult

class Mapper:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self.generic_field_terms = [
            'metadata', 'memo', 'data', 'info', 'information',
            'ai', 'scoring', 'generic', 'misc', 'miscellaneous',
            'other', 'extra', 'additional', 'temp', 'temporary',
            'custom', 'field', 'column', 'value', 'blob'
        ]
        self.client = OpenAI()

    def map_to_multiple_tables(self, file_analysis: FileAnalysisResult, candidate_tables: List[Dict], max_tables: int = 5) -> FieldMappingResult:
        source_columns = file_analysis.columns
        prioritized_tables = candidate_tables[:max_tables]
        mapping_objs = []
        unmapped_columns = {col.name: col for col in source_columns}

        # Log the columns of the selected table
        selected_table_fields = prioritized_tables[0].get('metadata', []) if prioritized_tables else []
        logging.info(f"Selected table '{prioritized_tables[0]['table_name'] if prioritized_tables else None}' fields: {selected_table_fields}")

        # Step 1: Exact match for each table
        for table in prioritized_tables:
            table_name = table.get('table_name')
            target_fields = table.get('metadata', [])
            table_kind = table.get('table_kind', 'Entity')
            for col_name, col in list(unmapped_columns.items()):
                for target_field in target_fields:
                    if self._normalize_name(col.name) == self._normalize_name(target_field):
                        mapping_objs.append(FieldMapping(
                            source_column=col.name,
                            source_column_english=getattr(col, 'english_name', col.name),
                            target_column=target_field,
                            confidence_score=1.0,
                            match_type='exact',
                            data_type_compatible=True,
                            source_data_type=getattr(col, 'data_type', None),
                            sample_source_values=getattr(col, 'sample_values', [])[:3],
                            validation_notes=[],
                        ))
                        del unmapped_columns[col_name]
                        break

        # Step 2: Batch LLM mapping for remaining columns, iterating over tables
        for table in prioritized_tables:
            if not unmapped_columns:
                break
            table_name = table.get('table_name')
            target_fields = table.get('fields', [])
            table_kind = table.get('table_kind', 'Entity')
            table_metadata = table.get('metadata', {})
            batch_prompt = self._build_batch_prompt(list(unmapped_columns.values()), target_fields, table_name, table_metadata)
            llm_results = self._batch_llm_call(batch_prompt)
            for match in llm_results.get('matches', []):
                source_column = match.get('source_column')
                target_field = match.get('target_field')
                confidence = match.get('confidence', 0.0)
                reasoning = match.get('reasoning', '')
                if source_column in unmapped_columns and target_field and target_field != 'NO_MATCH' and confidence >= 0.5:
                    col = unmapped_columns[source_column]
                    mapping_objs.append(FieldMapping(
                        source_column=col.name,
                        source_column_english=getattr(col, 'english_name', col.name),
                        target_column=target_field,
                        confidence_score=confidence,
                        match_type='llm',
                        data_type_compatible=True,
                        source_data_type=getattr(col, 'data_type', None),
                        sample_source_values=getattr(col, 'sample_values', [])[:3],
                        validation_notes=[reasoning] if reasoning else [],
                    ))
                    del unmapped_columns[source_column]

        # Step 3: Mark remaining columns as UNMAPPED
        for col_name, col in unmapped_columns.items():
            mapping_objs.append(FieldMapping(
                source_column=col.name,
                source_column_english=getattr(col, 'english_name', col.name),
                target_column=None,
                confidence_score=0.0,
                match_type='UNMAPPED',
                data_type_compatible=True,
                source_data_type=getattr(col, 'data_type', None),
                sample_source_values=getattr(col, 'sample_values', [])[:3],
                validation_notes=["No confident match found. Requires human review."]
            ))

        # Validation
        validation = self._validate_mappings(
            mapping_objs,
            source_columns,
            [f for t in prioritized_tables for f in t.get('fields', [])]
        )

        return FieldMappingResult(
            mappings=mapping_objs,
            validation=validation,
            source_table_name=file_analysis.structure.file_name,
            target_table_name=prioritized_tables[0]['table_name'] if prioritized_tables else None,
            mapping_method="automatic"
        )

    def _normalize_name(self, name: str) -> str:
        return ''.join(e for e in name.lower() if e.isalnum())

    def _make_field_mapping(self, col, target_field, confidence, match_type, table_name, table_kind, reasoning=None):
        return {
            'source_column': col.name,
            'source_column_english': getattr(col, 'english_name', col.name),
            'target_column': target_field,
            'confidence_score': confidence,
            'match_type': match_type,
            'table': table_name,
            'table_kind': table_kind,
            'reasoning': reasoning or ''
        }

    def _build_batch_prompt(self, columns, target_fields, table_name, table_metadata):
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

    def _batch_llm_call(self, prompt: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            return result if isinstance(result, dict) and "matches" in result else {"matches": []}
        except Exception as e:
            logging.error(f"Batch LLM call failed: {e}")
            return {"matches": []}

    def _get_system_prompt(self) -> str:
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
            "8. Avoid at all costs mapping to or from fields with captial F in their name.\n"
            "Return JSON: {matches: [{source_column, target_field, confidence, reasoning}]}"
        )

    def _validate_mappings(self, mappings: List[FieldMapping], source_columns: List[ColumnMetadata], target_fields: List[str]) -> MappingValidationResult:
        # Placeholder for validation logic
        return MappingValidationResult(
            is_valid=True,
            issues=[]
        )