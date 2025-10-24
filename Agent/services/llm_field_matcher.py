"""
LLM Field Matcher Service - Uses AI to intelligently match CSV columns to database fields.
Provides contextual reasoning and semantic understanding beyond simple string matching.
"""
import logging
import json
from typing import List, Optional, Dict, Any
from openai import OpenAI

from ..models.file_analysis_model import ColumnMetadata


class LLMFieldMatcherService:
    """
    Service that uses LLM to intelligently match source columns to target fields.
    Provides contextual reasoning about field semantics, data types, and business logic.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize LLM field matcher.
        
        Args:
            api_key: OpenAI API key (if None, will use env variable)
            model: Model to use (default: gpt-4o-mini for cost efficiency)
        """
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.temperature = 0.1  # Low temperature for consistent matching
        
    def find_llm_match(
        self,
        source_col: ColumnMetadata,
        target_fields: List[str],
        source_col_english: str,
        table_context: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to find the best matching target field for a source column.

        Args:
            source_col: Source column metadata with samples
            target_fields: List of available target field names
            source_col_english: English translation of source column name
            table_context: Optional context about the target table

        Returns:
            Dict with 'target_field', 'confidence', 'reasoning', 'table', or None if no match
        """
        try:
            prompt = self._build_matching_prompt(
                source_col,
                target_fields,
                source_col_english,
                table_context
            )

            logging.debug(f"🤖 Asking LLM to match: {source_col.name}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            # Validate result structure
            if not result.get("target_field") or result.get("target_field") == "NO_MATCH":
                logging.debug(f"LLM found no suitable match for {source_col.name}")
                return None

            # Normalize confidence to 0-1 range
            confidence = float(result.get("confidence", 0)) / 100.0

            # Extract table name from table_context if not provided in the result
            table_name = result.get("table")
            if not table_name and table_context:
                table_name = table_context.split("Table:")[1].strip() if "Table:" in table_context else "unknown"

            logging.info(f"🤖 LLM matched: {source_col.name} → {result['target_field']} "
                        f"(confidence: {confidence:.2f}, reasoning: {result.get('reasoning', 'N/A')[:50]}...)")

            return {
                "target_field": result["target_field"],
                "confidence": confidence,
                "reasoning": result.get("reasoning", ""),
                "match_type": "llm",
                "table": table_name
            }

        except Exception as e:
            logging.error(f"LLM matching failed for {source_col.name}: {str(e)}")
            return None
            
        except Exception as e:
            logging.error(f"LLM matching failed for {source_col.name}: {str(e)}")
            return None
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for field matching."""
        return """You are an expert data engineer specializing in database schema mapping and ETL processes.

Your task is to match CSV column names to database table fields by analyzing:
- Column names and their semantic meaning
- Sample data values and patterns
- Data types and formats
- Business domain context
- Common database naming conventions

Guidelines:
1. Consider the SEMANTIC MEANING, not just string similarity
2. Avoid mapping structured data (dates, numbers) to generic memo/blob fields
3. Penalize matches to overly generic fields (metadata, memo, misc, data, blob, etc.)
4. Consider data type compatibility
5. Use sample values to validate the match makes sense
6. Be conservative - if uncertain, return NO_MATCH rather than guessing
7. Provide clear reasoning for your decision

Return JSON with:
{
  "target_field": "field_name" or "NO_MATCH",
  "confidence": 0-100 (integer),
  "reasoning": "brief explanation of why this is the best match"
}"""
    
    def _build_matching_prompt(
        self,
        source_col: ColumnMetadata,
        target_fields: List[str],
        source_col_english: str,
        table_context: Optional[str] = None,
        table_name: str = "Unknown Table"
    ) -> str:
        """Build the user prompt for matching a specific column."""
        
        # Format sample values
        samples = source_col.sample_values[:5] if source_col.sample_values else []
        sample_text = ", ".join([f'"{v}"' for v in samples]) if samples else "No samples available"
        
        # Build prompt
        prompt = f"""Match this source column to the best target database field:

SOURCE COLUMN:
- Table name: {table_name}
- Name: {source_col.name}
- English Name: {source_col_english}
- Data Type: {source_col.data_type}
- Sample Values: {sample_text}

TARGET FIELDS AVAILABLE:
{self._format_target_fields(target_fields)}
"""
        
        if table_context:
            prompt += f"\nTABLE CONTEXT:\n{table_context}\n"
        
        prompt += """
ANALYSIS REQUIRED:
1. Which target field best represents this source column's semantic meaning?
2. Are the data types compatible?
3. Do the sample values make sense for the target field?
4. Is the target field too generic (memo, blob, metadata) for structured data?

Return your answer as JSON with target_field, confidence (0-100), and reasoning.
If no good match exists, return "NO_MATCH" as target_field with confidence 0."""
        
        return prompt
    
    def _format_target_fields(self, fields: List[str]) -> str:
        """Format target fields list for the prompt."""
        return "\n".join([f"  - {field}" for field in fields])
    
    def batch_match_fields(
        self,
        source_columns: List[ColumnMetadata],
        target_fields: List[str],
        english_names: Dict[str, str],
        table_context: Optional[str] = None
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Match multiple columns in a single batch (more efficient for API calls).
        
        Args:
            source_columns: List of source columns to match
            target_fields: List of target field names
            english_names: Dict mapping column names to English translations
            table_context: Optional table context
            
        Returns:
            Dict mapping source column names to match results
        """
        results = {}
        
        for col in source_columns:
            # Handle both ColumnMetadata objects and strings
            if isinstance(col, str):
                col_name = col
                english_name = english_names.get(col_name, col_name)
                match = self.find_llm_match(
                    ColumnMetadata(
                        name=col_name,
                        data_type='string',
                        sample_values=[]),  # Create a temporary ColumnMetadata object
                    target_fields,
                    english_name,
                    table_context
                )
            else:
                col_name = col.name
                english_name = english_names.get(col_name, col_name)
                match = self.find_llm_match(col, target_fields, english_name, table_context)
            
            results[col_name] = match
            
        return results
