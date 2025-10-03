from typing import List, Tuple, Dict
import re
import logging

from ..config.settings import get_settings

class UniversalTranslator:
    """Universal translator that normalizes all languages to English for better semantic matching"""
    
    def __init__(self, openai_client=None):
        """
        Initialize translator with OpenAI client.
        
        Args:
            openai_client: Optional pre-configured OpenAI client. If None, creates one from settings.
        """
        # Use provided client or create from settings
        if openai_client is not None:
            self.client = openai_client
        else:
            # Get settings and create OpenAI client
            from openai import OpenAI
            settings = get_settings()
            self.client = OpenAI(api_key=settings.openai_api_key)
        
        self.translation_cache = {}  # Cache translations to avoid repeated API calls
        
        # Common technical terms that shouldn't be translated
        self.technical_terms = {
            'id', 'key', 'ref', 'code', 'type', 'status', 'url', 'api', 'json', 'xml', 
            'csv', 'sql', 'uuid', 'guid', 'timestamp', 'datetime', 'boolean', 'integer',
            'varchar', 'text', 'blob', 'index', 'primary', 'foreign', 'null', 'auto'
        }
        
    def translate_column_names(self, columns: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Translate column names to English and return both translated names and mapping
        
        Args:
            columns: List of column names in any language
            
        Returns:
            Tuple of (translated_columns, translation_mapping)
        """
        if not columns:
            return [], {}
            
        # Filter out columns that are already in English or are technical terms
        columns_to_translate = []
        english_columns = []
        translation_mapping = {}
        
        for col in columns:
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            
            # Check if column is already English (contains only English words)
            if self._is_english_column(col):
                english_columns.append(col)
                translation_mapping[col] = col  # No translation needed
            else:
                columns_to_translate.append(col)
        
        # Batch translate non-English columns
        if columns_to_translate:
            batch_translations = self._batch_translate_columns(columns_to_translate)
            translation_mapping.update(batch_translations)
            english_columns.extend(batch_translations.values())
        
        return english_columns, translation_mapping
    
    def _is_english_column(self, column: str) -> bool:
        """Check if a column name is already in English"""
        col_clean = column.lower().replace('_', '').replace('-', '').replace(' ', '')
        
        # If it's a technical term, consider it English
        if col_clean in self.technical_terms:
            return True
            
        # Check for common English patterns
        english_patterns = [
            r'^[a-z_\-\s0-9]+$',  # Only English letters, numbers, underscores, hyphens, spaces
        ]
        
        # Common English words in database columns
        english_indicators = [
            'name', 'email', 'phone', 'address', 'date', 'time', 'created', 'updated',
            'first', 'last', 'user', 'customer', 'order', 'product', 'price', 'total',
            'description', 'title', 'category', 'status', 'active', 'deleted', 'count',
            'amount', 'value', 'number', 'code', 'reference', 'contact', 'company'
        ]
        
        # If column contains English indicators, likely English
        col_words = re.findall(r'\b\w+\b', column.lower())
        if any(word in english_indicators for word in col_words):
            return True
            
        # French/other language indicators (if found, not English)
        non_english_indicators = [
            'expéditeur', 'destinataire', 'société', 'objet', 'corps', 'message',
            'saisie', 'auteur', 'état', 'suivi', 'catégorie', 'diffusion', 'priorité',
            'référence', 'campagne', 'lieu', 'désignation', 'programmé', 'fin'
        ]
        
        if any(indicator in column.lower() for indicator in non_english_indicators):
            return False
            
        # Default to English if unsure (conservative approach)
        return True
    
    def _batch_translate_columns(self, columns: List[str]) -> Dict[str, str]:
        """Translate multiple columns in a single API call for efficiency"""
        if not columns:
            return {}
            
        # Check cache first
        cached_results = {}
        columns_to_translate = []
        
        for col in columns:
            if col in self.translation_cache:
                cached_results[col] = self.translation_cache[col]
            else:
                columns_to_translate.append(col)
        
        if not columns_to_translate:
            return cached_results
            
        # Prepare batch translation prompt
        columns_text = '\n'.join([f"- {col}" for col in columns_to_translate])
        
        prompt = f"""Translate the following database column names to English. These are field names from a business database.

Rules:
1. Translate to clear, descriptive English database field names
2. Use standard database naming conventions (lowercase, underscores)
3. Keep technical terms (ID, REF, CODE, etc.) as-is
4. Make translations specific and business-appropriate
5. Return ONLY the translated names, one per line, in the same order

Column names to translate:
{columns_text}

English translations:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert database architect who translates field names to standard English database conventions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            translated_lines = response.choices[0].message.content.strip().split('\n')
            
            # Parse results and create mapping
            translations = {}
            for i, col in enumerate(columns_to_translate):
                if i < len(translated_lines):
                    translated = translated_lines[i].strip().replace('- ', '').replace('* ', '')
                    if translated:
                        translations[col] = translated
                        self.translation_cache[col] = translated  # Cache for future use
                    else:
                        translations[col] = col  # Fallback to original
                else:
                    translations[col] = col  # Fallback to original
            
            # Combine with cached results
            translations.update(cached_results)
            return translations
            
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            # Fallback: return original columns
            fallback_translations = {col: col for col in columns_to_translate}
            fallback_translations.update(cached_results)
            return fallback_translations
    
    def translate_domain_context(self, text: str) -> str:
        """Translate domain context text to English"""
        if not text or self._is_mostly_english(text):
            return text
            
        prompt = f"""Translate the following business context to clear English:

{text}

English translation:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a business translator. Translate to clear, professional English."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Context translation failed: {e}")
            return text  # Fallback to original
    
    def _is_mostly_english(self, text: str) -> bool:
        """Check if text is mostly English"""
        # Simple heuristic - if no obvious non-English characters/words
        non_english_indicators = ['é', 'è', 'ê', 'ë', 'à', 'â', 'ç', 'ù', 'û', 'ô', 'î', 'ï']
        return not any(indicator in text.lower() for indicator in non_english_indicators)
