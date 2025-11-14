"""
Token-Based Semantic Matcher for Database Field Mapping

CORE RESPONSIBILITIES:
- Tokenize field names intelligently (snake_case, camelCase, abbreviations)
- Calculate multiple similarity metrics (Jaccard, Cosine, Weighted)
- Provide structured match results with confidence levels
- Optimize for performance with caching and batch processing
"""

import re
import logging
import unicodedata
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import Counter
import math

logger = logging.getLogger(__name__)


class AbbreviationExpander:
    """
    Expands common CRM and business abbreviations to their full forms.
    Helps normalize field names for better semantic matching.
    """
    
    DEFAULT_ABBREVIATIONS = {
        # CRM Entities
        'opp': 'opportunity',
        'oppo': 'opportunity',
        'cont': 'contact',
        'acc': 'account',
        'acct': 'account',
        'addr': 'address',
        'cust': 'customer',
        'org': 'organization',
        'dept': 'department',
        'mgr': 'manager',
        'emp': 'employee',
        'prod': 'product',
        'cat': 'category',
        
        # Common Fields
        'id': 'identifier',
        'amt': 'amount',
        'qty': 'quantity',
        'desc': 'description',
        'num': 'number',
        'img': 'image',
        'pic': 'picture',
        'doc': 'document',
        'ref': 'reference',
        'rel': 'relation',
        'val': 'value',
        'info': 'information',
        'spec': 'specification',
        
        # Config/Settings
        'config': 'configuration',
        'pref': 'preference',
        'temp': 'temporary',
        'prev': 'previous',
        'curr': 'current',
        'def': 'default',
        'opt': 'optional',
        'req': 'required',
        
        # Math/Aggregation
        'max': 'maximum',
        'min': 'minimum',
        'avg': 'average',
        'tot': 'total',
        'calc': 'calculated',
        
        # Access/Security
        'priv': 'private',
        'pub': 'public',
        'auth': 'authorized',
        'admin': 'administrator',
        
        # Technical
        'sys': 'system',
        'app': 'application',
        'proc': 'process',
        'auto': 'automatic',
        
        # French common abbreviations
        'soc': 'societe',
        'comp': 'company',
        'tel': 'telephone',
        'adr': 'address',
        'nom': 'name',
        'prenom': 'firstname',
    }
    
    def __init__(self, custom_abbreviations: Optional[Dict[str, str]] = None):
        """
        Initialize abbreviation expander with default and optional custom abbreviations.
        
        Args:
            custom_abbreviations: Optional dict of custom abbreviation mappings
        """
        self.abbreviations = self.DEFAULT_ABBREVIATIONS.copy()
        if custom_abbreviations:
            self.abbreviations.update(custom_abbreviations)
        logger.debug(f"Initialized AbbreviationExpander with {len(self.abbreviations)} abbreviations")
    
    def expand(self, token: str) -> str:
        """
        Expand an abbreviation to its full form if found, otherwise return original.
        
        Args:
            token: The token to potentially expand
            
        Returns:
            Expanded form if found in dictionary, otherwise original token
        """
        if not token:
            return token
        
        token_lower = token.lower()
        expanded = self.abbreviations.get(token_lower, token)
        
        if expanded != token:
            logger.debug(f"Expanded abbreviation: '{token}' -> '{expanded}'")
        
        return expanded
    
    def add_custom_abbreviation(self, abbrev: str, expansion: str) -> None:
        """
        Add a custom abbreviation at runtime.
        
        Args:
            abbrev: The abbreviation to add
            expansion: The full form expansion
        """
        if not abbrev or not expansion:
            logger.warning(f"Cannot add empty abbreviation or expansion")
            return
        
        abbrev_lower = abbrev.lower()
        self.abbreviations[abbrev_lower] = expansion.lower()
        logger.info(f"Added custom abbreviation: '{abbrev}' -> '{expansion}'")
    
    def load_from_dict(self, custom_dict: Dict[str, str]) -> None:
        """
        Bulk load custom abbreviations from a dictionary.
        
        Args:
            custom_dict: Dictionary of abbreviation mappings to load
        """
        if not custom_dict:
            logger.warning("Attempted to load from empty dictionary")
            return
        
        count = 0
        for abbrev, expansion in custom_dict.items():
            if abbrev and expansion:
                self.abbreviations[abbrev.lower()] = expansion.lower()
                count += 1
        
        logger.info(f"Loaded {count} custom abbreviations from dictionary")




class FieldNameTokenizer:
    """
    Intelligently tokenizes field names by handling multiple naming conventions:
    - snake_case: opportunity_contact_id
    - camelCase: OpportunityContactID
    - Mixed: oppoContID
    - Numbers: address1Line2
    - Special characters and unicode
    """
    
    def __init__(self, abbreviation_expander: Optional[AbbreviationExpander] = None):
        """
        Initialize tokenizer with optional abbreviation expander.
        
        Args:
            abbreviation_expander: Optional expander for abbreviations
        """
        self.expander = abbreviation_expander or AbbreviationExpander()
        logger.debug("Initialized FieldNameTokenizer")
    
    def tokenize(self, field_name: str) -> List[str]:
        """
        Tokenize a field name into normalized tokens.
        
        Process:
        1. Normalize unicode
        2. Split on underscores/hyphens
        3. Split camelCase
        4. Handle consecutive capitals
        5. Split on numbers
        6. Remove special chars
        7. Lowercase
        8. Expand abbreviations
        9. Filter short tokens
        
        Args:
            field_name: The field name to tokenize
            
        Returns:
            List of normalized tokens
        """
        if not field_name:
            return []
        
        # Step 1: Normalize unicode characters
        normalized = unicodedata.normalize('NFKD', field_name)
        normalized = normalized.encode('ascii', 'ignore').decode('ascii')
        
        # Step 2: Replace underscores and hyphens with spaces
        text = re.sub(r'[_\-]', ' ', normalized)
        
        # Step 3: Split camelCase - insert space between lowercase and uppercase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Step 4: Handle consecutive capitals (e.g., HTTPSConnection -> HTTPS Connection)
        text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
        
        # Step 5: Split on numbers
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        
        # Step 6: Remove special characters (keep only alphanumeric and spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Split on whitespace and filter
        tokens = text.split()
        
        # Step 7: Convert to lowercase
        tokens = [token.lower() for token in tokens]
        
        # Step 8: Expand abbreviations
        tokens = [self.expander.expand(token) for token in tokens]
        
        # Step 9: Filter out very short tokens (but keep numbers)
        tokens = [token for token in tokens if len(token) >= 2 or token.isdigit()]
        
        # Remove empty strings
        tokens = [token for token in tokens if token]
        
        logger.debug(f"Tokenized '{field_name}' -> {tokens}")
        return tokens
    
    def tokenize_with_metadata(self, field_name: str) -> Dict:
        """
        Tokenize a field name and return rich metadata.
        
        Args:
            field_name: The field name to tokenize
            
        Returns:
            Dict containing:
                - original: Original field name
                - tokens: List of tokens
                - token_set: Set of tokens for fast operations
                - expanded_abbreviations: List of (original, expanded) tuples
        """
        if not field_name:
            return {
                'original': field_name,
                'tokens': [],
                'token_set': set(),
                'expanded_abbreviations': []
            }
        
        # Track expansions during tokenization
        expanded_abbrevs = []
        
        # Get tokens
        tokens = self.tokenize(field_name)
        
        # Identify which tokens were expanded
        # Re-tokenize without expansion to compare
        temp_expander = self.expander
        self.expander = AbbreviationExpander(custom_abbreviations={})
        raw_tokens = self.tokenize(field_name)
        self.expander = temp_expander
        
        # Find expansions
        for raw, expanded in zip(raw_tokens, tokens):
            if raw != expanded:
                expanded_abbrevs.append((raw, expanded))
        
        metadata = {
            'original': field_name,
            'tokens': tokens,
            'token_set': set(tokens),
            'expanded_abbreviations': expanded_abbrevs
        }
        
        logger.debug(f"Tokenized with metadata: {field_name} -> {len(tokens)} tokens")
        return metadata

class TokenSimilarityCalculator:
    """
    Calculates multiple similarity metrics between tokenized field names.
    Combines Jaccard, Cosine, and Weighted similarities for robust matching.
    """
    
    # Default token weights based on importance
    DEFAULT_WEIGHTS = {
        # Entity types (high importance) - 3.0
        'opportunity': 3.0,
        'contact': 3.0,
        'account': 3.0,
        'lead': 3.0,
        'campaign': 3.0,
        'case': 3.0,
        'deal': 3.0,
        'company': 3.0,
        'societe': 3.0,
        
        # Attributes (medium importance) - 2.0
        'email': 2.0,
        'phone': 2.0,
        'telephone': 2.0,
        'address': 2.0,
        'name': 2.0,
        'title': 2.0,
        'description': 2.0,
        'amount': 2.0,
        'price': 2.0,
        'date': 2.0,
        
        # Modifiers (medium-low importance) - 1.5
        'first': 1.5,
        'last': 1.5,
        'primary': 1.5,
        'secondary': 1.5,
        'billing': 1.5,
        'shipping': 1.5,
        'home': 1.5,
        'work': 1.5,
        'mobile': 1.5,
        
        # Generic terms (low importance) - 1.0
        'identifier': 1.0,
        'id': 1.0,
        'number': 1.0,
        'code': 1.0,
        'flag': 1.0,
        'time': 1.0,
        'value': 1.0,
        'type': 1.0,
        'status': 1.0,
        'field': 1.0,
        
        # Default weight for unknown tokens
        'default': 1.0
    }
    
    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        """
        Initialize calculator with default and optional custom weights.
        
        Args:
            custom_weights: Optional dict of custom token weights
        """
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if custom_weights:
            self.weights.update(custom_weights)
        logger.debug(f"Initialized TokenSimilarityCalculator with {len(self.weights)} weighted tokens")
    
    def jaccard_similarity(self, tokens1: Set[str], tokens2: Set[str]) -> float:
        """
        Calculate Jaccard similarity: |intersection| / |union|
        
        Args:
            tokens1: First set of tokens
            tokens2: Second set of tokens
            
        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        if not union:
            return 0.0
        
        similarity = len(intersection) / len(union)
        logger.debug(f"Jaccard similarity: {similarity:.3f} (intersection={len(intersection)}, union={len(union)})")
        return similarity
    
    def cosine_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Calculate cosine similarity using bag-of-words vectors.
        
        Args:
            tokens1: First list of tokens
            tokens2: Second list of tokens
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not tokens1 or not tokens2:
            return 0.0
        
        # Create frequency vectors
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        
        # Get all unique tokens
        all_tokens = set(freq1.keys()) | set(freq2.keys())
        
        # Build vectors
        vec1 = [freq1.get(token, 0) for token in all_tokens]
        vec2 = [freq2.get(token, 0) for token in all_tokens]
        
        # Calculate dot product and magnitudes
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(v * v for v in vec1))
        magnitude2 = math.sqrt(sum(v * v for v in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        logger.debug(f"Cosine similarity: {similarity:.3f}")
        return similarity
    
    def weighted_token_similarity(
        self,
        tokens1: List[str],
        tokens2: List[str],
        token_weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate weighted token similarity based on token importance.
        
        Args:
            tokens1: First list of tokens
            tokens2: Second list of tokens
            token_weights: Optional custom token weights
            
        Returns:
            Weighted similarity score (0.0 to 1.0)
        """
        if not tokens1 or not tokens2:
            return 0.0
        
        weights = token_weights or self.weights
        default_weight = weights.get('default', 1.0)
        
        # Convert to sets for intersection/union
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        # Calculate weighted intersection
        intersection = set1 & set2
        weighted_intersection = sum(weights.get(token, default_weight) for token in intersection)
        
        # Calculate weighted union
        union = set1 | set2
        weighted_union = sum(weights.get(token, default_weight) for token in union)
        
        if weighted_union == 0:
            return 0.0
        
        similarity = weighted_intersection / weighted_union
        logger.debug(f"Weighted similarity: {similarity:.3f} (weighted_int={weighted_intersection:.1f}, weighted_union={weighted_union:.1f})")
        return similarity
    
    def calculate_all_similarities(
        self,
        tokens1_meta: Dict,
        tokens2_meta: Dict,
        token_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate all similarity metrics and return combined score.
        
        Args:
            tokens1_meta: Metadata dict for first field (from tokenize_with_metadata)
            tokens2_meta: Metadata dict for second field (from tokenize_with_metadata)
            token_weights: Optional custom token weights
            
        Returns:
            Dict with all similarity scores:
                - jaccard: Jaccard similarity
                - cosine: Cosine similarity
                - weighted: Weighted token similarity
                - combined: Weighted average of all three
        """
        tokens1 = tokens1_meta['tokens']
        tokens2 = tokens2_meta['tokens']
        set1 = tokens1_meta['token_set']
        set2 = tokens2_meta['token_set']
        
        # Calculate individual similarities
        jaccard = self.jaccard_similarity(set1, set2)
        cosine = self.cosine_similarity(tokens1, tokens2)
        weighted = self.weighted_token_similarity(tokens1, tokens2, token_weights)
        
        # Combined score: weighted average
        # Jaccard: 30%, Cosine: 30%, Weighted: 40%
        combined = (0.3 * jaccard) + (0.3 * cosine) + (0.4 * weighted)
        
        result = {
            'jaccard': jaccard,
            'cosine': cosine,
            'weighted': weighted,
            'combined': combined
        }
        
        logger.debug(f"All similarities calculated: combined={combined:.3f}")
        return result

class SemanticMatcher:
    """
    Main semantic matching engine for database field mapping.
    
    Orchestrates tokenization, similarity calculation, and confidence scoring
    to find the best matches between source and target fields.
    
    Designed to catch 60-70% of matches with high confidence, reducing the need
    for expensive LLM calls.
    """
    
    def __init__(
        self,
        high_threshold: float = 0.85,
        low_threshold: float = 0.4,
        custom_abbreviations: Optional[Dict[str, str]] = None,
        custom_token_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize semantic matcher with configurable thresholds and customizations.
        
        Args:
            high_threshold: Threshold for high confidence matches (default: 0.85)
            low_threshold: Threshold for medium confidence matches (default: 0.4)
            custom_abbreviations: Optional custom abbreviation dictionary
            custom_token_weights: Optional custom token importance weights
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
        # Initialize components
        expander = AbbreviationExpander(custom_abbreviations)
        self.tokenizer = FieldNameTokenizer(expander)
        self.similarity_calculator = TokenSimilarityCalculator(custom_token_weights)
        
        # Cache for tokenized target fields (performance optimization)
        self._target_cache: Dict[str, Dict] = {}
        
        logger.info(f"Initialized SemanticMatcher (high_threshold={high_threshold}, low_threshold={low_threshold})")
    
    def _get_confidence_level(self, combined_score: float) -> str:
        """
        Determine confidence level based on combined similarity score.
        
        Args:
            combined_score: The combined similarity score (0.0 to 1.0)
            
        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        if combined_score >= self.high_threshold:
            return 'high'
        elif combined_score >= self.low_threshold:
            return 'medium'
        else:
            return 'low'
    
    def match_field(self, source_field: str, target_fields: List[str]) -> List[Dict]:
        """
        Match a source field against multiple target fields and rank results.
        
        Args:
            source_field: The source field name to match
            target_fields: List of target field names to match against
            
        Returns:
            List of match results sorted by similarity score (descending), each containing:
                - target_field: Target field name
                - similarity_scores: Dict with jaccard, cosine, weighted, combined
                - confidence: 'high', 'medium', or 'low'
                - source_tokens: List of source tokens
                - target_tokens: List of target tokens
                - matched_tokens: Set of tokens in both
                - unmatched_source: Set of tokens only in source
                - unmatched_target: Set of tokens only in target
        """
        if not source_field or not target_fields:
            logger.warning("Empty source field or target fields list")
            return []
        
        # Tokenize source field once
        source_meta = self.tokenizer.tokenize_with_metadata(source_field)
        source_tokens = source_meta['tokens']
        source_set = source_meta['token_set']
        
        logger.debug(f"Matching source field '{source_field}' against {len(target_fields)} targets")
        
        results = []
        
        for target_field in target_fields:
            # Use cache if available, otherwise tokenize
            if target_field not in self._target_cache:
                self._target_cache[target_field] = self.tokenizer.tokenize_with_metadata(target_field)
            
            target_meta = self._target_cache[target_field]
            target_tokens = target_meta['tokens']
            target_set = target_meta['token_set']
            
            # Calculate all similarities
            similarities = self.similarity_calculator.calculate_all_similarities(
                source_meta,
                target_meta
            )
            
            # Determine confidence
            combined_score = similarities['combined']
            confidence = self._get_confidence_level(combined_score)
            
            # Calculate token overlaps
            matched_tokens = source_set & target_set
            unmatched_source = source_set - target_set
            unmatched_target = target_set - source_set
            
            result = {
                'target_field': target_field,
                'similarity_scores': similarities,
                'confidence': confidence,
                'source_tokens': source_tokens,
                'target_tokens': target_tokens,
                'matched_tokens': matched_tokens,
                'unmatched_source': unmatched_source,
                'unmatched_target': unmatched_target
            }
            
            results.append(result)
        
        # Sort by combined similarity score (descending)
        results.sort(key=lambda x: x['similarity_scores']['combined'], reverse=True)
        
        logger.info(f"Matched '{source_field}': top match = '{results[0]['target_field']}' "
                   f"(score={results[0]['similarity_scores']['combined']:.3f}, confidence={results[0]['confidence']})")
        
        return results
    
    def get_best_match(self, source_field: str, target_fields: List[str]) -> Optional[Dict]:
        """
        Get the single best match for a source field.
        
        Args:
            source_field: The source field name to match
            target_fields: List of target field names to match against
            
        Returns:
            Best match dict if confidence is high or medium, None otherwise
        """
        matches = self.match_field(source_field, target_fields)
        
        if not matches:
            return None
        
        best_match = matches[0]
        
        # Only return if confidence is acceptable
        if best_match['confidence'] in ['high', 'medium']:
            logger.info(f"Best match for '{source_field}': '{best_match['target_field']}' "
                       f"(confidence={best_match['confidence']})")
            return best_match
        else:
            logger.info(f"No acceptable match for '{source_field}' (best was low confidence)")
            return None
    
    def batch_match(
        self,
        source_fields: List[str],
        target_fields: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Efficiently match multiple source fields against target fields.
        
        Optimized by:
        - Tokenizing all target fields once
        - Reusing tokenized target fields for each source field
        
        Args:
            source_fields: List of source field names
            target_fields: List of target field names
            
        Returns:
            Dict mapping each source field to its ranked match results
        """
        if not source_fields or not target_fields:
            logger.warning("Empty source or target fields list")
            return {}
        
        logger.info(f"Batch matching {len(source_fields)} source fields against {len(target_fields)} target fields")
        
        # Pre-tokenize all target fields (fills cache)
        for target_field in target_fields:
            if target_field not in self._target_cache:
                self._target_cache[target_field] = self.tokenizer.tokenize_with_metadata(target_field)
        
        logger.debug(f"Pre-tokenized {len(target_fields)} target fields")
        
        # Match each source field
        results = {}
        for source_field in source_fields:
            results[source_field] = self.match_field(source_field, target_fields)
        
        logger.info(f"Batch matching complete: {len(results)} source fields processed")
        return results
    
    def clear_cache(self) -> None:
        """Clear the target field tokenization cache."""
        cache_size = len(self._target_cache)
        self._target_cache.clear()
        logger.debug(f"Cleared target cache ({cache_size} entries)")
    
    def get_cache_info(self) -> Dict:
        """Get information about the current cache state."""
        return {
            'cached_targets': len(self._target_cache),
            'cache_keys': list(self._target_cache.keys())
        }