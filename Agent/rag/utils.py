"""
Utility functions for RAG pipeline.
"""
import hashlib


def stable_id(*parts, length=32):
    """
    Generate stable ID from parts using SHA256 hash.
    
    Args:
        *parts: Variable parts to combine into ID
        length: Length of resulting hash (default: 32)
        
    Returns:
        str: Hex digest of specified length
    """
    base = '|'.join(str(p) for p in parts)
    return hashlib.sha256(base.encode()).hexdigest()[:length]


def calculate_confidence_level(score):
    """
    Calculate confidence level from composite score.
    
    Args:
        score: Composite score (0.0-1.0)
        
    Returns:
        str: 'High', 'Medium', or 'Low'
    """
    if score >= 0.6:
        return 'High'
    elif score >= 0.4:
        return 'Medium'
    else:
        return 'Low'


def requires_review(best_table):
    """
    Determine if human review is needed.
    
    Args:
        best_table: Best matching table dict with composite_score
        
    Returns:
        bool: True if review required
    """
    if not best_table:
        return True
    return best_table.get('composite_score', 0) < 0.7
