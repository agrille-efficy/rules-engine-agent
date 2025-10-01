import hashlib
from typing import Dict, List, Any
from langchain_openai import OpenAIEmbeddings

class EmbeddingCache:
    """Caching layer for OpenAI embeddings to reduce API calls"""
    
    def __init__(self, encoder: OpenAIEmbeddings):
        self.encoder = encoder
        self._cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a unique cache key for the text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding with caching"""
        cache_key = self._get_cache_key(text)
        
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        # Cache miss - call API
        self._cache_misses += 1
        embedding = self.encoder.embed_query(text)
        self._cache[cache_key] = embedding
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embed documents with caching"""
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
                self._cache_hits += 1
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
                self._cache_misses += 1
        
        # Batch process uncached texts
        if uncached_texts:
            new_embeddings = self.encoder.embed_documents(uncached_texts)
            
            # Update cache and results
            for idx, embedding in zip(uncached_indices, new_embeddings):
                cache_key = self._get_cache_key(texts[idx])
                self._cache[cache_key] = embedding
                results[idx] = embedding
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        
        return {
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percent': round(hit_rate, 2)
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0