"""BM25-Adpt implementation with adaptive parameter tuning.

BM25-Adpt is an adaptive variant of BM25 that dynamically adjusts parameters
based on query and collection characteristics. It modifies standard BM25 with:
- Query-length based k1/b adaptation
- Collection variance adaptation
- Query specificity (term rarity) adaptation
- BM25+ lower-bound for low term frequency

Reference:
    Improvements to BM25 and Language Models Examined
    https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf

Example:
    >>> corpus = ["def compute hash", "def authenticate user", "class FileHasher"]
    >>> bm25 = BM25Adpt(corpus)
    >>> scores = bm25.get_scores("compute hash")
    >>> print(scores)
"""

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class BM25Adpt:
    """BM25-Adpt: Adaptive BM25 with dynamic parameter tuning.
    
    This implementation extends BM25 with adaptive parameters that adjust
    based on query characteristics and collection statistics. Optimized
    for code search with tokenization handling snake_case and camelCase.
    
    Features:
        - Query length adaptation (short queries get lower saturation)
        - Collection variance adaptation (document length distribution)
        - Query specificity adaptation (rare terms boost k1)
        - BM25+ lower-bound for better low-TF handling
        - Code-optimized tokenization
    
    Attributes:
        corpus: List of document strings
        k1_base: Base k1 parameter for adaptation
        b_base: Base b parameter for adaptation
        delta: BM25+ delta parameter for low-TF bound
    """
    
    def __init__(
        self,
        corpus: List[str],
        k1_base: float = 1.2,
        b_base: float = 0.75,
        delta: float = 0.9
    ):
        """Initialize BM25-Adpt with a corpus of documents.
        
        Args:
            corpus: List of document strings to index
            k1_base: Base term saturation parameter (default: 1.2)
            b_base: Base length normalization parameter (default: 0.75)
            delta: BM25+ delta parameter for low-TF bound (default: 0.9)
        """
        self.corpus = corpus
        self.N = len(corpus)
        self.k1_base = k1_base
        self.b_base = b_base
        self.delta = delta
        
        # Document statistics
        self.doc_freqs: List[Dict[str, int]] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        
        # Collection statistics
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        
        # Precompute all statistics
        self._initialize()
        
        logger.debug(
            "BM25-Adpt initialized: %d documents, avgdl=%.2f, k1_base=%.2f, b_base=%.2f",
            self.N, self.avgdl, self.k1_base, self.b_base
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms suitable for code search.
        
        Handles snake_case, camelCase, and kebab-case identifiers
        by splitting appropriately.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        text = text.lower()
        
        # Replace common code separators with spaces
        text = re.sub(r'[_\-\.]+', ' ', text)
        
        # Split camelCase (e.g., "helloWorld" -> "hello World")
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Extract alphanumeric tokens
        tokens = []
        for token in re.findall(r'[a-z0-9]+', text):
            if len(token) > 1:
                tokens.append(token)
        
        return tokens
    
    def _initialize(self) -> None:
        """Precompute statistics for the corpus."""
        # Tokenize all documents
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        
        # Compute document frequencies, lengths, and term frequencies
        for doc_tokens in tokenized_corpus:
            freqs = Counter(doc_tokens)
            doc_len = len(doc_tokens)
            self.doc_freqs.append(freqs)
            self.doc_lengths.append(doc_len)
            
            for term in freqs:
                self.df[term] += 1
        
        self.avgdl = float(np.mean(self.doc_lengths)) if self.doc_lengths else 0.0
        
        # Compute IDF using standard BM25 formula
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
    
    def _adapt_parameters(self, query_tokens: List[str]) -> Tuple[float, float]:
        """Adapt k1 and b parameters based on query characteristics.
        
        Adaptation strategies:
        1. Query length: Shorter queries get lower saturation (lower k1)
        2. Collection variance: High variance increases length normalization
        3. Query specificity: Rare terms boost k1 for better discrimination
        
        Args:
            query_tokens: Tokenized query
            
        Returns:
            Tuple of (adapted_k1, adapted_b)
        """
        query_len = len(query_tokens)
        
        # Query length adaptation (shorter queries -> lower saturation)
        if query_len <= 2:
            k1 = 0.8 * self.k1_base
            b = 0.5 * self.b_base
        elif query_len <= 5:
            k1 = 1.2 * self.k1_base
            b = 0.75 * self.b_base
        else:
            k1 = self.k1_base
            b = self.b_base
        
        # Collection adaptation: scale by document length variance
        if len(self.doc_lengths) > 1:
            length_var = float(np.var(self.doc_lengths))
            b *= min(1.0, 1.0 + 0.5 * length_var / (self.avgdl + 1))
        
        # Query specificity adaptation (rare terms boost k1)
        unique_terms = set(query_tokens)
        if unique_terms:
            rarity_score = sum(
                1 for t in unique_terms 
                if self.df.get(t, self.N) < self.N * 0.1
            ) / len(unique_terms)
            k1 *= (0.8 + 0.4 * rarity_score)
        
        # Clamp to valid ranges
        return max(0.1, min(2.5, k1)), max(0.1, min(1.0, b))
    
    def get_scores(self, query: str) -> np.ndarray:
        """Compute BM25-Adpt scores for the query across all documents.
        
        Args:
            query: Search query string
            
        Returns:
            Numpy array of scores (one per document)
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return np.zeros(self.N)
        
        # Adapt parameters for this specific query
        k1, b = self._adapt_parameters(query_tokens)
        
        logger.debug("Adapted parameters: k1=%.3f, b=%.3f for query '%s'", k1, b, query[:50])
        
        scores = np.zeros(self.N)
        
        for i, (freqs, doc_len) in enumerate(zip(self.doc_freqs, self.doc_lengths)):
            score = 0.0
            
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                f = freqs.get(token, 0)
                
                if f > 0:
                    # Standard BM25 term frequency saturation
                    tf_sat = f * (k1 + 1) / (
                        f + k1 * (1 - b + b * (doc_len / self.avgdl))
                    )
                else:
                    # BM25+ lower-bound for low/zero TF
                    tf_sat = self.delta / (self.delta + 1)
                
                score += self.idf[token] * tf_sat
            
            scores[i] = score
        
        return scores
    
    def get_top_k(
        self, 
        query: str, 
        k: int = 10,
        return_scores: bool = True
    ) -> Union[List[int], List[Tuple[int, float]]]:
        """Get top-k documents for a query.
        
        Args:
            query: Search query string
            k: Number of top results to return
            return_scores: If True, return (doc_idx, score) tuples
            
        Returns:
            List of document indices or (doc_idx, score) tuples
        """
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:k]
        
        if return_scores:
            return [(int(idx), float(scores[idx])) for idx in top_indices]
        return [int(idx) for idx in top_indices]


class BM25AdptIndex:
    """Incremental BM25-Adpt index for dynamic document collections.
    
    Similar to BM25Adpt but allows incremental document addition
    rather than requiring all documents upfront.
    """
    
    def __init__(
        self,
        k1_base: float = 1.2,
        b_base: float = 0.75,
        delta: float = 0.9
    ):
        """Initialize an empty BM25-Adpt index.
        
        Args:
            k1_base: Base term saturation parameter (default: 1.2)
            b_base: Base length normalization parameter (default: 0.75)
            delta: BM25+ delta parameter (default: 0.9)
        """
        self.k1_base = k1_base
        self.b_base = b_base
        self.delta = delta
        
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.doc_lengths: List[int] = []
        
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        
        self._avgdl: float = 0.0
        self._dirty = True  # Flag to recompute stats
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text (same as BM25Adpt)."""
        text = text.lower()
        text = re.sub(r'[_\-\.]+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        tokens = [t for t in re.findall(r'[a-z0-9]+', text) if len(t) > 1]
        return tokens
    
    def _recompute_stats(self) -> None:
        """Recompute collection statistics."""
        if not self._dirty:
            return
        
        N = len(self.documents)
        if N == 0:
            self._avgdl = 0.0
            self.idf = {}
            self._dirty = False
            return
        
        # Recompute document frequencies
        self.df = defaultdict(int)
        for freqs in self.doc_freqs:
            for term in freqs:
                self.df[term] += 1
        
        # Recompute IDF
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
        
        # Recompute average document length
        self._avgdl = float(np.mean(self.doc_lengths)) if self.doc_lengths else 0.0
        
        self._dirty = False
    
    def add_document(self, doc_id: str, content: str) -> None:
        """Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            content: Document content
        """
        tokens = self._tokenize(content)
        freqs = Counter(tokens)
        doc_len = len(tokens)
        
        self.documents.append(content)
        self.doc_ids.append(doc_id)
        self.doc_freqs.append(freqs)
        self.doc_lengths.append(doc_len)
        
        self._dirty = True
    
    def add_documents(self, doc_ids: List[str], contents: List[str]) -> None:
        """Add multiple documents to the index.
        
        Args:
            doc_ids: List of unique document identifiers
            contents: List of document contents
        """
        for doc_id, content in zip(doc_ids, contents):
            self.add_document(doc_id, content)
    
    def _adapt_parameters(self, query_tokens: List[str]) -> Tuple[float, float]:
        """Adapt k1 and b parameters (same logic as BM25Adpt)."""
        self._recompute_stats()
        
        N = len(self.documents)
        query_len = len(query_tokens)
        
        # Query length adaptation
        if query_len <= 2:
            k1 = 0.8 * self.k1_base
            b = 0.5 * self.b_base
        elif query_len <= 5:
            k1 = 1.2 * self.k1_base
            b = 0.75 * self.b_base
        else:
            k1 = self.k1_base
            b = self.b_base
        
        # Collection variance adaptation
        if len(self.doc_lengths) > 1:
            length_var = float(np.var(self.doc_lengths))
            b *= min(1.0, 1.0 + 0.5 * length_var / (self._avgdl + 1))
        
        # Query specificity adaptation
        unique_terms = set(query_tokens)
        if unique_terms:
            rarity_score = sum(
                1 for t in unique_terms 
                if self.df.get(t, N) < N * 0.1
            ) / len(unique_terms)
            k1 *= (0.8 + 0.4 * rarity_score)
        
        return max(0.1, min(2.5, k1)), max(0.1, min(1.0, b))
    
    def score(self, query: str) -> Dict[str, float]:
        """Compute BM25-Adpt scores for all documents.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary mapping doc_id to score
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {doc_id: 0.0 for doc_id in self.doc_ids}
        
        self._recompute_stats()
        
        k1, b = self._adapt_parameters(query_tokens)
        N = len(self.documents)
        
        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in self.doc_ids}
        
        for i, (freqs, doc_len) in enumerate(zip(self.doc_freqs, self.doc_lengths)):
            doc_id = self.doc_ids[i]
            score = 0.0
            
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                f = freqs.get(token, 0)
                
                if f > 0:
                    tf_sat = f * (k1 + 1) / (
                        f + k1 * (1 - b + b * (doc_len / self._avgdl))
                    )
                else:
                    tf_sat = self.delta / (self.delta + 1)
                
                score += self.idf[token] * tf_sat
            
            scores[doc_id] = score
        
        return scores
    
    def score_top_k(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k documents by score.
        
        Args:
            query: Search query string
            k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        scores = self.score(query)
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
    
    def clear(self) -> None:
        """Clear all documents from the index."""
        self.documents.clear()
        self.doc_ids.clear()
        self.doc_freqs.clear()
        self.doc_lengths.clear()
        self.df.clear()
        self.idf.clear()
        self._avgdl = 0.0
        self._dirty = False
    
    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self.documents)


