"""Lightweight BM25 implementation for keyword-based document scoring.

This module provides a pure-Python BM25 implementation optimized for
code search use cases. It builds an inverted index from documents and
computes BM25 scores for queries.

BM25 Formula:
    Score = Σ IDF(q) * (f(q,d) * (k1 + 1)) / (f(q,d) + k1 * (1 - b + b * (|d|/avgdl)))

Where:
    - IDF(q) = log((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
    - N = total number of documents
    - n(q) = number of documents containing query term q
    - f(q,d) = term frequency of q in document d
    - |d| = document length (in tokens)
    - avgdl = average document length
    - k1 = term saturation parameter (default: 1.5)
    - b = length normalization parameter (default: 0.75)

Example:
    >>> documents = ["def hello world", "def foo bar", "hello foo"]
    >>> bm25 = BM25(documents)
    >>> scores = bm25.score("hello world")
    >>> print(scores)  # [1.23, 0.0, 0.45]
"""

import logging
import math
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BM25:
    """BM25 scoring algorithm for keyword-based document retrieval.
    
    This implementation is optimized for code search with:
    - Efficient inverted index for fast lookups
    - Tokenization suitable for code (preserves identifiers)
    - Configurable k1 and b parameters
    - Optional document length caching
    
    Attributes:
        documents: List of document strings
        k1: Term saturation parameter (higher = more saturation)
        b: Length normalization parameter (0-1, higher = more normalization)
        epsilon: Smoothing factor for IDF (prevents negative IDF)
    """
    
    def __init__(
        self,
        documents: List[str],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """Initialize BM25 with a corpus of documents.
        
        Args:
            documents: List of document strings to index
            k1: Term saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
            epsilon: IDF smoothing factor (default: 0.25)
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.N = len(documents)
        
        # Build the inverted index
        self._build_index()
        
        logger.debug(
            "BM25 initialized: %d documents, avgdl=%.2f, k1=%.2f, b=%.2f",
            self.N, self.avgdl, self.k1, self.b
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms suitable for code search.
        
        Preserves code identifiers (snake_case, camelCase) and
        normalizes to lowercase for case-insensitive matching.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Split on non-alphanumeric characters but preserve internal structure
        # This handles: snake_case, camelCase, kebab-case, etc.
        tokens = []
        
        # Replace common code separators with spaces
        text = re.sub(r'[_\-\.]+', ' ', text)
        
        # Split camelCase (e.g., "helloWorld" -> "hello World")
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Extract alphanumeric tokens
        for token in re.findall(r'[a-z0-9]+', text):
            # Filter out very short tokens (likely not meaningful)
            if len(token) > 1:
                tokens.append(token)
        
        return tokens
    
    def _build_index(self) -> None:
        """Build the inverted index and compute document statistics."""
        # Term -> {doc_idx: frequency}
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        
        # Document lengths (in tokens)
        self.doc_lengths: List[int] = []
        self.total_length = 0
        
        for doc_idx, doc in enumerate(self.documents):
            tokens = self._tokenize(doc)
            doc_length = len(tokens)
            self.doc_lengths.append(doc_length)
            self.total_length += doc_length
            
            # Count term frequencies for this document
            term_freq: Dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            
            # Update inverted index
            for term, freq in term_freq.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_idx] = freq
        
        # Compute average document length
        self.avgdl = self.total_length / self.N if self.N > 0 else 0
        
        # Pre-compute IDF values
        self.idf_cache: Dict[str, float] = {}
    
    def _compute_idf(self, term: str) -> float:
        """Compute IDF (Inverse Document Frequency) for a term.
        
        Uses BM25's IDF formula with smoothing to prevent negative values.
        
        Args:
            term: The term to compute IDF for
            
        Returns:
            IDF value for the term
        """
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        # Number of documents containing the term
        n_q = len(self.inverted_index.get(term, {}))
        
        # BM25 IDF with smoothing
        # log((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
        if n_q == 0:
            idf = 0.0
        else:
            numerator = self.N - n_q + 0.5
            denominator = n_q + 0.5
            idf = max(self.epsilon, math.log(1 + numerator / denominator))
        
        self.idf_cache[term] = idf
        return idf
    
    def score(self, query: str) -> List[float]:
        """Compute BM25 scores for all documents given a query.
        
        Args:
            query: Search query string
            
        Returns:
            List of scores, one per document (higher = more relevant)
        """
        import math
        
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return [0.0] * self.N
        
        scores = [0.0] * self.N
        
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            
            idf = self._compute_idf(token)
            
            # Score each document containing this term
            for doc_idx, term_freq in self.inverted_index[token].items():
                doc_length = self.doc_lengths[doc_idx]
                
                # BM25 scoring formula
                # f(q,d) * (k1 + 1) / (f(q,d) + k1 * (1 - b + b * (|d|/avgdl)))
                tf_component = term_freq * (self.k1 + 1)
                length_norm = self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
                denominator = term_freq + length_norm
                
                score_contribution = idf * tf_component / denominator
                scores[doc_idx] += score_contribution
        
        return scores
    
    def score_top_k(
        self,
        query: str,
        k: int = 10,
        candidate_indices: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """Get top-k documents by BM25 score.
        
        Args:
            query: Search query string
            k: Number of top results to return
            candidate_indices: Optional list of document indices to consider
                             (if None, considers all documents)
            
        Returns:
            List of (doc_index, score) tuples, sorted by score descending
        """
        if candidate_indices is not None:
            # Score only candidate documents
            query_tokens = self._tokenize(query)
            scores = {idx: 0.0 for idx in candidate_indices}
            
            for token in query_tokens:
                if token not in self.inverted_index:
                    continue
                
                idf = self._compute_idf(token)
                
                for doc_idx, term_freq in self.inverted_index[token].items():
                    if doc_idx in scores:
                        doc_length = self.doc_lengths[doc_idx]
                        tf_component = term_freq * (self.k1 + 1)
                        length_norm = self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
                        denominator = term_freq + length_norm
                        scores[doc_idx] += idf * tf_component / denominator
            
            # Sort and return top-k
            sorted_results = sorted(
                scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        else:
            # Score all documents
            all_scores = self.score(query)
            sorted_results = sorted(
                enumerate(all_scores),
                key=lambda x: x[1],
                reverse=True
            )
        
        return sorted_results[:k]


class BM25Index:
    """Incremental BM25 index for dynamic document collections.
    
    Unlike BM25 which requires all documents upfront, BM25Index allows
    adding documents incrementally and supports efficient updates.
    
    This is useful for the RAG use case where documents are retrieved
    from ChromaDB and need BM25 scoring on-demand.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        """Initialize an empty BM25 index.
        
        Args:
            k1: Term saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
            epsilon: IDF smoothing factor (default: 0.25)
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_lengths: List[int] = []
        self.total_length = 0
        
        # Term -> {doc_idx: frequency}
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        self.idf_cache: Dict[str, float] = {}
        
        self._tokenizer = BM25._tokenize
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text (same as BM25 class)."""
        text = text.lower()
        tokens = []
        text = re.sub(r'[_\-\.]+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        for token in re.findall(r'[a-z0-9]+', text):
            if len(token) > 1:
                tokens.append(token)
        return tokens
    
    def add_document(self, doc_id: str, content: str) -> None:
        """Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            content: Document content
        """
        doc_idx = len(self.documents)
        self.documents.append(content)
        self.doc_ids.append(doc_id)
        
        tokens = self._tokenize(content)
        doc_length = len(tokens)
        self.doc_lengths.append(doc_length)
        self.total_length += doc_length
        
        # Update inverted index
        term_freq: Dict[str, int] = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        
        for term, freq in term_freq.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
            self.inverted_index[term][doc_idx] = freq
        
        # Clear IDF cache (needs recomputation)
        self.idf_cache.clear()
    
    def add_documents(self, doc_ids: List[str], contents: List[str]) -> None:
        """Add multiple documents to the index.
        
        Args:
            doc_ids: List of unique document identifiers
            contents: List of document contents
        """
        for doc_id, content in zip(doc_ids, contents):
            self.add_document(doc_id, content)
    
    def score(self, query: str) -> Dict[str, float]:
        """Compute BM25 scores for all documents given a query.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary mapping doc_id to score
        """
        import math
        
        N = len(self.documents)
        if N == 0:
            return {}
        
        avgdl = self.total_length / N
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return {doc_id: 0.0 for doc_id in self.doc_ids}
        
        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in self.doc_ids}
        
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            
            # Compute IDF
            n_q = len(self.inverted_index[token])
            numerator = N - n_q + 0.5
            denominator = n_q + 0.5
            idf = max(self.epsilon, math.log(1 + numerator / denominator))
            
            # Score documents containing this term
            for doc_idx, term_freq in self.inverted_index[token].items():
                doc_length = self.doc_lengths[doc_idx]
                doc_id = self.doc_ids[doc_idx]
                
                tf_component = term_freq * (self.k1 + 1)
                length_norm = self.k1 * (1 - self.b + self.b * (doc_length / avgdl))
                denominator = term_freq + length_norm
                
                scores[doc_id] += idf * tf_component / denominator
        
        return scores
    
    def score_top_k(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k documents by BM25 score.
        
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
        self.doc_lengths.clear()
        self.total_length = 0
        self.inverted_index.clear()
        self.idf_cache.clear()
    
    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self.documents)


