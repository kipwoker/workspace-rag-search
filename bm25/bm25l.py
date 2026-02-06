"""BM25L implementation with logarithmic term frequency normalization.

BM25L modifies standard BM25 by using a logarithmic term frequency normalization
that grows more slowly than the hyperbolic saturation in standard BM25. This
provides better handling of very frequent terms in long documents.

BM25L Formula:
    Score = Σ IDF(q) * log(1 + f(q,d) / (1 + μ))
    where μ = k1 / (1 - b + b * (|d|/avgdl))

Unlike standard BM25's hyperbolic saturation:
    tf_bm25 = f * (k1 + 1) / (f + k1 * (1 - b + b * (|d|/avgdl)))

BM25L uses logarithmic growth:
    tf_bm25l = log(1 + f / (1 + μ))

This provides:
- Better handling of very frequent terms in long documents
- Slower saturation curve for high term frequencies
- More stable scoring across varying document lengths

Reference:
    "Improvements to BM25 and Language Models Examined"
    https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf

Example:
    >>> corpus = ["def compute hash", "def authenticate user", "class FileHasher"]
    >>> bm25 = BM25L(corpus)
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


class BM25L:
    """BM25L: BM25 with logarithmic term frequency normalization.
    
    This implementation extends BM25 with a logarithmic TF component that
    grows more slowly than standard BM25's hyperbolic saturation. This is
    particularly beneficial for:
    - Long documents with frequently repeated terms
    - Scenarios requiring more stable scoring across document lengths
    - Better discrimination between moderately high and very high term frequencies
    
    Includes code-optimized tokenization handling snake_case and camelCase.
    
    Features:
        - Logarithmic term frequency normalization (slower saturation)
        - Better handling of very frequent terms in long documents
        - More stable scoring across varying document lengths
        - Code-optimized tokenization
        - Efficient precomputed statistics
    
    Attributes:
        corpus: List of document strings
        k1: Term frequency saturation parameter (controls curve steepness)
        b: Length normalization parameter (0-1, higher = more normalization)
    """
    
    def __init__(
        self,
        corpus: List[str],
        k1: float = 1.2,
        b: float = 0.75
    ):
        """Initialize BM25L with a corpus of documents.
        
        Args:
            corpus: List of document strings to index
            k1: Term saturation parameter (default: 1.2)
            b: Length normalization parameter (default: 0.75)
        """
        self.corpus = corpus
        self.N = len(corpus)
        self.k1 = k1
        self.b = b
        
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
            "BM25L initialized: %d documents, avgdl=%.2f, k1=%.2f, b=%.2f",
            self.N, self.avgdl, self.k1, self.b
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
    
    def _compute_tf_component(self, f: int, doc_len: int) -> float:
        """Compute BM25L logarithmic term frequency component.
        
        Unlike standard BM25's hyperbolic saturation:
            tf_bm25 = f * (k1 + 1) / (f + k1 * (1 - b + b * (|d|/avgdl)))
        
        BM25L uses logarithmic normalization:
            tf_bm25l = log(1 + f / (1 + μ))
            where μ = k1 / (1 - b + b * (|d|/avgdl))
        
        This provides slower saturation for high term frequencies,
        giving better handling of very frequent terms in long documents.
        
        Args:
            f: Raw term frequency in document
            doc_len: Document length (in tokens)
            
        Returns:
            BM25L term frequency component
        """
        if f == 0:
            return 0.0
        
        # Length normalization factor (same as standard BM25)
        length_norm = 1 - self.b + self.b * (doc_len / self.avgdl)
        
        # BM25L: μ parameter
        mu = self.k1 / length_norm
        
        # BM25L logarithmic TF component: log(1 + f / (1 + μ))
        tf_bm25l = math.log(1 + f / (1 + mu))
        
        return tf_bm25l
    
    def get_scores(self, query: str) -> np.ndarray:
        """Compute BM25L scores for the query across all documents.
        
        Args:
            query: Search query string
            
        Returns:
            Numpy array of scores (one per document)
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return np.zeros(self.N)
        
        scores = np.zeros(self.N)
        
        for i, (freqs, doc_len) in enumerate(zip(self.doc_freqs, self.doc_lengths)):
            score = 0.0
            
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                # Document term frequency
                f = freqs.get(token, 0)
                
                # BM25L term frequency component
                tf_component = self._compute_tf_component(f, doc_len)
                
                # BM25L scoring: IDF * TF (standard IDF, logarithmic TF)
                score += self.idf[token] * tf_component
            
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


class BM25LIndex:
    """Incremental BM25L index for dynamic document collections.
    
    Similar to BM25L but allows incremental document addition
    rather than requiring all documents upfront.
    """
    
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75
    ):
        """Initialize an empty BM25L index.
        
        Args:
            k1: Term saturation parameter (default: 1.2)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.doc_lengths: List[int] = []
        
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        
        self._avgdl: float = 0.0
        self._dirty = True  # Flag to recompute stats
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text (same as BM25L)."""
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
    
    def _compute_tf_component(self, f: int, doc_len: int) -> float:
        """Compute BM25L logarithmic term frequency component."""
        if f == 0:
            return 0.0
        
        length_norm = 1 - self.b + self.b * (doc_len / self._avgdl)
        mu = self.k1 / length_norm
        tf_bm25l = math.log(1 + f / (1 + mu))
        
        return tf_bm25l
    
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
    
    def score(self, query: str) -> Dict[str, float]:
        """Compute BM25L scores for all documents.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary mapping doc_id to score
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {doc_id: 0.0 for doc_id in self.doc_ids}
        
        self._recompute_stats()
        
        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in self.doc_ids}
        
        for i, (freqs, doc_len) in enumerate(zip(self.doc_freqs, self.doc_lengths)):
            doc_id = self.doc_ids[i]
            score = 0.0
            
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                f = freqs.get(token, 0)
                tf_component = self._compute_tf_component(f, doc_len)
                
                score += self.idf[token] * tf_component
            
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


