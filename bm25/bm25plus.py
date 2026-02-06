"""BM25+ implementation with lower bound for zero-frequency terms.

BM25+ modifies standard BM25 by adding a lower bound δ (typically 1.0) to ensure
that even documents without query terms contribute minimally to the score. This
addresses a key limitation of standard BM25 where terms with f=0 contribute nothing.

BM25+ Formula:
    Score = Σ IDF(q) * TF+(f(q,d))
    
    TF+(f) = R + f * (k1 + 1) / (f + k1 * (1 - b + b * (|d|/avgdl)))
    where R = δ / (f + δ)

Standard BM25 TF:
    tf_bm25 = f * (k1 + 1) / (f + k1 * (1 - b + b * (|d|/avgdl)))

BM25+ TF (with lower bound):
    tf_bm25+ = R + tf_bm25
             = δ/(f+δ) + f * (k1 + 1) / (f + k1 * (1 - b + b * (|d|/avgdl)))

This provides:
- Non-zero scores even for documents without query terms (via lower bound δ)
- Better discrimination between relevant and non-relevant documents
- More stable scoring across varying corpus sizes

Reference:
    "Improvements to BM25 and Language Models Examined"
    https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf

Example:
    >>> corpus = ["def compute hash", "def authenticate user", "class FileHasher"]
    >>> bm25 = BM25Plus(corpus, delta=1.0)
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


class BM25Plus:
    """BM25+: BM25 with lower bound for zero-frequency terms.
    
    This implementation extends standard BM25 by adding a lower bound δ that
    ensures even documents without query terms contribute minimally to scores.
    This is particularly beneficial for:
    - Better handling of documents with partial term matches
    - More stable scoring when some query terms are absent
    - Improved discrimination between relevant and non-relevant documents
    
    Includes code-optimized tokenization handling snake_case and camelCase.
    
    Features:
        - Lower bound δ for zero-frequency terms (ensures minimal contribution)
        - Better handling of documents with partial query term matches
        - More stable scoring across varying query term presence
        - Code-optimized tokenization
        - Efficient precomputed statistics
    
    Attributes:
        corpus: List of document strings
        k1: Term frequency saturation parameter
        b: Length normalization parameter
        delta: Lower bound for zero-frequency terms (typically 1.0)
    """
    
    def __init__(
        self,
        corpus: List[str],
        k1: float = 1.2,
        b: float = 0.75,
        delta: float = 1.0
    ):
        """Initialize BM25Plus with a corpus of documents.
        
        Args:
            corpus: List of document strings to index
            k1: Term saturation parameter (default: 1.2)
            b: Length normalization parameter (default: 0.75)
            delta: Lower bound for zero-frequency terms (default: 1.0)
        """
        self.corpus = corpus
        self.N = len(corpus)
        self.k1 = k1
        self.b = b
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
            "BM25Plus initialized: %d documents, avgdl=%.2f, k1=%.2f, b=%.2f, delta=%.2f",
            self.N, self.avgdl, self.k1, self.b, self.delta
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
        """Compute BM25+ term frequency component with lower bound.
        
        Standard BM25:
            tf_bm25 = f * (k1 + 1) / (f + k1 * (1 - b + b * (|d|/avgdl)))
        
        BM25+:
            tf_bm25+ = R + tf_bm25
                     = δ/(f+δ) + f * (k1 + 1) / (f + k1 * (1 - b + b * (|d|/avgdl)))
        
        The lower bound R = δ/(f+δ) ensures:
        - When f=0: tf_bm25+ = δ/δ = 1.0 (minimum contribution)
        - When f>0: tf_bm25+ ≈ tf_bm25 + small_value (minimal impact on matches)
        
        Args:
            f: Raw term frequency in document
            doc_len: Document length (in tokens)
            
        Returns:
            BM25+ term frequency component
        """
        # Length normalization factor (same as standard BM25)
        length_norm = 1 - self.b + self.b * (doc_len / self.avgdl)
        
        # Lower bound component: R = δ / (f + δ)
        # When f=0, this equals 1.0; when f>0, it's a small positive value
        lower_bound = self.delta / (f + self.delta)
        
        # Standard BM25 TF component (when f=0, this equals 0)
        if f == 0:
            tf_standard = 0.0
        else:
            tf_standard = f * (self.k1 + 1) / (f + self.k1 * length_norm)
        
        # BM25+: sum of lower bound and standard TF
        return lower_bound + tf_standard
    
    def get_scores(self, query: str) -> np.ndarray:
        """Compute BM25+ scores for the query across all documents.
        
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
                
                # BM25+ term frequency component
                tf_component = self._compute_tf_component(f, doc_len)
                
                # BM25+ scoring: IDF * TF+
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


class BM25PlusIndex:
    """Incremental BM25+ index for dynamic document collections.
    
    Similar to BM25Plus but allows incremental document addition
    rather than requiring all documents upfront.
    """
    
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        delta: float = 1.0
    ):
        """Initialize an empty BM25+ index.
        
        Args:
            k1: Term saturation parameter (default: 1.2)
            b: Length normalization parameter (default: 0.75)
            delta: Lower bound for zero-frequency terms (default: 1.0)
        """
        self.k1 = k1
        self.b = b
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
        """Tokenize text (same as BM25Plus)."""
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
        """Compute BM25+ term frequency component with lower bound."""
        length_norm = 1 - self.b + self.b * (doc_len / self._avgdl)
        lower_bound = self.delta / (f + self.delta)
        
        if f == 0:
            tf_standard = 0.0
        else:
            tf_standard = f * (self.k1 + 1) / (f + self.k1 * length_norm)
        
        return lower_bound + tf_standard
    
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
        """Compute BM25+ scores for all documents.
        
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


