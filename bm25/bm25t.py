"""BM25T implementation with two-stage term frequency transformation.

BM25T modifies standard BM25 by transforming term frequencies using a two-stage
saturation function that better handles both low and high frequency terms.

The first stage uses linear growth (f) for f <= 1, providing better handling
of rare terms. The second stage uses hyperbolic saturation for f > 1.

Reference:
    "Improvements to BM25 and Language Models Examined"
    https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf

Example:
    >>> corpus = ["def compute hash", "def authenticate user", "class FileHasher"]
    >>> bm25 = BM25T(corpus)
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


class BM25T:
    """BM25T: BM25 with two-stage term frequency transformation.
    
    This implementation extends BM25 with a two-stage saturation function:
    - Stage 1: Linear growth (f) for f <= 1 (better handling of rare terms)
    - Stage 2: Hyperbolic saturation for f > 1
    
    Includes query term frequency normalization (k3 parameter) and is
    optimized for code search with tokenization handling snake_case and camelCase.
    
    Features:
        - Two-stage TF transformation for better term frequency handling
        - Query TF normalization with k3 parameter
        - Code-optimized tokenization
        - Efficient precomputed statistics
    
    Attributes:
        corpus: List of document strings
        k1: Term frequency saturation parameter
        b: Length normalization parameter
        k3: Query term frequency saturation parameter
    """
    
    def __init__(
        self,
        corpus: List[str],
        k1: float = 1.2,
        b: float = 0.75,
        k3: float = 1000.0
    ):
        """Initialize BM25T with a corpus of documents.
        
        Args:
            corpus: List of document strings to index
            k1: Term frequency saturation parameter (default: 1.2)
            b: Length normalization parameter (default: 0.75)
            k3: Query term frequency saturation parameter (default: 1000.0)
        """
        self.corpus = corpus
        self.N = len(corpus)
        self.k1 = k1
        self.b = b
        self.k3 = k3
        
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
            "BM25T initialized: %d documents, avgdl=%.2f, k1=%.2f, b=%.2f, k3=%.2f",
            self.N, self.avgdl, self.k1, self.b, self.k3
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
    
    def _tf_transform(self, f: int, doc_len: int) -> float:
        """BM25T two-stage term frequency transformation.
        
        First stage: Linear growth up to 1.0 for f <= 1
        Second stage: Hyperbolic saturation for f > 1
        
        Args:
            f: Raw term frequency in document
            doc_len: Document length (in tokens)
            
        Returns:
            Transformed term frequency
        """
        # First stage: linear up to f=1
        if f <= 1:
            return float(f)
        
        # Second stage: saturation function
        # denominator = k1 * (1 - b + b * (doc_len / avgdl))
        denom = self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
        return 1.0 + (f - 1) / (1 + denom)
    
    def _query_tf_transform(self, qf: int) -> float:
        """Query term frequency transformation.
        
        Args:
            qf: Query term frequency
            
        Returns:
            Transformed query term frequency
        """
        return qf * (self.k3 + 1) / (qf + self.k3)
    
    def get_scores(self, query: str) -> np.ndarray:
        """Compute BM25T scores for the query across all documents.
        
        Args:
            query: Search query string
            
        Returns:
            Numpy array of scores (one per document)
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return np.zeros(self.N)
        
        # Query term frequencies
        qf = Counter(query_tokens)
        
        scores = np.zeros(self.N)
        
        for i, (freqs, doc_len) in enumerate(zip(self.doc_freqs, self.doc_lengths)):
            score = 0.0
            
            for token, qfreq in qf.items():
                if token not in self.idf:
                    continue
                
                # Document term frequency
                f = freqs.get(token, 0)
                
                # BM25T term frequency transformation
                tf_doc = self._tf_transform(f, doc_len)
                
                # Query term frequency transformation
                tf_query = self._query_tf_transform(qfreq)
                
                # BM25T scoring: IDF * tf_doc * tf_query
                score += self.idf[token] * tf_doc * tf_query
            
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


class BM25TIndex:
    """Incremental BM25T index for dynamic document collections.
    
    Similar to BM25T but allows incremental document addition
    rather than requiring all documents upfront.
    """
    
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        k3: float = 1000.0
    ):
        """Initialize an empty BM25T index.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.2)
            b: Length normalization parameter (default: 0.75)
            k3: Query term frequency saturation parameter (default: 1000.0)
        """
        self.k1 = k1
        self.b = b
        self.k3 = k3
        
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.doc_lengths: List[int] = []
        
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}
        
        self._avgdl: float = 0.0
        self._dirty = True  # Flag to recompute stats
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text (same as BM25T)."""
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
    
    def _tf_transform(self, f: int, doc_len: int) -> float:
        """BM25T two-stage term frequency transformation."""
        if f <= 1:
            return float(f)
        
        denom = self.k1 * (1 - self.b + self.b * (doc_len / self._avgdl))
        return 1.0 + (f - 1) / (1 + denom)
    
    def _query_tf_transform(self, qf: int) -> float:
        """Query term frequency transformation."""
        return qf * (self.k3 + 1) / (qf + self.k3)
    
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
        """Compute BM25T scores for all documents.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary mapping doc_id to score
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {doc_id: 0.0 for doc_id in self.doc_ids}
        
        self._recompute_stats()
        
        qf = Counter(query_tokens)
        
        scores: Dict[str, float] = {doc_id: 0.0 for doc_id in self.doc_ids}
        
        for i, (freqs, doc_len) in enumerate(zip(self.doc_freqs, self.doc_lengths)):
            doc_id = self.doc_ids[i]
            score = 0.0
            
            for token, qfreq in qf.items():
                if token not in self.idf:
                    continue
                
                f = freqs.get(token, 0)
                tf_doc = self._tf_transform(f, doc_len)
                tf_query = self._query_tf_transform(qfreq)
                
                score += self.idf[token] * tf_doc * tf_query
            
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


