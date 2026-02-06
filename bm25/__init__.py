"""BM25 implementations for keyword-based document scoring.

This package provides multiple BM25 algorithm variants:
- BM25: Standard BM25 implementation (bm25_scorer)
- BM25L: Logarithmic term frequency normalization (bm25l)
- BM25Plus: With lower bound for zero-frequency terms (bm25plus)
- BM25T: Two-stage term frequency transformation (bm25t)
- BM25Adpt: Adaptive parameter tuning (bm25_adpt)

Example:
    >>> from bm25 import BM25, BM25Index
    >>> from bm25 import BM25L, BM25Plus, BM25T, BM25Adpt
    >>> 
    >>> # Standard BM25
    >>> bm25 = BM25(documents)
    >>> scores = bm25.score(query)
    >>> 
    >>> # Incremental index
    >>> index = BM25Index()
    >>> index.add_document("doc1", content)
    >>> scores = index.score(query)
    >>> 
    >>> # Factory function for creating BM25 indexes
    >>> from bm25 import create_bm25_index
    >>> index = create_bm25_index("plus")
"""

from typing import Union

from bm25.bm25_scorer import BM25, BM25Index
from bm25.bm25l import BM25L, BM25LIndex
from bm25.bm25plus import BM25Plus, BM25PlusIndex
from bm25.bm25t import BM25T, BM25TIndex
from bm25.bm25_adpt import BM25Adpt, BM25AdptIndex

__all__ = [
    # Standard BM25
    "BM25",
    "BM25Index",
    # BM25L - Logarithmic TF
    "BM25L",
    "BM25LIndex",
    # BM25Plus - With lower bound
    "BM25Plus",
    "BM25PlusIndex",
    # BM25T - Two-stage transformation
    "BM25T",
    "BM25TIndex",
    # BM25Adpt - Adaptive parameters
    "BM25Adpt",
    "BM25AdptIndex",
    # Factory function
    "create_bm25_index",
]


# Type alias for all BM25 index types
BM25IndexType = Union[
    BM25Index,
    BM25PlusIndex,
    BM25LIndex,
    BM25TIndex,
    BM25AdptIndex,
]


def create_bm25_index(
    implementation: str = "plus",
    **kwargs
) -> BM25IndexType:
    """Create a BM25 index based on the specified implementation.
    
    Factory function to instantiate the appropriate BM25 index variant
    based on the implementation name. This allows runtime selection
    of BM25 algorithms through configuration.
    
    Args:
        implementation: BM25 variant to use. Options:
            - "standard": Standard BM25 (k1=1.5, b=0.75, epsilon=0.25)
            - "plus": BM25+ with lower bound (k1=1.2, b=0.75, delta=1.0)
            - "l": BM25L logarithmic TF (k1=1.2, b=0.75)
            - "t": BM25T two-stage (k1=1.2, b=0.75, k3=1000.0)
            - "adpt": BM25-Adpt adaptive (k1_base=1.2, b_base=0.75, delta=0.9)
        **kwargs: Additional parameters to pass to the index constructor.
            Common parameters include:
            - k1/k1_base: Term saturation parameter
            - b/b_base: Length normalization parameter
            - delta: Lower bound for zero-frequency terms (plus, adpt)
            - k3: Query term frequency saturation (t only)
            - epsilon: IDF smoothing factor (standard only)
    
    Returns:
        BM25 index instance of the specified type
    
    Raises:
        ValueError: If the implementation name is not recognized
    
    Example:
        >>> # Create default BM25+ index
        >>> index = create_bm25_index("plus")
        >>> 
        >>> # Create BM25L with custom parameters
        >>> index = create_bm25_index("l", k1=1.5, b=0.8)
        >>> 
        >>> # Add documents and score
        >>> index.add_document("doc1", "content here")
        >>> scores = index.score("query")
    """
    implementation = implementation.lower().strip()
    
    if implementation == "standard":
        return BM25Index(**kwargs)
    elif implementation == "plus":
        return BM25PlusIndex(**kwargs)
    elif implementation == "l":
        return BM25LIndex(**kwargs)
    elif implementation == "t":
        return BM25TIndex(**kwargs)
    elif implementation == "adpt":
        return BM25AdptIndex(**kwargs)
    else:
        raise ValueError(
            f"Unknown BM25 implementation: '{implementation}'. "
            f"Available options: 'standard', 'plus', 'l', 't', 'adpt'"
        )
