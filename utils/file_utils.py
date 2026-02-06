"""File utility functions for workspace indexing.

This module provides helper functions for file operations including
text detection, hashing, and size formatting.
"""

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_text_file(file_path: Path, sample_size: int = 8192) -> bool:
    """Check if a file is text-readable by attempting to decode it as UTF-8.
    
    This is more reliable than extension-based filtering as it handles:
    - Files without extensions
    - Files with wrong extensions
    - Binary files that happen to have text extensions
    - Various text encodings
    
    Args:
        file_path: Path to the file to check
        sample_size: Number of bytes to read for detection (default 8KB)
        
    Returns:
        True if file can be read as text, False otherwise
    """
    try:
        with open(file_path, "rb") as f:
            raw = f.read(sample_size)
        
        if not raw:
            return True
        
        raw.decode("utf-8", errors="strict")
        return True
    except (UnicodeDecodeError, IOError, OSError, PermissionError):
        return False


def compute_file_hash(file_path: Path) -> str:
    """Compute a hash of the file content for change detection.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash of the file content
    """
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.warning("Could not hash file %s: %s", file_path, e)
        return ""


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human readable string (e.g., "1.5MB", "256KB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"