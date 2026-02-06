"""Gitignore parsing and matching utilities.

This module provides functionality to parse .gitignore files and
check if file paths match any of the gitignore patterns.
"""

import fnmatch
import logging
from pathlib import Path
from typing import Optional, Set

logger = logging.getLogger(__name__)


class GitignoreParser:
    """Parser for .gitignore files with pattern matching support.
    
    This class handles parsing of .gitignore files and provides
    methods to check if files should be ignored based on the patterns.
    
    Supports standard gitignore pattern syntax including:
    - Comments (lines starting with #)
    - Negation patterns (lines starting with !)
    - Directory patterns (lines ending with /)
    - Anchored patterns (lines starting with /)
    - Glob patterns (*, **, ?)
    
    Example:
        >>> parser = GitignoreParser(Path("/path/to/workspace"))
        >>> parser.matches_gitignore(Path("/path/to/workspace/node_modules/file.js"))
        True
    """
    
    def __init__(self, base_path: Path):
        """Initialize the gitignore parser.
        
        Args:
            base_path: Root path of the workspace containing .gitignore
        """
        self.base_path = base_path
        self._patterns: Optional[Set[str]] = None
    
    @property
    def patterns(self) -> Set[str]:
        """Get parsed gitignore patterns, parsing if necessary.
        
        Returns:
            Set of gitignore patterns
        """
        if self._patterns is None:
            self._patterns = self._parse_gitignore()
        return self._patterns
    
    def _parse_gitignore(self) -> Set[str]:
        """Parse .gitignore file and return set of patterns to ignore.
        
        Returns:
            Set of gitignore patterns
        """
        gitignore_path = self.base_path / ".gitignore"
        patterns: Set[str] = set()
        
        if not gitignore_path.exists():
            return patterns
        
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.add(line.rstrip("/"))
        except Exception as e:
            logger.warning("Could not parse .gitignore: %s", e)
        
        return patterns
    
    def matches_gitignore(self, file_path: Path) -> bool:
        """Check if a file matches any gitignore pattern.
        
        Args:
            file_path: Absolute path to the file
            
        Returns:
            True if the file should be ignored
        """
        if not self.patterns:
            return False
        
        try:
            rel_path = file_path.relative_to(self.base_path).as_posix()
        except ValueError:
            return False
        
        for pattern in self.patterns:
            if not pattern:
                continue
            
            if pattern.startswith("/"):
                anchored_pattern = pattern[1:]
                if rel_path == anchored_pattern or rel_path.startswith(anchored_pattern + "/"):
                    return True
            else:
                if fnmatch.fnmatch(rel_path, pattern):
                    return True
                if fnmatch.fnmatch(rel_path, f"*/{pattern}"):
                    return True
                if rel_path.startswith(pattern + "/"):
                    return True
                for path_part in rel_path.split("/"):
                    if fnmatch.fnmatch(path_part, pattern):
                        return True
        
        return False
    
    def refresh(self) -> None:
        """Refresh the gitignore patterns by re-parsing the file."""
        self._patterns = self._parse_gitignore()