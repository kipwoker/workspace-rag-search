"""Path utility functions for workspace safety.

This module provides functions for path resolution with security checks
to prevent directory traversal attacks.
"""

from pathlib import Path


class PathResolver:
    """Path resolver with security checks for workspace operations.
    
    This class provides safe path resolution to prevent directory
    traversal attacks by ensuring resolved paths stay within the
    workspace boundaries.
    
    Example:
        >>> resolver = PathResolver(Path("/workspace"))
        >>> resolver.resolve("subdir/file.txt")
        PosixPath('/workspace/subdir/file.txt')
        >>> resolver.resolve("../../../etc/passwd")
        ValueError: Path traversal detected
    """
    
    def __init__(self, base_path: Path):
        """Initialize the path resolver.
        
        Args:
            base_path: Root path of the workspace
        """
        self.base_path = base_path.resolve()
    
    def resolve(self, relative_path: str) -> Path:
        """Convert a relative path to an absolute path within the workspace.
        
        This method ensures path safety by preventing directory traversal attacks.
        
        Args:
            relative_path: Path relative to the workspace root
            
        Returns:
            Resolved absolute path
            
        Raises:
            ValueError: If the resolved path is outside the workspace
        """
        rel_path = Path(relative_path)
        if rel_path.is_absolute():
            abs_path = rel_path.resolve()
        else:
            abs_path = (self.base_path / rel_path).resolve()
        
        try:
            abs_path.relative_to(self.base_path)
        except ValueError as e:
            raise ValueError(
                f"Path traversal detected: '{relative_path}' resolves outside workspace"
            ) from e
        
        return abs_path