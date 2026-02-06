"""Example usage of WorkspaceRagSearchTool with JSON input file.

This script reads search parameters from a JSON file and executes
the search using WorkspaceRagSearchTool.

Prerequisites:
    1. Ollama installed and running (ollama serve)
    2. nomic-embed-text model pulled (ollama pull nomic-embed-text)
    3. Virtual environment activated with dependencies installed

JSON format:
    {
        "workspace_path": ".",
        "query": "search query here",
        "path_filter": "",
        "limit": 5,
        "preview_window": 500
    }

Usage:
    cd ..
    python _examples/example.py [path_to_input.json]

Example:
    python example.py                 # Uses default example_in.json
    python example.py myinput.json    # Uses custom JSON file
"""

import os
import logging
import sys
import json
from pathlib import Path
from typing import Optional

# ChromaDB's telemetry off to prevent calls to https://us.i.posthog.com:443
os.environ["ANONYMIZED_TELEMETRY"] = "FALSE"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from workspace_rag_search_tool import WorkspaceRagSearchTool

class Colors:
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"
    DIM = "\033[2m"
    RESET = "\033[0m"

logging.basicConfig(
    level=logging.INFO,
    format= Colors.DIM + '%(asctime)s [%(levelname)s] ◦ %(name)s ◦ %(message)s' + Colors.RESET,
    handlers=[
        logging.StreamHandler()
    ]
)

logging.getLogger("httpx").setLevel(logging.WARNING)


class SearchParams:
    """Data class for search parameters."""
    def __init__(
        self,
        workspace_path: str,
        query: str,
        path_filter: str = "",
        limit: int = 5,
        preview_window: Optional[int] = None
    ):
        self.workspace_path = workspace_path
        self.query = query
        self.path_filter = path_filter
        self.limit = limit
        self.preview_window = preview_window


def parse_prompt_file(filepath: str) -> SearchParams:
    """Parse JSON input file and extract search parameters.
    
    Args:
        filepath: Path to the JSON input file
        
    Returns:
        SearchParams object with all search parameters
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        raise ValueError("JSON input must be an object")
    
    # Required fields
    workspace_path = data.get("workspace_path")
    query = data.get("query")
    
    if not workspace_path:
        raise ValueError("workspace_path is required")
    
    # Optional fields with defaults
    path_filter = data.get("path_filter", "")
    limit = data.get("limit", 5)
    preview_window = data.get("preview_window")
    
    return SearchParams(
        workspace_path=workspace_path,
        query=query or "",
        path_filter=path_filter,
        limit=limit,
        preview_window=preview_window
    )


def print_tool_call(tool_name: str, params: dict) -> None:
    """Print tool call in the specified format."""
    params_str = json.dumps(params, ensure_ascii=False)
    print(f"🛠️  {Colors.YELLOW}tool → → → ◦ [{tool_name}] ◦ {Colors.BRIGHT_YELLOW}{params_str}{Colors.RESET}")


def print_tool_response(tool_name: str, response: str) -> None:
    """Print tool response in the specified format."""
    print(f"📄 {Colors.CYAN}tool ← ← ← ◦ [{tool_name}] ◦")
    try:
        data = json.loads(response)
        pretty_data = json.dumps(data, indent=2, ensure_ascii=False)
        pretty_data = pretty_data.replace("\\n", "\n")
        print(f"{Colors.BRIGHT_CYAN}{pretty_data}{Colors.RESET}")
    except json.JSONDecodeError:
        print(response)


def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "example_in.json"
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_file}")
        print()
        print("Create a JSON file with the following format:")
        print("-" * 40)
        print(json.dumps({
            "workspace_path": ".",
            "query": "search query here",
            "path_filter": "",
            "limit": 5,
            "preview_window": 500
        }, indent=4))
        print("-" * 40)
        sys.exit(1)
    
    print("=" * 60)
    print("Workspace RAG Search Tool")
    print("=" * 60)
    print()
    
    try:
        params = parse_prompt_file(str(input_path))
    except Exception as e:
        print(f"❌ Error parsing input file: {e}")
        sys.exit(1)
    
    if not params.workspace_path:
        print("❌ Workspace path is empty in input file")
        sys.exit(1)
    
    print(f"📁 Workspace: {params.workspace_path}")
    print(f"🔍 Query: {params.query}")
    print(f"🔧 Path filter: {params.path_filter or '(none)'}")
    print(f"🔢 Limit: {params.limit}")
    print(f"📏 Preview window: {params.preview_window or '(full content)'}")
    print()
    
    print("⚙️  Initializing indexer...")
    print("   (This may take a while for the first run)")
    print()
    
    try:
        tool = WorkspaceRagSearchTool(
            params.workspace_path,
            exclude_extensions={".json"}
        )
    except Exception as e:
        print(f"❌ Error initializing tool: {e}")
        print()
        print("Make sure Ollama is running:")
        print("  ollama serve")
        print()
        print("And the embedding model is pulled:")
        print("  ollama pull nomic-embed-text")
        sys.exit(1)
    
    print("✅ Index ready!")
    print()
    
    search_params = {
        'query': params.query,
        'limit': params.limit,
        'path_filter': params.path_filter,
        'preview_window': params.preview_window
    }
    
    print_tool_call('search_workspace', search_params)
    
    results = tool.search_workspace(
        query=params.query,
        limit=params.limit,
        path_filter=params.path_filter or None,
        preview_window=params.preview_window
    )
    
    print_tool_response('search_workspace', results)


if __name__ == "__main__":
    main()
