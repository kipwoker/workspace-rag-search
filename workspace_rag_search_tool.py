import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import chromadb
from chromadb.api.models.Collection import Collection

from async_ollama_embedding_function import AsyncOllamaEmbeddingFunction
from bm25 import create_bm25_index, BM25IndexType
from ollama_config import OLLAMA_BASE_URL, RAG_CONFIG_DEFAULT, RAGConfig
from rrf import ReciprocalRankFusion
from utils import is_text_file, compute_file_hash, format_size, GitignoreParser, PathResolver

logger = logging.getLogger(__name__)


class WorkspaceRagSearchTool:
    """Semantic search tool for workspace code using RAG (Retrieval-Augmented Generation).
    
    This tool indexes workspace files using ChromaDB with Ollama embeddings,
    supporting hybrid search combining semantic similarity and BM25 keyword scoring
    using Reciprocal Rank Fusion (RRF).
    
    Files are automatically filtered using:
    - .gitignore rules (built-in parsing)
    - Size limits (configurable)
    - Text detection (binary files are skipped)
    - Optional user-provided extension filters
    
    Features:
        - Incremental indexing (only processes changed files)
        - Content-based file detection (no extension hardcoding)
        - Hybrid search with RRF (combines semantic + BM25 rankings)
        - Path-based filtering for search results
        - Configurable chunking with overlap for better context
        - Code-optimized BM25 with camelCase/snake_case tokenization
    
    Example:
        >>> tool = WorkspaceRagSearchTool("/path/to/workspace")
        >>> results = tool.search_workspace("authentication middleware")
        >>> print(results)
    """

    def __init__(
        self,
        base_path: str,
        rag_config: Optional[RAGConfig] = None,
        ollama_base_url: str = OLLAMA_BASE_URL,
        include_extensions: Optional[Set[str]] = None,
        exclude_extensions: Optional[Set[str]] = None,
        force_reindex: bool = False,
    ):
        """Initialize the workspace RAG search tool.
        
        Files are automatically discovered from the workspace, respecting .gitignore.
        Binary files are automatically skipped based on content detection.
        
        Args:
            base_path: Root path of the workspace to index
            rag_config: RAG configuration (uses default if not provided)
            ollama_base_url: URL for the Ollama embedding server
            include_extensions: Optional set of extensions to include (e.g., {".py", ".js"}).
                              If provided, only these extensions are indexed.
            exclude_extensions: Optional set of extensions to exclude (e.g., {".min.js"}).
                              Applied after include_extensions filter.
            force_reindex: If True, delete and recreate the vector store
            
        Raises:
            ValueError: If base_path doesn't exist or is not a directory
        """
        self.base_path = Path(base_path).resolve()
        self.rag_config = rag_config or RAG_CONFIG_DEFAULT
        self.ollama_base_url = ollama_base_url
        self.include_extensions = include_extensions
        self.exclude_extensions = exclude_extensions
        self._collection: Optional[Collection] = None
        self._gitignore_parser: Optional[GitignoreParser] = None
        self._path_resolver: Optional[PathResolver] = None
        
        self._validate_config()
        
        if not self.base_path.exists():
            raise ValueError(f"Base path does not exist: {base_path}")
        if not self.base_path.is_dir():
            raise ValueError(f"Base path is not a directory: {base_path}")
        
        logger.info("Initializing workspace search index for: %s", self.base_path)
        
        self._gitignore_parser = GitignoreParser(self.base_path)
        self._path_resolver = PathResolver(self.base_path)
        
        if force_reindex:
            self._delete_vector_store()
        
        self._get_or_create_collection()
        logger.info("Workspace index ready!")

    def _validate_config(self) -> None:
        """Validate RAG configuration parameters.
        
        Raises:
            ValueError: If configuration parameters are invalid
        """
        if self.rag_config.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.rag_config.chunk_size}")
        if self.rag_config.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.rag_config.chunk_overlap}")
        if self.rag_config.chunk_overlap >= self.rag_config.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.rag_config.chunk_overlap}) must be less than "
                f"chunk_size ({self.rag_config.chunk_size})"
            )
        if self.rag_config.bytes_limit <= 0:
            raise ValueError(f"bytes_limit must be positive, got {self.rag_config.bytes_limit}")

    def _get_workspace_files(self) -> List[Path]:
        """Get all files from the workspace, respecting .gitignore.
        
        Uses Path.rglob for efficient recursive file discovery while
        filtering out ignored directories and files.
        
        Returns:
            List of file paths (sorted)
        """
        files: List[Path] = []
        ignored_count = 0
        
        skip_dirs = {".git"}
        
        for path in self.base_path.rglob("*"):
            if path.is_dir():
                continue
            
            try:
                rel_parts = path.relative_to(self.base_path).parts
                if any(part in skip_dirs for part in rel_parts):
                    continue
            except ValueError:
                continue
            
            if self._gitignore_parser and self._gitignore_parser.matches_gitignore(path):
                ignored_count += 1
                continue
            
            files.append(path)
        
        logger.debug("Found %d files, %d ignored by .gitignore", len(files), ignored_count)
        return sorted(files)

    def _delete_vector_store(self) -> None:
        """Delete the vector store directory and all associated files."""
        vector_store_path = Path(self.rag_config.vector_store_path)
        
        logger.debug("Deleting vector store: %s", vector_store_path)
        
        if vector_store_path.exists():
            shutil.rmtree(vector_store_path)
            logger.debug("Vector store deleted")
        
        for path in Path(".").glob("*.chroma*"):
            if path.is_dir():
                shutil.rmtree(path)
                logger.debug("Deleted orphaned directory: %s", path)

    def _should_index_file(self, file_path: Path) -> bool:
        """Check if a file should be indexed.
        
        Filtering order:
        1. User-provided include_extensions (if specified)
        2. User-provided exclude_extensions (if specified)
        3. Content-based text detection (skip binary files)
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file should be indexed, False otherwise
        """
        suffix = file_path.suffix.lower()
        
        if self.include_extensions is not None:
            if suffix not in self.include_extensions:
                logger.debug("Skipping file (not in include_extensions): %s", file_path)
                return False
        
        if self.exclude_extensions is not None:
            if suffix in self.exclude_extensions:
                logger.debug("Skipping file (in exclude_extensions): %s", file_path)
                return False
        
        if not is_text_file(file_path):
            logger.debug("Skipping binary file: %s", file_path)
            return False
        
        return True

    def _chunk_content(self, content: str, file_path: Path) -> List[Tuple[str, int]]:
        """Split content into chunks with overlap.
        
        Args:
            content: File content to chunk
            file_path: Path to the file (for context)
            
        Returns:
            List of tuples (chunk_text, chunk_index)
        """
        content_len = len(content)
        chunk_size = self.rag_config.chunk_size
        chunk_overlap = self.rag_config.chunk_overlap
        rel_path = file_path.relative_to(self.base_path).as_posix()
        
        chunks: List[Tuple[str, int]] = []
        
        if content_len <= chunk_size:
            if content.strip():
                chunk_with_context = f"[File: {rel_path}]\n\n{content}"
                chunks.append((chunk_with_context, 0))
        else:
            for i in range(0, content_len, chunk_size - chunk_overlap):
                chunk = content[i:i + chunk_size]
                if chunk.strip():
                    chunk_with_context = f"[File: {rel_path}]\n\n{chunk}"
                    chunks.append((chunk_with_context, i // (chunk_size - chunk_overlap)))
        
        return chunks

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create a new one with incremental indexing.
        
        Uses AsyncOllamaEmbeddingFunction for faster concurrent embedding
        processing during indexing.
        
        Returns:
            ChromaDB collection instance
        """
        if self._collection is not None:
            return self._collection
        
        vector_store_path = Path(self.rag_config.vector_store_path)
        vector_store_path.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(path=str(vector_store_path))
        collection_name = "workspace_code_index"
        
        existing_collections = client.list_collections()
        collection_exists = any(c.name == collection_name for c in existing_collections)

        ollama_ef = AsyncOllamaEmbeddingFunction(
            model_name=self.rag_config.embedding_model,
            url=self.ollama_base_url,
            max_concurrent=self.rag_config.max_concurrent_requests,
            batch_size=self.rag_config.embedding_batch_size
        )
        
        if collection_exists:
            collection = client.get_collection(name=collection_name)
            logger.info("Using existing collection: %s", collection_name)

            collection._embedding_function = ollama_ef
        else:
            collection = client.create_collection(
                name=collection_name,
                embedding_function=ollama_ef,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Created new collection: %s", collection_name)
        
        self._index_files(collection)
        
        self._collection = collection
        return self._collection

    def _index_files(self, collection: Collection) -> None:
        """Index workspace files with incremental updates.
        
        Args:
            collection: ChromaDB collection to add documents to
        """
        all_files = self._get_workspace_files()
        
        files_to_index: List[Path] = []
        binary_skipped = 0
        
        for file_path in all_files:
            if self._should_index_file(file_path):
                files_to_index.append(file_path)
            else:
                binary_skipped += 1
        
        logger.info(
            "Found %d files to index (%d from workspace, %d binary/non-text skipped)",
            len(files_to_index), len(all_files), binary_skipped
        )
        
        existing_ids: Set[str] = set()
        try:
            existing_data = collection.get()
            existing_ids = set(existing_data.get("ids", []))
        except Exception as e:
            logger.warning("Could not get existing document IDs: %s", e)
        
        new_docs: List[str] = []
        new_ids: List[str] = []
        new_metadatas: List[Dict[str, Any]] = []
        current_file_ids: Set[str] = set()
        skipped = 0
        doc_counter = 0
        
        bytes_limit = self.rag_config.bytes_limit
        total_files = len(files_to_index)
        next_progress_threshold = 10
        
        total_bytes = sum(f.stat().st_size for f in files_to_index)
        processed_bytes = 0
        
        for file_idx, file_path in enumerate(files_to_index, 1):
            file_size = file_path.stat().st_size
            processed_bytes += file_size
            
            progress_percent = (file_idx / total_files) * 100
            if progress_percent >= next_progress_threshold:
                logger.info("Indexing progress: %d%% (%d/%d files, %s/%s)", 
                           int(progress_percent), file_idx, total_files,
                           format_size(processed_bytes), format_size(total_bytes))
                next_progress_threshold += 10
            
            try:
                if file_size > bytes_limit:
                    logger.debug("Skipping large file %s (%d bytes)", file_path, file_size)
                    skipped += 1
                    continue
                
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                file_hash = compute_file_hash(file_path)
                rel_path = file_path.relative_to(self.base_path).as_posix()
                
                chunks = self._chunk_content(content, file_path)
                
                for chunk_text, chunk_idx in chunks:
                    doc_id = f"{rel_path}::chunk_{chunk_idx}::{file_hash[:8]}"
                    current_file_ids.add(doc_id)
                    
                    if doc_id in existing_ids:
                        continue
                    
                    new_docs.append(chunk_text)
                    new_ids.append(doc_id)
                    new_metadatas.append({
                        "file_path": rel_path,
                        "file_hash": file_hash,
                        "chunk_index": chunk_idx,
                        "file_size": file_size
                    })
                    doc_counter += 1
                    
                    if len(new_docs) >= 100:
                        self._add_documents_batch(collection, new_docs, new_ids, new_metadatas)
                        new_docs, new_ids, new_metadatas = [], [], []
                        
            except Exception as e:
                logger.warning("Error processing %s: %s", file_path, e)
                skipped += 1
                continue
        
        if new_docs:
            self._add_documents_batch(collection, new_docs, new_ids, new_metadatas)
        
        stale_ids = existing_ids - current_file_ids
        if stale_ids:
            try:
                collection.delete(ids=list(stale_ids))
                logger.info("Removed %d stale documents", len(stale_ids))
            except Exception as e:
                logger.warning("Could not remove stale documents: %s", e)
        
        logger.info("Indexed %d new chunks (%d files skipped, %d stale removed)",
                   doc_counter, skipped, len(stale_ids))

    def _add_documents_batch(
        self,
        collection: Collection,
        documents: List[str],
        ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """Add a batch of documents to the collection.
        
        Args:
            collection: ChromaDB collection
            documents: List of document texts
            ids: List of document IDs
            metadatas: List of metadata dictionaries
        """
        try:
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            logger.debug("Added batch of %d documents", len(documents))
        except Exception as e:
            logger.error("Error adding documents batch: %s", e)
            raise

    def _build_global_bm25_index(self, path_filter: Optional[str] = None) -> BM25IndexType:
        """Build a BM25 index from all documents in the collection.
        
        This creates an index over the entire corpus for independent BM25 retrieval,
        which is essential for proper RRF fusion. The BM25 variant is selected
        based on the rag_config.bm25_implementation setting.
        
        Args:
            path_filter: Optional filter to only include documents matching path
            
        Returns:
            BM25 index populated with all documents
        """
        bm25_index = create_bm25_index(self.rag_config.bm25_implementation)
        
        # Get all documents from collection
        batch_size = 1000
        offset = 0
        
        while True:
            batch = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"]
            )
            
            if not batch or not batch.get("ids"):
                break
            
            ids = batch["ids"]
            documents = batch.get("documents", [])
            metadatas = batch.get("metadatas", [])
            
            for i, doc_id in enumerate(ids):
                doc = documents[i] if i < len(documents) else ""
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                
                # Apply path filter if specified
                if path_filter:
                    file_path = metadata.get("file_path", "")
                    if path_filter.lower() not in file_path.lower():
                        continue
                
                bm25_index.add_document(doc_id, doc)
            
            if len(ids) < batch_size:
                break
            
            offset += batch_size
        
        logger.debug("Built BM25 index with %d documents", len(bm25_index))
        return bm25_index

    def search_workspace(
        self,
        query: str,
        limit: int = 5,
        path_filter: Optional[str] = None,
        preview_window: Optional[int] = None,
        rrf_k: int = 60
    ) -> str:
        """Search the workspace knowledge base for relevant code snippets.
        
        Uses Reciprocal Rank Fusion (RRF) to combine semantic similarity and BM25
        keyword rankings. RRF fuses independent rankings from both methods,
        providing better recall than simple score blending.
        
        RRF Formula: score(d) = sum(1 / (k + rank_d)) for each retrieval method
        
        Args:
            query: Search query to find relevant code/text
            limit: Maximum number of results to return (default 5)
            path_filter: Optional substring to filter results by file path
            preview_window: Maximum characters for content preview (default None).
                            Set to None for no truncation.
            rrf_k: RRF constant (default 60). Higher values reduce impact of ranking differences.

        Returns:
            JSON string with search results containing:
                - status: "success" or "error"
                - count: Number of results found
                - results: Formatted string with snippets and RRF scores
                - query: The original search query
                
        Example:
            >>> result = tool.search_workspace("authentication function", limit=3)
            >>> data = json.loads(result)
            >>> print(data["results"])
        """
        if not self._collection:
            return json.dumps({
                "status": "error",
                "reason": "Collection not initialized",
                "query": query
            })
        
        if not query or not query.strip():
            return json.dumps({
                "status": "error",
                "reason": "Query cannot be empty",
                "query": query
            })
        
        try:
            # Phase 1: Independent Retrieval
            # Get semantic results from ChromaDB
            semantic_limit = min(limit * 20, 100)  # Get more candidates for better fusion
            semantic_results = self._collection.query(
                query_texts=[query.strip()],
                n_results=semantic_limit
            )
            
            # Get BM25 results from global index
            bm25_index = self._build_global_bm25_index(path_filter)
            
            if len(bm25_index) == 0:
                return json.dumps({
                    "status": "success",
                    "count": 0,
                    "message": "No documents in index" if not path_filter else f"No documents matching path filter: {path_filter}",
                    "query": query
                })
            
            bm25_scores = bm25_index.score(query)
            bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
            bm25_ranked = bm25_ranked[:semantic_limit]  # Match semantic result count
            
            # Check if we have any results
            semantic_empty = (not semantic_results or 
                            not semantic_results.get("documents") or 
                            not semantic_results["documents"][0])
            
            if semantic_empty and not bm25_ranked:
                return json.dumps({
                    "status": "success",
                    "count": 0,
                    "message": "No results found",
                    "query": query
                })
            
            # Phase 2: Prepare ranked lists for RRF
            # Format: list of (doc_id, score) tuples, sorted by relevance
            semantic_ranked: List[Tuple[str, float]] = []
            if not semantic_empty:
                doc_ids = semantic_results["ids"][0]
                distances = semantic_results.get("distances", [[]])[0]
                metadatas = semantic_results.get("metadatas", [[]])[0]
                
                for i, doc_id in enumerate(doc_ids):
                    # Apply path filter if specified
                    if path_filter and metadatas and i < len(metadatas):
                        file_path = metadatas[i].get("file_path", "")
                        if path_filter.lower() not in file_path.lower():
                            continue
                    
                    # Convert distance to similarity score (cosine distance -> similarity)
                    distance = distances[i] if i < len(distances) else 1.0
                    semantic_score = max(0.0, 1.0 - float(distance))
                    semantic_ranked.append((doc_id, semantic_score))
            
            # Phase 3: RRF Fusion
            rrf = ReciprocalRankFusion(k=rrf_k)
            ranked_lists = []
            if semantic_ranked:
                ranked_lists.append(semantic_ranked)
            if bm25_ranked:
                ranked_lists.append(bm25_ranked)
            
            fused_results = rrf.fuse(ranked_lists, limit=limit)
            
            if not fused_results:
                return json.dumps({
                    "status": "success",
                    "count": 0,
                    "message": "No results after fusion",
                    "query": query
                })
            
            # Phase 4: Fetch document details for top results
            fused_doc_ids = [doc_id for doc_id, _ in fused_results]
            doc_data = self._collection.get(
                ids=fused_doc_ids,
                include=["documents", "metadatas"]
            )
            
            # Create lookup maps
            doc_lookup = {}
            metadata_lookup = {}
            if doc_data:
                for i, doc_id in enumerate(doc_data.get("ids", [])):
                    docs = doc_data.get("documents", [])
                    metas = doc_data.get("metadatas", [])
                    if i < len(docs):
                        doc_lookup[doc_id] = docs[i]
                    if i < len(metas):
                        metadata_lookup[doc_id] = metas[i]
            
            # Get individual ranks for display
            semantic_rank_map = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(semantic_ranked)}
            bm25_rank_map = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(bm25_ranked)}
            
            # Build final results
            scored_results = []
            for doc_id, rrf_score in fused_results:
                doc = doc_lookup.get(doc_id, "")
                metadata = metadata_lookup.get(doc_id, {})
                
                semantic_rank = semantic_rank_map.get(doc_id, None)
                bm25_rank = bm25_rank_map.get(doc_id, None)
                
                # Get individual scores if available
                semantic_score = next((score for id, score in semantic_ranked if id == doc_id), 0.0)
                bm25_score = bm25_scores.get(doc_id, 0.0)
                
                scored_results.append({
                    "doc": doc,
                    "doc_id": doc_id,
                    "metadata": metadata,
                    "rrf_score": round(rrf_score, 4),
                    "semantic_rank": semantic_rank,
                    "bm25_rank": bm25_rank,
                    "semantic_score": round(semantic_score, 3),
                    "bm25_score": round(bm25_score, 3)
                })
            
            # Format output
            output_lines = [
                f"Found {len(scored_results)} relevant snippets using RRF fusion:",
                ""
            ]
            
            for i, result in enumerate(scored_results, 1):
                doc = result["doc"]
                
                # Build rank info string
                rank_info = f"RRF: {result['rrf_score']}"
                if result['semantic_rank']:
                    rank_info += f" | Semantic rank: #{result['semantic_rank']}"
                if result['bm25_rank']:
                    rank_info += f" | BM25 rank: #{result['bm25_rank']}"
                
                scores = f"(semantic: {result['semantic_score']}, bm25: {result['bm25_score']})"
                
                # Parse document format: [File: path]\n\ncontent
                lines = doc.split("\n")
                file_line = lines[0] if lines and lines[0].startswith("[File:") else ""
                content = "\n".join(lines[2:]) if file_line else doc
                
                if preview_window is None:
                    content_preview = content
                else:
                    content_preview = content[:preview_window] + ("..." if len(content) > preview_window else "")
                
                output_lines.extend([
                    f"--- Result {i} {rank_info} {scores} ---",
                    file_line,
                    content_preview,
                    ""
                ])
            
            # Calculate coverage stats
            semantic_hits = sum(1 for r in scored_results if r['semantic_rank'] is not None)
            bm25_hits = sum(1 for r in scored_results if r['bm25_rank'] is not None)
            both_hits = sum(1 for r in scored_results if r['semantic_rank'] and r['bm25_rank'])
            
            return json.dumps({
                "status": "success",
                "count": len(scored_results),
                "rrf_k": rrf_k,
                "coverage": {
                    "semantic_only": semantic_hits - both_hits,
                    "bm25_only": bm25_hits - both_hits,
                    "both_methods": both_hits
                },
                "results": "\n".join(output_lines).strip(),
                "query": query
            }, indent=2)
            
        except Exception as e:
            logger.exception("Search failed for query: %s", query)
            return json.dumps({
                "status": "error",
                "reason": "Search operation failed",
                "exception": str(e),
                "query": query
            })

    def get_index_stats(self) -> str:
        """Get statistics about the current index.
        
        Returns:
            JSON string with index statistics
        """
        if not self._collection:
            return json.dumps({
                "status": "error",
                "reason": "Collection not initialized"
            })
        
        try:
            data = self._collection.get()
            ids = data.get("ids", [])
            metadatas = data.get("metadatas", [])
            
            unique_files = set()
            total_size = 0
            for meta in metadatas:
                if meta:
                    unique_files.add(meta.get("file_path", "unknown"))
                    total_size += meta.get("file_size", 0)
            
            return json.dumps({
                "status": "success",
                "total_documents": len(ids),
                "unique_files": len(unique_files),
                "total_size_bytes": total_size,
                "collection_name": "workspace_code_index",
                "base_path": str(self.base_path)
            }, indent=2)
            
        except Exception as e:
            logger.exception("Failed to get index stats")
            return json.dumps({
                "status": "error",
                "reason": "Failed to retrieve statistics",
                "exception": str(e)
            })

    def refresh_index(self) -> str:
        """Force a refresh of the index by re-scanning all files.
        
        Returns:
            JSON string with refresh status
        """
        try:
            if self._collection:
                if self._gitignore_parser:
                    self._gitignore_parser.refresh()
                self._index_files(self._collection)
                return json.dumps({
                    "status": "success",
                    "message": "Index refreshed successfully"
                })
            else:
                return json.dumps({
                    "status": "error",
                    "reason": "Collection not initialized"
                })
        except Exception as e:
            logger.exception("Index refresh failed")
            return json.dumps({
                "status": "error",
                "reason": "Refresh operation failed",
                "exception": str(e)
            })