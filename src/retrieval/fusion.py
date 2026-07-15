import hashlib
import os
import re
from typing import List, Dict, Tuple, Any
from langchain_core.documents import Document
from .vector_engine import VectorRetrievalEngine
from .sparse_engine import SparseRetrievalEngine
from .graph_engine import GraphRetrievalEngine

def normalize_text(text: str) -> str:
    """Helper to normalize whitespace and lowercase text for robust substring matching."""
    return re.sub(r"\s+", " ", text).strip().lower()

class HybridRetriever:
    """Fuses Sparse (BM25), Dense (FAISS), and Structural Graph retrieval using Reciprocal Rank Fusion."""
    
    def __init__(
        self, 
        vector_engine: VectorRetrievalEngine, 
        sparse_engine: SparseRetrievalEngine, 
        graph_engine: GraphRetrievalEngine
    ) -> None:
        """Initializes the Hybrid Retriever with all three engines.
        
        Args:
            vector_engine: Reference to the dense vector database engine.
            sparse_engine: Reference to the sparse lexical BM25 engine.
            graph_engine: Reference to the structural graph engine.
        """
        self.vector = vector_engine
        self.sparse = sparse_engine
        self.graph = graph_engine
         
    def retrieve(self, query: str, top_k: int = 5, k: int = 60) -> List[Document]:
        """Runs retrieval across all three engines and merges results using Reciprocal Rank Fusion.
        
        Args:
            query: The user prompt or search query.
            top_k: The number of final fused documents to return.
            k: The RRF smoothing constant (default: 60).
            
        Returns:
            List of combined and re-ranked Document objects.
        """
        # 1. Fetch search results from Vector and Sparse engines
        vector_results = self.vector.retrieve(query, top_k=20)
        sparse_results = self.sparse.retrieve(query, top_k=20)
        
        # 2. Fetch structural graph nodes
        graph_node_results = self.graph.retrieve(query, top_k=10)
        
        # 3. Map hierarchical graph nodes back to their constituent text chunks in the corpus
        graph_chunk_results: List[Tuple[Document, float]] = []
        seen_chunk_hashes = set()
        
        for rank_idx, (graph_node_doc, score) in enumerate(graph_node_results):
            node_content = graph_node_doc.page_content.strip()
            if not node_content:
                continue
                
            node_norm = normalize_text(node_content)
            # Filter out very short lines (e.g. headers, list dots) and extract long lines for robust matching
            node_lines = [line.strip() for line in node_content.split("\n") if len(line.strip()) > 20]
            
            # Match chunks that reside within or overlap with this graph node's content
            for chunk in self.sparse.documents:
                chunk_source = os.path.basename(chunk.metadata.get("source", ""))
                node_source = os.path.basename(graph_node_doc.metadata.get("source", ""))
                
                # Restrict matches to the same source file
                if chunk_source != node_source:
                    continue
                    
                chunk_content = chunk.page_content.strip()
                chunk_norm = normalize_text(chunk_content)
                
                # A chunk matches a graph node if:
                # - The chunk is a substring of the node content, or
                # - The node content is a substring of the chunk, or
                # - Any of the long text lines in the node content matches inside the chunk
                is_match = (
                    chunk_norm in node_norm or
                    node_norm in chunk_norm or
                    (len(node_lines) > 0 and any(normalize_text(line) in chunk_norm for line in node_lines))
                )
                
                if is_match:
                    chunk_hash = hashlib.md5(chunk_content.encode("utf-8")).hexdigest()
                    if chunk_hash not in seen_chunk_hashes:
                        seen_chunk_hashes.add(chunk_hash)
                        # Copy the chunk and annotate it with the graph metadata
                        chunk_copy = Document(
                            page_content=chunk.page_content,
                            metadata=dict(chunk.metadata)
                        )
                        chunk_copy.metadata["graph_node_id"] = graph_node_doc.metadata["node_id"]
                        chunk_copy.metadata["graph_node_title"] = graph_node_doc.metadata["node_title"]
                        graph_chunk_results.append((chunk_copy, score))
                        
        # 4. Fused store mapping: doc_hash -> {"doc": Document, "ranks": Dict[str, int], "rrf_score": float}
        fused_store: Dict[str, Dict[str, Any]] = {}
        
        # Helper to process ranks and accumulate RRF score
        def process_engine_results(results: List[Tuple[Document, float]], engine_name: str) -> None:
            for rank_idx, (doc, _) in enumerate(results):
                rank = rank_idx + 1 # 1-indexed rank
                content = doc.page_content.strip()
                # Create a content-based hash to deduplicate the same chunk across engines
                doc_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
                
                if doc_hash not in fused_store:
                    fused_store[doc_hash] = {
                        "doc": doc,
                        "ranks": {"vector": None, "sparse": None, "graph": None},
                        "rrf_score": 0.0
                    }
                    
                fused_store[doc_hash]["ranks"][engine_name] = rank
                # Apply RRF score formula
                fused_store[doc_hash]["rrf_score"] += 1.0 / (k + rank)

        process_engine_results(vector_results, "vector")
        process_engine_results(sparse_results, "sparse")
        process_engine_results(graph_chunk_results, "graph")
        
        # 5. Sort documents by total RRF score in descending order
        sorted_store = sorted(fused_store.values(), key=lambda x: x["rrf_score"], reverse=True)
        
        # 6. Construct final return Document list
        final_docs = []
        for item in sorted_store[:top_k]:
            doc = item["doc"]
            # Save RRF metadata directly into the document so it can be monitored in evaluation & UI
            doc.metadata["rrf_score"] = item["rrf_score"]
            doc.metadata["rrf_ranks"] = item["ranks"]
            final_docs.append(doc)
            
        return final_docs
