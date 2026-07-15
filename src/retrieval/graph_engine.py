import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

class GraphRetrievalEngine:
    """Graph Retrieval Engine that matches concept nodes semantically and fetches structural context (parents/children)."""
    
    def __init__(self, graph_path: str) -> None:
        """Initializes the Graph Engine and embeds all concept node titles.
        
        Args:
            graph_path: Absolute or relative path to the processed graph JSON file.
        """
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
            
        print(f"Loading Graph Data from {graph_path}...")
        with open(graph_path, "r", encoding="utf-8") as f:
            self.graph_data = json.load(f)
            
        self.nodes = self.graph_data.get("nodes", [])
        self.edges = self.graph_data.get("edges", [])
        
        print("Loading local embeddings model for Graph Node matching...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Pre-embed all node titles (concepts)
        self.node_titles = [node["title"] for node in self.nodes]
        print(f"Embedding {len(self.node_titles)} graph node concepts...")
        self.node_embeddings = self.embeddings.embed_documents(self.node_titles)
        
    def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[Document, float]]:
        """Finds the semantically closest node and returns it alongside parent/child context.
        
        Args:
            query: The search query string.
            top_k: Number of semantic concept nodes to match (defaults to 1).
            
        Returns:
            List of Tuples of (Document, score).
        """
        if not self.nodes:
            return []
            
        # 1. Semantic matching
        query_emb = self.embeddings.embed_query(query)
        
        # Since normalize_embeddings=True, dot product is exactly cosine similarity
        similarities = np.dot(self.node_embeddings, query_emb)
        
        # Get the index of the highest similarity
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_node = self.nodes[best_idx]
        best_node_id = best_node["id"]
        
        # Helper to format a graph node into a LangChain Document
        def to_doc(node: Dict[str, Any]) -> Document:
            return Document(
                page_content=node["content"],
                metadata={
                    "source": node["source_file"],
                    "node_id": node["id"],
                    "node_title": node["title"],
                    "node_level": node["level"],
                    "type": "graph_context"
                }
            )
            
        retrieved_docs: List[Tuple[Document, float]] = []
        
        # Add the best matched node (only if it has content)
        if best_node.get("content"):
            retrieved_docs.append((to_doc(best_node), best_score))
            
        # 2. Walk edges for immediate parent node(s)
        parent_ids = [edge["source"] for edge in self.edges if edge["target"] == best_node_id]
        for p_id in parent_ids:
            parent_node = next((n for n in self.nodes if n["id"] == p_id), None)
            if parent_node and parent_node.get("content"):
                # Give parents a slightly lower score weighting
                retrieved_docs.append((to_doc(parent_node), best_score * 0.8))
                
        # 3. Walk edges for immediate child node(s)
        child_ids = [edge["target"] for edge in self.edges if edge["source"] == best_node_id]
        for c_id in child_ids:
            child_node = next((n for n in self.nodes if n["id"] == c_id), None)
            if child_node and child_node.get("content"):
                # Give children a lower score weighting
                retrieved_docs.append((to_doc(child_node), best_score * 0.6))
                
        # Deduplicate results by node ID to ensure safety
        seen_ids = set()
        deduplicated_docs = []
        for doc, score in retrieved_docs:
            n_id = doc.metadata["node_id"]
            if n_id not in seen_ids:
                seen_ids.add(n_id)
                deduplicated_docs.append((doc, score))
                
        return deduplicated_docs
