import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorRetrievalEngine:
    """Dense Semantic Retrieval Engine using all-MiniLM-L6-v2 embeddings and in-memory/cached FAISS."""
    
    def __init__(self, documents: List[Document], persist_dir: str = None) -> None:
        """Initializes the Vector DB with the provided document chunks or loads cached FAISS index.
        
        Args:
            documents: List of LangChain Document objects.
            persist_dir: Directory path to load/save the FAISS index.
        """
        print("Loading local embeddings model (all-MiniLM-L6-v2)...")
        # Load the lightweight MiniLM model (approx 90MB RAM footprint)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"local_files_only": True},
            encode_kwargs={"normalize_embeddings": True}  # Normalizing yields cosine similarity
        )
        
        if persist_dir and os.path.exists(os.path.join(persist_dir, "index.faiss")):
            print(f"Loading cached FAISS index from {persist_dir}...")
            self.db = FAISS.load_local(persist_dir, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("Indexing documents into in-memory FAISS Vector Store...")
            self.db = FAISS.from_documents(documents, self.embeddings)
            if persist_dir:
                print(f"Caching FAISS index to {persist_dir}...")
                os.makedirs(persist_dir, exist_ok=True)
                self.db.save_local(persist_dir)
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieves top_k chunks matching the query semantically.
        
        Args:
            query: The search query string.
            top_k: Number of chunks to return.
            
        Returns:
            List of Tuples of (Document, similarity_score).
        """
        # similarity_search_with_relevance_scores returns (doc, score) where score is cosine similarity
        results = self.db.similarity_search_with_relevance_scores(query, k=top_k)
        return results
