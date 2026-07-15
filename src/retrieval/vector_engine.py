from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorRetrievalEngine:
    """Dense Semantic Retrieval Engine using all-MiniLM-L6-v2 embeddings and in-memory FAISS."""
    
    def __init__(self, documents: List[Document]) -> None:
        """Initializes the Vector DB with the provided document chunks.
        
        Args:
            documents: List of LangChain Document objects.
        """
        print("Loading local embeddings model (all-MiniLM-L6-v2)...")
        # Load the lightweight MiniLM model (approx 90MB RAM footprint)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}  # Normalizing yields cosine similarity
        )
        
        print("Indexing documents into in-memory FAISS Vector Store...")
        self.db = FAISS.from_documents(documents, self.embeddings)
        
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
