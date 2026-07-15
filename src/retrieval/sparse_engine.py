import re
from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> List[str]:
    """Tokenizes text by lowercasing and splitting alphanumeric sequences.
    
    Args:
        text: The raw input string.
        
    Returns:
        List of lowercase token strings.
    """
    return re.findall(r"\w+", text.lower())

class SparseRetrievalEngine:
    """Sparse Lexical Retrieval Engine using BM25 (rank_bm25)."""
    
    def __init__(self, documents: List[Document]) -> None:
        """Initializes the BM25 index with the provided document chunks.
        
        Args:
            documents: List of LangChain Document objects.
        """
        print("Indexing documents into BM25 Lexical Store...")
        self.documents = documents
        self.tokenized_corpus = [tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieves top_k documents matching the query keywords.
        
        Args:
            query: The search query string.
            top_k: Number of documents to return.
            
        Returns:
            List of Tuples of (Document, bm25_score).
        """
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Zip documents with scores and sort by score descending
        doc_scores = list(zip(self.documents, scores))
        sorted_docs = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        # Return the top_k hits
        return sorted_docs[:top_k]
