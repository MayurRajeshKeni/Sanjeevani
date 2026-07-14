from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(
    documents: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """Splits a list of Documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: A list of LangChain Document objects.
        chunk_size: The target maximum size of each chunk.
        chunk_overlap: The overlap between consecutive chunks.
        
    Returns:
        A list of split Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)
