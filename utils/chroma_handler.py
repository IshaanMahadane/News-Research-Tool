from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Use global store (memory-only for Railway)
global_vectorstore = None

def create_chroma_index(docs):
    """
    Create an in-memory Chroma index from text chunks.
    """
    global global_vectorstore
    embeddings = OpenAIEmbeddings()
    global_vectorstore = Chroma.from_texts([d.page_content for d in docs], embedding=embeddings)
    return global_vectorstore

def load_chroma_index():
    """
    Retrieve the in-memory Chroma index (if created).
    """
    global global_vectorstore
    if not global_vectorstore:
        raise ValueError("No in-memory vector store found. Process URLs first!")
    return global_vectorstore
