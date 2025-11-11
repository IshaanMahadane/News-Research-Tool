import os
from langchain_community.vectorstores import Chroma
from utils.embeddings import SafeOpenAIEmbeddings

INDEX_DIR = "chroma_db"

def create_chroma_index(docs):
    """Create a Chroma index from documents."""
    embeddings = SafeOpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=INDEX_DIR)
    vectorstore.persist()
    return vectorstore

def load_chroma_index():
    """Load existing Chroma index if available."""
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError("Chroma index not found. Please process URLs first.")
    embeddings = SafeOpenAIEmbeddings()
    return Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings)
