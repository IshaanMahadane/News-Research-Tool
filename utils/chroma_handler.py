from langchain_openai import OpenAIEmbeddings
from chromadb.utils import embedding_functions
from chromadb import Client
from chromadb.config import Settings
import os

def create_chroma_index(docs):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing OPENAI_API_KEY environment variable!")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key  # ✅ Modern API (no proxies)
    )

    from langchain_community.vectorstores import Chroma
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    vectorstore.persist()
    return vectorstore


def load_chroma_index():
    from langchain_community.vectorstores import Chroma
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vectorstore
