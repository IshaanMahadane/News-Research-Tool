import os
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from langchain.embeddings.base import Embeddings

# ✅ Custom wrapper to bypass 'proxies' issue
class SimpleOpenAIEmbeddings(Embeddings):
    def __init__(self, model="text-embedding-3-small"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [data.embedding for data in response.data]

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding


def create_chroma_index(docs):
    embeddings = SimpleOpenAIEmbeddings()  # ✅ Uses direct OpenAI API
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    vectorstore.persist()
    return vectorstore


def load_chroma_index():
    embeddings = SimpleOpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vectorstore
