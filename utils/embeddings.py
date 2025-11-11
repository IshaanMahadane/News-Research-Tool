import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class SafeOpenAIEmbeddings:
    def embed_query(self, text: str):
        try:
            response = openai.embeddings.create(model="text-embedding-3-small", input=text)
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Embedding error: {e}")

    def embed_documents(self, texts):
        try:
            response = openai.embeddings.create(model="text-embedding-3-small", input=texts)
            return [r.embedding for r in response.data]
        except Exception as e:
            raise RuntimeError(f"Embedding error: {e}")
