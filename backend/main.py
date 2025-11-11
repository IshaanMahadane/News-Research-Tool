from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai
import nltk

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


try:
    from langchain_community.vectorstores import Chroma
    chroma_available = True
except ImportError:
    chroma_available = False



app = FastAPI(title="News Research Tool API", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    raise Exception("❌ OpenAI API key not found in environment.")
openai.api_key = api_key


nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

index_folder = "chroma_db"



class SafeOpenAIEmbeddings:
    def embed_query(self, text: str):
        response = openai.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding

    def embed_documents(self, texts):
        response = openai.embeddings.create(model="text-embedding-3-small", input=texts)
        return [r.embedding for r in response.data]



class URLRequest(BaseModel):
    urls: list[str]


class QueryRequest(BaseModel):
    question: str



@app.post("/process-urls")
async def process_urls(req: URLRequest):
    try:
        valid_urls = [u.strip() for u in req.urls if u.strip()]
        if not valid_urls:
            raise HTTPException(status_code=400, detail="No valid URLs provided.")

        loader = UnstructuredURLLoader(urls=valid_urls)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(data)

        embeddings = SafeOpenAIEmbeddings()

        if chroma_available:
            vectorstore_chroma = Chroma.from_documents(
                docs, embeddings, persist_directory=index_folder
            )
            vectorstore_chroma.persist()
            return {"message": "✅ URLs processed and Chroma index saved."}
        else:
            return {"message": "⚠️ Chroma not installed locally — processed without saving index."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/ask")
async def ask_question(req: QueryRequest):
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        if chroma_available and os.path.exists(index_folder):
            embeddings = SafeOpenAIEmbeddings()
            vectorstore = Chroma(
                persist_directory=index_folder,
                embedding_function=embeddings
            )
            retriever = vectorstore.as_retriever()
        else:
            retriever = None

        prompt = ChatPromptTemplate.from_template(
            "Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
        )
        document_chain = prompt | llm | StrOutputParser()

        if retriever:
            retriever_chain = RunnableParallel({
                "context": retriever,
                "question": RunnablePassthrough(),
            })
            retrieval_chain = retriever_chain | document_chain
            result = retrieval_chain.invoke(req.question)
        else:
            result = llm.invoke(f"Answer this based on general knowledge: {req.question}")

        return {"answer": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/")
async def health():
    return {"status": "✅ API is running"}
