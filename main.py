import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from utils.text_processing import split_text
from utils.chroma_handler import create_chroma_index, load_chroma_index
from utils.logger import get_logger

# Load environment variables
load_dotenv()

# Initialize FastAPI and Logger
app = FastAPI(title="🧠 SmartBot: News Research Tool", version="1.0")
logger = get_logger("SmartBot")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Request Models
class URLRequest(BaseModel):
    urls: list[str]

class QuestionRequest(BaseModel):
    question: str

# Root route for health check
@app.get("/")
def root():
    return {"message": "🧠 SmartBot API is running!"}


# Endpoint: Process URLs
@app.post("/process-urls")
async def process_urls(request: URLRequest):
    """
    1️⃣ Fetch text from given URLs
    2️⃣ Split into chunks
    3️⃣ Embed & store in Chroma
    """
    try:
        urls = [u.strip() for u in request.urls if u.strip()]
        if not urls:
            raise HTTPException(status_code=400, detail="No valid URLs provided.")

        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        docs = split_text(data)
        create_chroma_index(docs)

        logger.info("✅ Chroma index created successfully.")
        return {"message": "✅ URLs processed and Chroma index saved!"}

    except Exception as e:
        logger.error(f"Error processing URLs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint: Ask a Question
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    1️⃣ Load Chroma vector store
    2️⃣ Retrieve relevant context
    3️⃣ Use OpenAI to answer the question
    """
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        # Load existing Chroma index
        vectorstore = load_chroma_index()

        # Create retriever + LLM chain
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        prompt = ChatPromptTemplate.from_template(
            "Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
        )

        document_chain = prompt | llm | StrOutputParser()
        retriever_chain = RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough(),
        })
        retrieval_chain = retriever_chain | document_chain

        result = retrieval_chain.invoke(question)
        logger.info(f"💬 Question answered: {question}")

        return {"answer": result}

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))
