import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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


load_dotenv()
logger = get_logger("SmartBot")
app = FastAPI(title="🧠 SmartBot: News Research Tool", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class URLRequest(BaseModel):
    urls: list[str]

class QuestionRequest(BaseModel):
    question: str


@app.post("/process-urls")
async def process_urls(request: URLRequest):
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


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        vectorstore = load_chroma_index()
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        prompt = ChatPromptTemplate.from_template(
            "Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
        )

        chain = (
            RunnableParallel({
                "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                "question": RunnablePassthrough(),
            })
            | (prompt | llm | StrOutputParser())
        )

        result = chain.invoke(question)
        return {"answer": result}

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")

app.mount("/static", StaticFiles(directory="frontend"), name="static")
