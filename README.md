**SmartBot — AI-Powered News Research Assistant**

SmartBot is a production-style Retrieval-Augmented Generation (RAG) system that enables users to ingest online news articles, index them in a vector database, and ask natural-language questions grounded in the retrieved content.

The system is built using FastAPI, LangChain, OpenAI APIs, and ChromaDB, and includes a lightweight web interface for interactive querying.

**Key Highlights**

Designed and implemented a full-stack RAG pipeline for news research and knowledge extraction

Built REST APIs using FastAPI for document ingestion and question answering

Integrated ChromaDB for embedding storage and semantic retrieval

Implemented prompt orchestration with LangChain runnables

Added logging and error handling for production readiness

Served a static frontend through FastAPI for seamless deployment

Enabled CORS for cross-origin clients and UI experimentation

**Architecture Overview**

URLs are ingested via the /process-urls endpoint

Article content is extracted using Unstructured loaders

Documents are chunked and embedded

Embeddings are persisted in ChromaDB

A retriever selects relevant passages

Context and question are composed into a prompt

OpenAI models generate grounded responses

**Tech Stack**

*Backend*

Python, FastAPI

LangChain Core & OpenAI Integration

ChromaDB Vector Store

Gunicorn / Uvicorn

*Frontend*

HTML, CSS, JavaScript

*Data & Processing*

NumPy, Pandas

