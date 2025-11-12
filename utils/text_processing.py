from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

def split_text(docs):
    """
    Split raw text from loaded documents into chunks for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_texts = []
    for d in docs:
        text = d.page_content if hasattr(d, "page_content") else str(d)
        all_texts.extend(splitter.split_text(text))
    return [Document(page_content=t) for t in all_texts]
