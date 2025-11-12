from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

def split_text(docs):
    """
    Split raw text from loaded documents into chunks for embedding.
    This version avoids NLTK downloads (Railway/AWS safe).
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_texts = []
    for d in docs:
        text = getattr(d, "page_content", str(d))
        
        chunks = splitter.split_text(text)
        all_texts.extend(chunks)
    return [Document(page_content=t) for t in all_texts]
