import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure required downloads
nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

def split_text(documents, chunk_size=1000, chunk_overlap=100):
    """Split text documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
