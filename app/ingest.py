import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

def ingest_documents(docs_path="docs", persist_directory="db/chroma_db"):

    # Ensure docs folder exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist")

    # Load text files
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No documents found in the docs folder")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # Create embedding model
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # Create vector store (auto-persisted in new versions)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    return {
        "message": "Documents ingested successfully",
        "total_documents": len(documents),
        "total_chunks": len(chunks)
    }