# 🚀 RAG Document QA System (FastAPI + LangChain + ChromaDB)

This project implements a **Retrieval-Augmented Generation (RAG)** system that allows users to:

- 📂 Ingest new documents (PDFs)
- 🧠 Build a semantic knowledge base
- ❓ Ask questions in natural language
- 🤖 Get accurate, context-aware answers using LLMs

The system is built using **FastAPI**, **LangChain**, **ChromaDB**, and **OpenAI (ChatGPT API)**, and is fully **Dockerized** for easy deployment.

---

# 📌 Features

- 📥 Dynamic document ingestion  
- 🧩 Smart chunking using RecursiveCharacterTextSplitter  
- 🧠 Embedding-based semantic search  
- ⚡ Fast retrieval with ChromaDB  
- 🤖 LLM-powered responses (OpenAI API)  
- 🐳 Dockerized FastAPI backend  
- 📚 Answers grounded in your own documents  

---

# 🧠 RAG Pipeline Overview

1. **Document Ingestion**
   - Upload PDF documents
   - Extract text from files

2. **Chunking**
   - Split text into manageable chunks
   - Uses `RecursiveCharacterTextSplitter`

3. **Embedding**
   - Convert chunks into vector embeddings
   - Model: `text-embedding-3-small`

4. **Storage**
   - Store embeddings in **ChromaDB**

5. **Query Processing**
   - Convert user query into embedding

6. **Retrieval**
   - Retrieve top-K relevant chunks

7. **Generation**
   - Pass context + query to LLM
   - Generate final answer

---

# 🏗️ Tech Stack

- **Backend:** FastAPI  
- **LLM:** OpenAI (ChatGPT API)  
- **Framework:** LangChain  
- **Vector DB:** ChromaDB  
- **Text Splitter:** RecursiveCharacterTextSplitter  
- **Containerization:** Docker  

---


---

# 📊 Dataset

The system uses **5 PDF documents** from top IT companies as the knowledge base.

These documents are ingested into the system and can be queried using natural language questions.

---

# ⚙️ Installation & Setup

## 🔹 Setup & Run

```bash
# Clone repository
git clone https://github.com/your-username/rag-fastapi-project.git
cd rag-fastapi-project

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Mac/Linux)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file and add your API key
echo OPENAI_API_KEY=your_api_key_here > .env
echo EMBEDDING_MODEL=text-embedding-3-small >> .env
echo CHAT_MODEL=gpt-4o-mini >> .env
echo CHUNK_SIZE=1200 >> .env
echo CHUNK_OVERLAP=200 >> .env

# Run FastAPI application
uvicorn app.main:app --reload

# Open in browser
# http://127.0.0.1:8000/docs
And test by giving different commands.


You can also run the docker container by running below commands and test the api.
## Build Docker image
docker build -t rag-fastapi-app .

# Run container
docker run -p 8000:8000 rag-fastapi-app

# Open in browser
# http://127.0.0.1:8000/docs
