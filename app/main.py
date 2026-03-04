from fastapi import FastAPI
from pydantic import BaseModel
from app.ingest import ingest_documents
from app.rag import answer_query

app = FastAPI(title="RAG API")

class QuestionRequest(BaseModel):
    question: str

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/ingest")
def ingest():
    return ingest_documents()

@app.post("/ask")
def ask_question(request: QuestionRequest):
    return answer_query(request.question)