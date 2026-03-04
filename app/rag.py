from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "db/chroma_db"

# ✅ Load ONCE (when rag.py is imported)
_embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=_embedding_model)
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def answer_query(question: str):

    retriever = _db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3}
    )

    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "Not found in documents.",
            "sources": [],
            "citations": []
        }

    citations = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "snippet": doc.page_content[:220]
        }
        for doc in docs
    ]
    # remove duplicate citations
    unique = []
    seen = set()
    for c in citations:
        key = (c["source"], c["snippet"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    citations = unique

    sources = sorted({c["source"] for c in citations})
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a retrieval-based assistant.
Use ONLY the provided context.
If answer not found, say "Not found in documents."

Question: {question}

Context:
{context}
"""

    response = _llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources,
        "citations": citations
    }