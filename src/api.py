from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import sqlite3
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import analyze_document, collection, embedder

# ---- App ----
app = FastAPI(
    title="NLP Document Analyzer API",
    description="Multi-task NLP pipeline — NER, Classification, Summarization, Semantic Search",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---- SQLite ----
DB_PATH = "data/documents.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            doc_type TEXT,
            confidence REAL,
            entities TEXT,
            summary TEXT,
            extracted_fields TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_document(result: Dict[str, Any], text: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO documents
        (id, text, doc_type, confidence, entities, summary, extracted_fields)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        result["doc_id"],
        text,
        result["doc_type"],
        result["confidence"],
        json.dumps(result["entities"]),
        result["summary"],
        json.dumps(result["extracted_fields"])
    ))
    conn.commit()
    conn.close()

init_db()

# ---- Models ----
class DocumentRequest(BaseModel):
    text: str

class EntityResponse(BaseModel):
    text: str
    type: str

class DocumentResponse(BaseModel):
    doc_id: str
    doc_type: str
    confidence: float
    entities: List[EntityResponse]
    summary: str
    extracted_fields: Dict[str, Any]

class SearchRequest(BaseModel):
    query: str
    n_results: int = 5

# ---- Endpoints ----

@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0.0"}


@app.post("/analyze", response_model=DocumentResponse)
def analyze(request: DocumentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(request.text) > 15000:
        raise HTTPException(status_code=400, detail="Text too long — max 15,000 characters")
    try:
        result = analyze_document(request.text)
        save_document(result, request.text)
        return DocumentResponse(
            doc_id=result["doc_id"],
            doc_type=result["doc_type"],
            confidence=result["confidence"],
            entities=[EntityResponse(**e) for e in result["entities"]],
            summary=result["summary"],
            extracted_fields=result["extracted_fields"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
def get_documents():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, doc_type, confidence, entities, summary, extracted_fields, created_at FROM documents ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        documents = []
        for row in rows:
            documents.append({
                "doc_id": row[0],
                "text_preview": row[1][:200],
                "doc_type": row[2],
                "confidence": row[3],
                "entities": json.loads(row[4]),
                "summary": row[5],
                "extracted_fields": json.loads(row[6]),
                "created_at": row[7]
            })
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        count = collection.count()
        if count == 0:
            return {"query": request.query, "results": [], "total": 0}
        query_embedding = embedder.encode(request.query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_results, count)
        )
        search_results = []
        if results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                search_results.append({
                    "text_preview": doc[:300],
                    "doc_type": meta.get("doc_type", ""),
                    "summary": meta.get("summary", "")
                })
        return {"query": request.query, "results": search_results, "total": len(search_results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type ORDER BY COUNT(*) DESC")
        type_counts = dict(cursor.fetchall())
        conn.close()
        return {
            "total_documents": total,
            "documents_by_type": type_counts,
            "vector_store_count": collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
