from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sqlite3
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import analyze_document, collection

# ---- App Setup ----
app = FastAPI(
    title="NLP Document Analyzer API",
    description="Multi-task NLP pipeline — NER, Classification, Summarization, Semantic Search",
    version="1.0.0"
)

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

print("API initialized!")

# ---- SQLite Setup ----
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
            similar_docs TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialized!")

def save_document(result: Dict[str, Any], text: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO documents 
        (id, text, doc_type, confidence, entities, summary, similar_docs, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        result["doc_id"],
        text,
        result["doc_type"],
        result["confidence"],
        json.dumps(result["entities"]),
        result["summary"],
        json.dumps(result["similar_docs"])
    ))
    conn.commit()
    conn.close()

def get_all_documents():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

# Initialize DB on startup
init_db()

# ---- Request/Response Models ----
class DocumentRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Apple Inc reported record quarterly earnings as CEO Tim Cook announced strong iPhone sales in Asia."
            }
        }

class EntityResponse(BaseModel):
    text: str
    type: str

class SimilarDocResponse(BaseModel):
    text: str
    doc_type: str
    summary: str

class DocumentResponse(BaseModel):
    doc_id: str
    doc_type: str
    confidence: float
    entities: List[EntityResponse]
    summary: str
    similar_docs: List[SimilarDocResponse]
    extracted_fields: Dict[str, Any]
    message: str = "Document analyzed successfully"

class DocumentListResponse(BaseModel):
    doc_id: str
    doc_type: str
    confidence: float
    summary: str
    created_at: str

class SearchRequest(BaseModel):
    query: str
    n_results: int = 3

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]

# ---- API Endpoints ----

# Health Check
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "NLP Document Analyzer API is live",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Analyze Single Document
@app.post("/analyze", response_model=DocumentResponse)
def analyze(request: DocumentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(request.text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long, max 10000 characters")

    try:
        result = analyze_document(request.text)
        save_document(result, request.text)
        return DocumentResponse(
            doc_id=result["doc_id"],
            doc_type=result["doc_type"],
            confidence=result["confidence"],
            entities=[EntityResponse(**e) for e in result["entities"]],
            summary=result["summary"],
            similar_docs=[SimilarDocResponse(**d) for d in result["similar_docs"]],
            extracted_fields=result["extracted_fields"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get All Documents
@app.get("/documents")
def get_documents():
    try:
        rows = get_all_documents()
        documents = []
        for row in rows:
            documents.append({
                "doc_id": row[0],
                "text_preview": row[1][:200],
                "doc_type": row[2],
                "confidence": row[3],
                "entities": json.loads(row[4]),
                "summary": row[5],
                "created_at": row[7]
            })
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get Single Document
@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "doc_id": row[0],
            "text": row[1],
            "doc_type": row[2],
            "confidence": row[3],
            "entities": json.loads(row[4]),
            "summary": row[5],
            "similar_docs": json.loads(row[6]),
            "created_at": row[7]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Semantic Search
@app.post("/search")
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        from pipeline import embedder
        query_embedding = embedder.encode(request.query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(request.n_results, max(1, collection.count()))
        )

        search_results = []
        if results["documents"][0]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                search_results.append({
                    "text_preview": doc[:200],
                    "doc_type": meta.get("doc_type", ""),
                    "summary": meta.get("summary", "")
                })

        return {
            "query": request.query,
            "results": search_results,
            "total": len(search_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Delete Document
@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()

        collection.delete(ids=[doc_id])

        return {"message": f"Document {doc_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Stats
@app.get("/stats")
def get_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type")
        type_counts = dict(cursor.fetchall())
        conn.close()

        return {
            "total_documents": total,
            "documents_by_type": type_counts,
            "vector_store_count": collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
