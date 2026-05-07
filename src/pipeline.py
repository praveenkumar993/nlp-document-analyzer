from fastapi import requests

import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    BertTokenizerFast,
    BertForSequenceClassification
)
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import json
import requests

load_dotenv()

# ---- Labels ----
NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
CLASSIFIER_LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# ---- Devices ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---- Load Models ----
print("Loading NER model...")
ner_tokenizer = DistilBertTokenizerFast.from_pretrained("models/ner_model")
ner_model = DistilBertForTokenClassification.from_pretrained("models/ner_model")
ner_model.to(DEVICE)
ner_model.eval()

print("Loading Classifier model...")
cls_tokenizer = BertTokenizerFast.from_pretrained("models/classifier_model")
cls_model = BertForSequenceClassification.from_pretrained("models/classifier_model")
cls_model.to(DEVICE)
cls_model.eval()

print("Loading Sentence Transformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Setting up ChromaDB...")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("documents")

print("Setting up Summarizer...")
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_TOKEN = os.getenv("HF_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

print("All models loaded!")

# ---- LangGraph State ----
class DocumentState(TypedDict):
    text: str
    entities: List[Dict[str, str]]
    doc_type: str
    confidence: float
    summary: str
    doc_id: str
    similar_docs: List[Dict[str, Any]]
    error: str


# ---- Node 1: Preprocessor ----
def preprocess(state: DocumentState) -> DocumentState:
    text = state["text"].strip()
    text = " ".join(text.split())  # remove extra whitespace
    state["text"] = text
    state["doc_id"] = str(abs(hash(text)))[:8]
    return state


# ---- Node 2: NER ----
def run_ner(state: DocumentState) -> DocumentState:
    text = state["text"]
    inputs = ner_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = ner_model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    tokens = ner_tokenizer.convert_ids_to_tokens(
        inputs["input_ids"].squeeze().tolist()
    )

    entities = []
    current_entity = None

    for token, pred in zip(tokens, predictions):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        label = NER_LABELS[pred]

        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": token,
                "type": label[2:]
            }
        elif label.startswith("I-") and current_entity:
            if token.startswith("##"):
                # subword token - append without space
                current_entity["text"] += token[2:]
            else:
                # new word - append with space
                current_entity["text"] += " " + token
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    state["entities"] = entities
    return state

# ---- Node 3: Classifier ----
def run_classifier(state: DocumentState) -> DocumentState:
    text = state["text"]
    inputs = cls_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = cls_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    pred_id = torch.argmax(probs).item()
    confidence = probs[pred_id].item()

    state["doc_type"] = CLASSIFIER_LABELS[pred_id]
    state["confidence"] = round(confidence, 4)
    return state

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_TOKEN = os.getenv("HF_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def run_summarizer(state: DocumentState) -> DocumentState:
    text = state["text"]

    payload = {
        "inputs": text[:1000],
        "parameters": {
            "max_length": 150,
            "min_length": 50,
            "do_sample": False
        }
    }

    response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload)

    if response.status_code == 200:
        summary = response.json()[0]["summary_text"]
    else:
        sentences = text.strip().split(".")
        summary = ". ".join(sentences[:2]).strip() + "."

    state["summary"] = summary
    return state


# ---- Node 5: Vector Store ----
def run_vector_store(state: DocumentState) -> DocumentState:
    text = state["text"]
    doc_id = state["doc_id"]

    # Store document
    embedding = embedder.encode(text).tolist()
    collection.upsert(
        documents=[text],
        embeddings=[embedding],
        ids=[doc_id],
        metadatas=[{
            "doc_type": state["doc_type"],
            "summary": state["summary"]
        }]
    )

    # Find similar documents
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(3, collection.count())
    )

    similar = []
    if results["documents"][0]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            if doc != text:
                similar.append({
                    "text": doc[:200],
                    "doc_type": meta.get("doc_type", ""),
                    "summary": meta.get("summary", "")
                })

    state["similar_docs"] = similar
    return state

# ---- Build LangGraph ----
def build_pipeline():
    graph = StateGraph(DocumentState)

    # Add nodes
    graph.add_node("preprocess", preprocess)
    graph.add_node("ner", run_ner)
    graph.add_node("classifier", run_classifier)
    graph.add_node("summarizer", run_summarizer)
    graph.add_node("vector_store", run_vector_store)

    # Add edges
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "ner")
    graph.add_edge("ner", "classifier")
    graph.add_edge("classifier", "summarizer")
    graph.add_edge("summarizer", "vector_store")
    graph.add_edge("vector_store", END)

    return graph.compile()


# ---- Run Pipeline ----
def analyze_document(text: str) -> Dict[str, Any]:
    pipeline = build_pipeline()

    initial_state = DocumentState(
        text=text,
        entities=[],
        doc_type="",
        confidence=0.0,
        summary="",
        doc_id="",
        similar_docs=[],
        error=""
    )

    result = pipeline.invoke(initial_state)

    return {
        "doc_id": result["doc_id"],
        "doc_type": result["doc_type"],
        "confidence": result["confidence"],
        "entities": result["entities"],
        "summary": result["summary"],
        "similar_docs": result["similar_docs"]
    }


# ---- Test ----
if __name__ == "__main__":
    test_text = """
    Apple Inc reported record quarterly earnings yesterday as CEO Tim Cook 
    announced strong iPhone sales in Asia. The company based in Cupertino 
    California saw revenues rise 15 percent driven by services growth. 
    Goldman Sachs raised their price target following the announcement.
    """

    print("Analyzing document...")
    result = analyze_document(test_text)

    print("\n=== RESULTS ===")
    print(f"Document ID: {result['doc_id']}")
    print(f"Document Type: {result['doc_type']} (confidence: {result['confidence']})")
    print(f"\nEntities Found: {result['entities']}")
    print(f"\nSummary: {result['summary']}")
    print(f"\nSimilar Documents: {result['similar_docs']}")