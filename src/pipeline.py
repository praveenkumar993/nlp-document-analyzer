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

import re

# ---- Document Type Detector ----
def detect_document_type(text: str) -> str:
    text_lower = text.lower()

    # Invoice patterns
    invoice_keywords = [
        "invoice", "total due", "payment due", "invoice number",
        "bill to", "ship to", "subtotal", "tax", "amount due",
        "purchase order", "receipt", "invoice date", "due date"
    ]
    invoice_score = sum(1 for kw in invoice_keywords if kw in text_lower)

    # Email patterns
    email_keywords = [
        "dear", "regards", "sincerely", "hi ", "hello",
        "subject:", "from:", "to:", "cc:", "best regards",
        "please find", "attached", "let me know", "thank you for"
    ]
    email_score = sum(1 for kw in email_keywords if kw in text_lower)

    # Support ticket patterns
    ticket_keywords = [
        "ticket", "issue", "priority", "bug", "error",
        "support", "request", "problem", "resolve", "status",
        "assigned to", "reported by", "severity", "incident"
    ]
    ticket_score = sum(1 for kw in ticket_keywords if kw in text_lower)

    # Decide based on scores
    scores = {
        "Invoice": invoice_score,
        "Email": email_score,
        "Support Ticket": ticket_score
    }

    max_type = max(scores, key=scores.get)
    max_score = scores[max_type]

    # Only use rule-based if confident enough
    if max_score >= 2:
        return max_type
    else:
        return "General"  # fall back to ML classifier
    
# ---- Specialized Field Extractor ----
def extract_invoice_fields(text: str) -> dict:
    fields = {}

    # Invoice number
    inv_match = re.search(r'invoice\s*#?\s*:?\s*([A-Z0-9\-]+)', text, re.IGNORECASE)
    if inv_match:
        fields["invoice_number"] = inv_match.group(1)

    # Order number
    order_match = re.search(r'order\s*#?\s*:?\s*([A-Z0-9\-]+)', text, re.IGNORECASE)
    if order_match:
        fields["order_number"] = order_match.group(1)

    # Total amount
    amount_match = re.search(r'total\s*due\s*:?\s*\$?([\d,]+\.?\d*)', text, re.IGNORECASE)
    if amount_match:
        fields["total_due"] = f"${amount_match.group(1)}"

    # Tax
    tax_match = re.search(r'tax\s*:?\s*\$?([\d,]+\.?\d*)', text, re.IGNORECASE)
    if tax_match:
        fields["tax"] = f"${tax_match.group(1)}"

    # Invoice date
    date_match = re.search(r'invoice\s*date\s*:?\s*([A-Za-z]+\s+\d+,?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
    if date_match:
        fields["invoice_date"] = date_match.group(1)

    # Due date
    due_match = re.search(r'due\s*date\s*:?\s*([A-Za-z]+\s+\d+,?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text, re.IGNORECASE)
    if due_match:
        fields["due_date"] = due_match.group(1)

    # Vendor name (From section)
    vendor_match = re.search(r'from:\s*\n?(.+?)(?:\n|$)', text, re.IGNORECASE)
    if vendor_match:
        fields["vendor"] = vendor_match.group(1).strip()

    # Client name (To section)
    client_match = re.search(r'to:\s*\n?(.+?)(?:\n|$)', text, re.IGNORECASE)
    if client_match:
        fields["client"] = client_match.group(1).strip()

    return fields


def extract_email_fields(text: str) -> dict:
    fields = {}

    # Subject
    subject_match = re.search(r'subject\s*:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if subject_match:
        fields["subject"] = subject_match.group(1).strip()

    # From
    from_match = re.search(r'from\s*:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if from_match:
        fields["from"] = from_match.group(1).strip()

    # To
    to_match = re.search(r'to\s*:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if to_match:
        fields["to"] = to_match.group(1).strip()

    # Detect intent
    text_lower = text.lower()
    if any(w in text_lower for w in ["complaint", "issue", "problem", "unhappy", "disappointed"]):
        fields["intent"] = "Complaint"
    elif any(w in text_lower for w in ["request", "please", "could you", "can you"]):
        fields["intent"] = "Request"
    elif any(w in text_lower for w in ["thank", "thanks", "appreciate"]):
        fields["intent"] = "Appreciation"
    elif any(w in text_lower for w in ["follow up", "following up", "checking in"]):
        fields["intent"] = "Follow Up"
    else:
        fields["intent"] = "General"

    return fields


def extract_ticket_fields(text: str) -> dict:
    fields = {}

    # Ticket ID
    ticket_match = re.search(r'ticket\s*#?\s*:?\s*([A-Z0-9\-]+)', text, re.IGNORECASE)
    if ticket_match:
        fields["ticket_id"] = ticket_match.group(1)

    # Priority
    priority_match = re.search(r'priority\s*:?\s*(low|medium|high|critical|urgent)', text, re.IGNORECASE)
    if priority_match:
        fields["priority"] = priority_match.group(1).capitalize()
    else:
        text_lower = text.lower()
        if any(w in text_lower for w in ["urgent", "critical", "asap", "immediately"]):
            fields["priority"] = "High"
        elif any(w in text_lower for w in ["low", "minor", "whenever"]):
            fields["priority"] = "Low"
        else:
            fields["priority"] = "Medium"

    # Status
    status_match = re.search(r'status\s*:?\s*(open|closed|pending|resolved|in progress)', text, re.IGNORECASE)
    if status_match:
        fields["status"] = status_match.group(1).capitalize()
    else:
        fields["status"] = "Open"

    # Issue type
    text_lower = text.lower()
    if any(w in text_lower for w in ["login", "password", "access", "authentication"]):
        fields["issue_type"] = "Access Issue"
    elif any(w in text_lower for w in ["crash", "error", "bug", "not working"]):
        fields["issue_type"] = "Bug Report"
    elif any(w in text_lower for w in ["slow", "performance", "timeout", "latency"]):
        fields["issue_type"] = "Performance Issue"
    elif any(w in text_lower for w in ["install", "setup", "configure"]):
        fields["issue_type"] = "Installation Issue"
    else:
        fields["issue_type"] = "General Issue"

    return fields


def extract_fields(doc_type: str, text: str) -> dict:
    if doc_type == "Invoice":
        return extract_invoice_fields(text)
    elif doc_type == "Email":
        return extract_email_fields(text)
    elif doc_type == "Support Ticket":
        return extract_ticket_fields(text)
    else:
        return {}
    


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
    extracted_fields: Dict[str, Any]
    error: str


# ---- Node 1: Preprocessor ----
def preprocess(state: DocumentState) -> DocumentState:
    text = state["text"].strip()
    text = " ".join(text.split())  # remove extra whitespace
    state["text"] = text
    state["doc_id"] = str(abs(hash(text)))[:8]
    return state

def clean_entities(entities):
    cleaned = []
    for entity in entities:
        text = entity["text"].strip()
        if len(text) < 2:
            continue
        if text.startswith("##"):
            continue
        cleaned.append(entity)
    return cleaned

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

    state["entities"] = clean_entities(entities)
    return state

# ---- Node 3: Classifier ----
def run_classifier(state: DocumentState) -> DocumentState:
    text = state["text"]

    # Rule based detection first
    rule_based_type = detect_document_type(text)

    if rule_based_type != "General":
        # Rule based was confident
        state["doc_type"] = rule_based_type
        state["confidence"] = 1.0
    else:
        # Fall back to ML classifier
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

    # Extract specialized fields
    state["extracted_fields"] = extract_fields(state["doc_type"], text)

    return state

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_TOKEN = os.getenv("HF_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def run_summarizer(state: DocumentState) -> DocumentState:
    text = state["text"]
    doc_type = state["doc_type"]
    entities = state["entities"]

    entity_names = [e["text"] for e in entities]
    entity_str = ", ".join(entity_names) if entity_names else "none"

    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]

    if len(sentences) <= 2:
        summary = text.strip()
    else:
        scored = []
        for i, sent in enumerate(sentences):
            score = 0
            if i == 0:
                score += 3
            for entity in entity_names:
                if entity.lower() in sent.lower():
                    score += 2
            score += max(0, 3 - len(sent.split()) // 20)
            scored.append((score, sent))

        top_sentences = sorted(scored, reverse=True)[:2]
        ordered = [s for _, s in sorted(
            [(sentences.index(s), s) for _, s in top_sentences]
        )]
        summary = ". ".join(ordered) + "."

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
        extracted_fields={},
        error=""
    )

    result = pipeline.invoke(initial_state)

    return {
        "doc_id": result["doc_id"],
        "doc_type": result["doc_type"],
        "confidence": result["confidence"],
        "entities": result["entities"],
        "summary": result["summary"],
        "similar_docs": result["similar_docs"],
        "extracted_fields": result["extracted_fields"]
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