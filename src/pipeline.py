import torch
import re
import os
import json
from chromadb.config import Settings
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForTokenClassification,
    BertTokenizerFast,
    BertForSequenceClassification,
    pipeline as hf_pipeline
)
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ---- Labels ----
NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
CLASSIFIER_LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# ---- Device ----
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

print("Loading DistilBART Summarizer...")
bart_summarizer = hf_pipeline(
    task="summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1  # CPU
)

print("Setting up ChromaDB...")

chroma_client = chromadb.Client(
    Settings(
        anonymized_telemetry=False
    )
)

collection = chroma_client.get_or_create_collection("documents")

print("All models loaded!")


# ================================================================
# DOCUMENT TYPE DETECTION — Rule-based first, ML fallback
# ================================================================
def detect_document_type(text: str) -> str:
    text_lower = text.lower()

    invoice_keywords = [
        "invoice", "total due", "payment due", "invoice number",
        "bill to", "ship to", "subtotal", "amount due",
        "purchase order", "invoice date", "due date", "receipt",
        "gstin", "hsn", "tax invoice", "proforma"
    ]
    email_keywords = [
        "dear", "regards", "sincerely", "best regards",
        "subject:", "please find", "attached", "let me know",
        "thank you for", "hi ", "hello ", "greetings", "warm regards"
    ]
    ticket_keywords = [
        "ticket", "priority", "bug", "assigned to",
        "reported by", "severity", "incident", "resolve",
        "support request", "status:", "issue #", "case #",
        "escalation", "sla", "helpdesk"
    ]

    invoice_score = sum(1 for kw in invoice_keywords if kw in text_lower)
    email_score = sum(1 for kw in email_keywords if kw in text_lower)
    ticket_score = sum(1 for kw in ticket_keywords if kw in text_lower)

    scores = {
        "Invoice": invoice_score,
        "Email": email_score,
        "Support Ticket": ticket_score
    }
    max_type = max(scores, key=scores.get)
    max_score = scores[max_type]

    return max_type if max_score >= 2 else "General"


# ================================================================
# FIELD EXTRACTORS — per document type
# ================================================================
def extract_invoice_fields(text: str) -> dict:
    fields = {}

    patterns = {
       "invoice_number":  r'invoice\s*(?:number|#|no\.?)\s*[:\-]?\s*([A-Z]{2,10}-\d{2,6}(?:-[A-Z0-9]+)*)',
        "order_number":   r'order\s*(?:number|#|no\.?)?\s*[:\-]?\s*([A-Z0-9\-]{3,})',
        "total_due":      r'total\s*due\s*[:\-]?\s*(?:rs\.?|inr|₹|\$)?\s*([\d,]+\.\d+|[\d,]+)',
        "tax":            r'\btax\b\s*[:\-]?\s*(?:rs\.?|inr|₹|\$)?\s*([\d,]+\.?\d*)',
        "sub_total":      r'sub\s*total\s*[:\-]?\s*(?:rs\.?|inr|₹|\$)?\s*([\d,]+\.?\d*)',
        "invoice_date": r'invoice\s*date\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},?\s*\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        "due_date": r'due\s*date\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},?\s*\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
    }

    for field, pattern in patterns.items():
        if field == "total_due":
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                val = matches[-1].strip()
            else:
                continue
        else:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            val = match.group(1).strip()
        if field in ["total_due", "tax", "sub_total"]:
            val = f"₹{val}" if any(c in text for c in ["₹", "INR", "Rs", "Crore", "Lakh"]) else f"${val}"
        fields[field] = val

    # Vendor Extraction — smarter multiline block parsing
    from_block = re.search(
        r'from\s*:\s*(.*?)bill\s*to\s*:',
        text,
        re.IGNORECASE | re.DOTALL
    )

    if from_block:
        block = from_block.group(1)

        # Split lines and clean
        lines = [
            l.strip()
            for l in block.split("\n")
            if len(l.strip()) > 3
        ]

        # Look for likely company names
        vendor_candidates = [
            l for l in lines
            if any(word in l.lower() for word in [
                "limited",
                "ltd",
                "pvt",
                "corp",
                "solutions",
                "technologies",
                "services",
                "systems"
            ])
        ]

        if vendor_candidates:
            vendor = vendor_candidates[0]

            # Remove trailing invoice/order/date text
            vendor = re.split(
                r'invoice\s*number|order\s*number|invoice\s*date|due\s*date',
                vendor,
                flags=re.IGNORECASE
            )[0].strip()

            fields["vendor"] = vendor

    # Client — line after "To:" or "Bill To:"
    to_match = re.search(r'(?:^|\n)\s*(?:bill\s*to|to)\s*:\s*\n?\s*(.+)', text, re.IGNORECASE)
    if to_match:
        fields["client"] = to_match.group(1).strip()

    # GSTIN
    gstin_match = re.search(r'gstin\s*[:\-]?\s*([A-Z0-9]{15})', text, re.IGNORECASE)
    if gstin_match:
        fields["gstin"] = gstin_match.group(1).strip()

    return fields


def extract_email_fields(text: str) -> dict:
    fields = {}
    lines = text.strip().split("\n")

    for line in lines[:10]:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("subject:"):
            fields["subject"] = stripped.split(":", 1)[1].strip()
        elif lower.startswith("from:"):
            fields["from"] = stripped.split(":", 1)[1].strip()
        elif lower.startswith("to:"):
            fields["to"] = stripped.split(":", 1)[1].strip()
        elif lower.startswith("cc:"):
            fields["cc"] = stripped.split(":", 1)[1].strip()
        elif lower.startswith("date:"):
            fields["date"] = stripped.split(":", 1)[1].strip()

    text_lower = text.lower()
    if any(w in text_lower for w in ["complaint", "unhappy", "disappointed", "not satisfied", "issue with", "problem with"]):
        fields["intent"] = "Complaint"
    elif any(w in text_lower for w in ["follow up", "following up", "checking in", "any update", "status update"]):
        fields["intent"] = "Follow Up"
    elif any(w in text_lower for w in ["thank you", "thanks", "appreciate", "grateful", "well received"]):
        fields["intent"] = "Appreciation"
    elif any(w in text_lower for w in ["please find", "attached", "quotation", "proposal", "request", "partnership"]):
        fields["intent"] = "Request"
    else:
        fields["intent"] = "General"

    return fields


def extract_ticket_fields(text: str) -> dict:
    fields = {}

    ticket_match = re.search(
    r'(?:ticket|case|issue)\s*(?:id|number|#)?\s*[:\-]?\s*([A-Z]{1,5}-\d{2,6}(?:-\d{1,6})?)',
    text,
    re.IGNORECASE
    )
    if ticket_match:
        fields["ticket_id"] = ticket_match.group(1).strip()

    priority_match = re.search(
    r'priority\s*(?:[:\-]?\s*)?(low|medium|high|critical|urgent|p0|p1|p2|p3)',text,re.IGNORECASE)
    if priority_match:
        fields["priority"] = priority_match.group(1).capitalize()
    else:
        text_lower = text.lower()
        if any(w in text_lower for w in ["urgent", "critical", "asap", "immediately", "blocker", "p0", "p1"]):
            fields["priority"] = "High"
        elif any(w in text_lower for w in ["low priority", "minor", "whenever possible", "p3", "p4"]):
            fields["priority"] = "Low"
        else:
            fields["priority"] = "Medium"

    status_match = re.search(r'status\s*(?:[:\-]?\s*)?(open|closed|pending|resolved|in\s*progress)',text,re.IGNORECASE)
    fields["status"] = status_match.group(1).strip().capitalize() if status_match else "Open"

    text_lower = text.lower()
    if any(w in text_lower for w in ["login", "password", "access", "authentication", "permission", "ldap", "sso"]):
        fields["issue_type"] = "Access Issue"
    elif any(w in text_lower for w in ["crash", "error", "bug", "not working", "broken", "failed", "exception"]):
        fields["issue_type"] = "Bug Report"
    elif any(w in text_lower for w in ["slow", "performance", "timeout", "latency", "hang", "freeze"]):
        fields["issue_type"] = "Performance Issue"
    elif any(w in text_lower for w in ["install", "setup", "configure", "deployment", "update", "upgrade"]):
        fields["issue_type"] = "Installation Issue"
    else:
        fields["issue_type"] = "General Issue"

    assigned_match = re.search(
        r'assigned\s*to\s*[:\-]?\s*([A-Za-z\s]+?)(?:department|\n|$)',
        text,
        re.IGNORECASE
    )
    if assigned_match:
        fields["assigned_to"] = assigned_match.group(1).strip()

    reported_match = re.search(
        r'reported\s*by\s*[:\-]?\s*([A-Za-z\s]+?)(?:date|\n|$)',
        text,
        re.IGNORECASE
    )
    if reported_match:
        fields["reported_by"] = reported_match.group(1).strip()

    return fields


def extract_fields(doc_type: str, text: str) -> dict:
    if doc_type == "Invoice":
        return extract_invoice_fields(text)
    elif doc_type == "Email":
        return extract_email_fields(text)
    elif doc_type == "Support Ticket":
        return extract_ticket_fields(text)
    return {}


# ================================================================
# ENTITY CLEANING
# ================================================================
NOISE_ENTITIES = {
    "office supplies", "web design", "sample", "services", "payment",
    "invoice", "total", "tax", "sub", "amount", "date", "number",
    "dear", "regards", "sincerely", "hello", "hi", "subject",
    "attached", "please", "find", "thank", "note", "items"
}

def clean_entities(entities: list) -> list:
    cleaned = []
    seen = set()
    for entity in entities:
        text = entity["text"].strip()
        if len(text) < 3:
            continue
        if text.startswith("##"):
            continue
        if text.lower() in NOISE_ENTITIES:
            continue
        if text.lower() in seen:
            continue
        # Skip pure numbers
        if re.match(r'^[\d\s\.,]+$', text):
            continue
        seen.add(text.lower())
        cleaned.append(entity)
    return cleaned


# ================================================================
# SUMMARIZER
# ================================================================
def generate_structured_summary(doc_type: str, text: str, entities: list, extracted_fields: dict) -> str:
    """
    For Invoice, Email, Support Ticket — generate precise structured summaries.
    These are deterministic and accurate. BART is NOT used here.
    """
    per_entities = [e["text"].title() for e in entities if e["type"] == "PER"]
    org_entities = [e["text"].title() for e in entities if e["type"] == "ORG"]
    loc_entities = [e["text"].title() for e in entities if e["type"] == "LOC"]

    if doc_type == "Invoice":
        vendor = extracted_fields.get("vendor", org_entities[0] if org_entities else "the vendor")
        client = extracted_fields.get("client", "the client")
        total = extracted_fields.get("total_due", "N/A")
        inv_num = extracted_fields.get("invoice_number", "N/A")
        due = extracted_fields.get("due_date", "N/A")
        inv_date = extracted_fields.get("invoice_date", "N/A")
        gstin = extracted_fields.get("gstin", "")
        gstin_str = f" (GSTIN: {gstin})" if gstin else ""
        return (
            f"Invoice {inv_num} issued by {vendor}{gstin_str} to {client} on {inv_date}. "
            f"Total amount due: {total}, payment deadline: {due}."
        )

    elif doc_type == "Email":
        subject = extracted_fields.get("subject", "")
        sender = extracted_fields.get("from", per_entities[0] if per_entities else "the sender")
        intent = extracted_fields.get("intent", "General")
        org = org_entities[0] if org_entities else ""
        loc = loc_entities[0] if loc_entities else ""
        parts = [f"{intent} email"]
        if subject:
            parts.append(f"regarding \"{subject}\"")
        if sender:
            parts.append(f"from {sender}")
        if org:
            parts.append(f"at {org}")
        if loc:
            parts.append(f"based in {loc}")
        return " ".join(parts) + "."

    elif doc_type == "Support Ticket":
        ticket_id = extracted_fields.get("ticket_id", "")
        priority = extracted_fields.get("priority", "Medium")
        issue_type = extracted_fields.get("issue_type", "General Issue")
        status = extracted_fields.get("status", "Open")
        assigned = extracted_fields.get("assigned_to", "")
        reported = extracted_fields.get("reported_by", "")
        sentences = [
            s.strip()
            for s in re.split(r'[.\n]', text)
            if len(s.strip()) > 30
        ]

        filtered_sentences = [
            s for s in sentences
            if not any(
                noise in s.lower()
                for noise in [
                    "ticket #",
                    "priority",
                    "status",
                    "assigned to",
                    "reported by",
                    "company",
                    "location"
                ]
            )
        ]

        detail = filtered_sentences[0] if filtered_sentences else ""
        summary = f"{priority} priority {issue_type}"
        if ticket_id:
            summary += f" (#{ticket_id})"
        summary += f". Status: {status}."
        if assigned:
            summary += f" Assigned to {assigned}."
        if reported:
            summary += f" Reported by {reported}."
        if detail:
            summary += f" {detail}."
        return summary

    return ""


def generate_bart_summary(text: str, entities: list) -> str:
    """
    For News/General documents — use DistilBART for abstractive summarization.
    BART is designed for news articles and works best here.
    """
    # Clean text — remove very short lines and noise
    clean_lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]
    clean_text = " ".join(clean_lines)

    # BART works best with 100-600 words
    words = clean_text.split()
    if len(words) > 500:
        clean_text = " ".join(words[:500])

    # If too short for BART — use extractive fallback
    if len(words) < 30:
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        return sentences[0] + "." if sentences else text.strip()

    try:
        result = bart_summarizer(
            clean_text,
            max_length=120,
            min_length=40,
            do_sample=False,
            truncation=True
        )
        summary = result[0]["summary_text"].strip()
        # Clean up spacing issues
        summary = re.sub(r'\s+([.,])', r'\1', summary)
        summary = re.sub(r'\s+', ' ', summary)
        return summary
    except Exception as e:
        print(f"BART summarization failed: {e}")
        # Extractive fallback
        entity_names = [e["text"].lower() for e in entities]
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 25]
        if not sentences:
            return text[:200].strip()
        scored = []
        for i, sent in enumerate(sentences):
            score = 4 if i == 0 else 0
            for ent in entity_names:
                if ent in sent.lower():
                    score += 2
            scored.append((score, i, sent))
        top = sorted(scored, reverse=True)[:2]
        ordered = [s for _, _, s in sorted(top, key=lambda x: x[1])]
        return ". ".join(ordered).strip() + "."


# ================================================================
# LANGGRAPH STATE
# ================================================================
class DocumentState(TypedDict):
    text: str
    entities: List[Dict[str, str]]
    doc_type: str
    confidence: float
    summary: str
    doc_id: str
    extracted_fields: Dict[str, Any]
    error: str


# ================================================================
# PIPELINE NODES
# ================================================================
def preprocess(state: DocumentState) -> DocumentState:
    text = state["text"].strip()
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    state["text"] = text
    state["doc_id"] = str(abs(hash(text)))[:8]
    return state


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
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    entities = []
    current_entity = None

    for token, pred in zip(tokens, predictions):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        label = NER_LABELS[pred]
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"text": token, "type": label[2:]}
        elif label.startswith("I-") and current_entity:
            if token.startswith("##"):
                current_entity["text"] += token[2:]
            else:
                current_entity["text"] += " " + token
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    state["entities"] = clean_entities(entities)
    return state


def run_classifier(state: DocumentState) -> DocumentState:
    text = state["text"]
    rule_type = detect_document_type(text)

    if rule_type != "General":
        state["doc_type"] = rule_type
        state["confidence"] = 1.0
    else:
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
        state["doc_type"] = CLASSIFIER_LABELS[pred_id]
        state["confidence"] = round(probs[pred_id].item(), 4)

    state["extracted_fields"] = extract_fields(state["doc_type"], text)
    return state


def run_summarizer(state: DocumentState) -> DocumentState:
    doc_type = state["doc_type"]
    text = state["text"]
    entities = state["entities"]
    extracted_fields = state["extracted_fields"]

    structured_types = ["Invoice", "Email", "Support Ticket"]

    if doc_type in structured_types:
        # Use precise structured summary — no BART needed
        # BART would garble structured summaries
        state["summary"] = generate_structured_summary(
            doc_type, text, entities, extracted_fields
        )
    else:
        # Use BART for news/general — this is what BART is designed for
        state["summary"] = generate_bart_summary(text, entities)

    return state


def run_vector_store(state: DocumentState) -> DocumentState:
    text = state["text"]
    doc_id = state["doc_id"]
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
    return state


# ================================================================
# BUILD AND RUN PIPELINE
# ================================================================
def build_pipeline():
    graph = StateGraph(DocumentState)
    graph.add_node("preprocess", preprocess)
    graph.add_node("ner", run_ner)
    graph.add_node("classifier", run_classifier)
    graph.add_node("summarizer", run_summarizer)
    graph.add_node("vector_store", run_vector_store)
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "ner")
    graph.add_edge("ner", "classifier")
    graph.add_edge("classifier", "summarizer")
    graph.add_edge("summarizer", "vector_store")
    graph.add_edge("vector_store", END)
    return graph.compile()


def analyze_document(text: str) -> Dict[str, Any]:
    p = build_pipeline()
    initial_state = DocumentState(
        text=text,
        entities=[],
        doc_type="",
        confidence=0.0,
        summary="",
        doc_id="",
        extracted_fields={},
        error=""
    )
    result = p.invoke(initial_state)
    return {
        "doc_id": result["doc_id"],
        "doc_type": result["doc_type"],
        "confidence": result["confidence"],
        "entities": result["entities"],
        "summary": result["summary"],
        "extracted_fields": result["extracted_fields"]
    }
