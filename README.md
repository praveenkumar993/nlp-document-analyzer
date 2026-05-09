# 🔬 DocuLens — NLP Document Analyzer

> An end-to-end multi-task NLP pipeline that automatically analyzes business documents — invoices, emails, support tickets, and news articles — extracting structured information, classifying document type, generating summaries, and enabling semantic search.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

---

## 🎯 Problem Statement

Business professionals deal with hundreds of documents daily — invoices, support tickets, emails, news articles. Manually reading each one to extract key information is slow and error-prone. DocuLens automates this: paste or upload any document and get structured insights in seconds.

---

## ✨ What It Does

| Input | Output |
|---|---|
| Invoice PDF/text | Invoice number, vendor, client, total, dates |
| Business email | Subject, sender, intent classification |
| Support ticket | Priority, issue type, ticket ID, status |
| News article | Named entities, classification, summary |

**For all document types:**
- 🏷️ Named Entity Recognition — People, Organizations, Locations
- 📊 Document Classification — Invoice / Email / Support Ticket / Business / World / Sports / Sci-Tech
- 📝 Intelligent Summarization — document-type aware, never copies input
- 🔍 Semantic Search — find similar documents by meaning
- ⬇️ Export results as JSON or CSV

---

## 🏗️ Architecture

```
Input (Text / PDF / TXT)
         │
         ▼
┌─────────────────────────────────┐
│         LangGraph Pipeline      │
│                                 │
│  Preprocessor                   │
│       ↓                         │
│  NER (DistilBERT fine-tuned)    │  ← trained on WikiANN
│       ↓                         │
│  Classifier (BERT-base)         │  ← trained on AG News
│       ↓                         │
│  Rule-based Field Extractor     │  ← per document type
│       ↓                         │
│  Smart Summarizer               │  ← document-type aware
│       ↓                         │
│  Vector Store (ChromaDB)        │  ← semantic search
└─────────────────────────────────┘
         │
         ▼
   FastAPI Backend
         │
         ▼
   Streamlit Frontend (DocuLens UI)
```

---

## 🤖 Models

| Model | Task | Dataset | Performance |
|---|---|---|---|
| DistilBERT (fine-tuned) | Named Entity Recognition | WikiANN English | **F1: 92.63%** |
| BERT-base (fine-tuned) | Document Classification | AG News (4 classes) | **F1: 92.77%** |
| all-MiniLM-L6-v2 | Semantic Embeddings | — | 22MB, fast |

Both models were **trained from scratch** using custom PyTorch training loops with:
- Gradient accumulation
- Mixed precision (fp16)
- MLflow experiment tracking
- Best model checkpoint saving

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch, HuggingFace Transformers |
| NLP Pipeline | LangGraph, spaCy |
| Semantic Search | ChromaDB, Sentence Transformers |
| Backend API | FastAPI, Uvicorn |
| Database | SQLite |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Deployment | Render |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker (optional)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/praveenkumar993/nlp-document-analyzer.git
cd nlp-document-analyzer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Set up environment variables
cp .env.example .env
# Add your HF_TOKEN to .env

# 5. Download model weights
# Models are not included in the repo due to size.
# Download from HuggingFace Hub or train using provided scripts.

# 6. Start the backend
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# 7. Start the frontend (new terminal)
streamlit run src/app.py
```

Open **http://localhost:8501** in your browser.

---

### Docker Setup

```bash
# Build and run everything
docker-compose up --build

# API → http://localhost:8000
# UI  → http://localhost:8501
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/analyze` | Analyze a document |
| GET | `/documents` | Get all analyzed documents |
| POST | `/search` | Semantic search |
| GET | `/stats` | System statistics |

### Example Request

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Invoice #INV-3337 from DEMO Invoices. Total Due: $93.50. Due Date: January 31, 2016."}'
```

### Example Response

```json
{
  "doc_id": "13597789",
  "doc_type": "Invoice",
  "confidence": 1.0,
  "entities": [
    {"text": "DEMO Invoices", "type": "ORG"}
  ],
  "summary": "Invoice INV-3337 issued by DEMO Invoices. Total amount due: $93.50, payment deadline: January 31, 2016.",
  "extracted_fields": {
    "invoice_number": "INV-3337",
    "total_due": "$93.50",
    "due_date": "January 31, 2016"
  }
}
```

---

## 🏋️ Training Your Own Models

### NER Model (DistilBERT)
```bash
python src/train_ner.py
# Trains on WikiANN English dataset
# Saves to models/ner_model/
# Expected F1: ~92%
```

### Classifier (BERT-base)
```bash
python src/train_classifier.py
# Trains on AG News dataset
# Saves to models/classifier_model/
# Expected F1: ~93%
```

For faster training use Google Colab with T4 GPU (free).

---

## 📁 Project Structure

```
nlp-document-analyzer/
├── src/
│   ├── pipeline.py          # LangGraph NLP pipeline
│   ├── api.py               # FastAPI backend
│   ├── app.py               # Streamlit frontend
│   ├── train_ner.py         # NER training script
│   └── train_classifier.py  # Classifier training script
├── models/
│   ├── ner_model/           # Fine-tuned DistilBERT (gitignored)
│   └── classifier_model/    # Fine-tuned BERT-base (gitignored)
├── notebooks/
│   └── eda.ipynb            # Dataset EDA
├── data/
│   ├── documents.db         # SQLite database
│   └── eda_plots.png        # EDA visualizations
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── render.yaml
├── requirements.txt
└── README.md
```

---

## 🌐 Deployment on Render

1. Push your code to GitHub
2. Go to [render.com](https://render.com) and connect your repository
3. Render auto-detects `render.yaml` and deploys both services
4. Add environment variables in Render dashboard:
   - `HF_TOKEN` — your HuggingFace token

> **Note:** Free tier on Render sleeps after 15 minutes of inactivity. First request after sleep may take ~30 seconds.

---

## ⚠️ Known Limitations

- NER performs best on news/article-style text — trained on Wikipedia data
- PDF extraction works only on text-based PDFs, not scanned image PDFs
- Summarization is extractive for general documents — not abstractive
- Free tier Render deployment has cold start delays

---

## 📈 Results Summary

```
NER Model (DistilBERT fine-tuned on WikiANN):
  Precision: 0.89  Recall: 0.88  F1: 0.9263

Classifier (BERT-base fine-tuned on AG News):
  Accuracy: 92.78%  F1: 0.9277
  World: 0.94  Sports: 0.98  Business: 0.89  Sci/Tech: 0.90
```

---

## 👨‍💻 Author

**Praveen Kumar**
- GitHub: [@praveenkumar993](https://github.com/praveenkumar993)

---

## 📄 License

MIT License — feel free to use this project for learning and portfolio purposes.
