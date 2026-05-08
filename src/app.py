import streamlit as st
import requests
import json



# ---- Page Config ----
st.set_page_config(
    page_title="NLP Document Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- API URL ----
API_URL = "http://localhost:8000"

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .entity-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .PER { background-color: #ffeb3b; color: #333; }
    .ORG { background-color: #4caf50; color: white; }
    .LOC { background-color: #2196f3; color: white; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.markdown("## 🔍 NLP Document Analyzer")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📄 Analyze Document", "📚 Document History", "🔎 Semantic Search", "📊 Stats"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Models Used:**")
st.sidebar.markdown("- DistilBERT (NER) — F1: 92.63%")
st.sidebar.markdown("- BERT-base (Classifier) — F1: 92.77%")
st.sidebar.markdown("- BART-large-CNN (Summarizer)")
st.sidebar.markdown("- MiniLM-L6 (Embeddings)")


# ---- Helper Functions ----
def analyze_document(text: str):
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={"text": text},
            timeout=60
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, response.json().get("detail", "Unknown error")
    except Exception as e:
        return None, str(e)


def get_documents():
    try:
        response = requests.get(f"{API_URL}/documents", timeout=30)
        if response.status_code == 200:
            return response.json(), None
        return None, "Failed to fetch documents"
    except Exception as e:
        return None, str(e)


def search_documents(query: str, n_results: int = 3):
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query, "n_results": n_results},
            timeout=30
        )
        if response.status_code == 200:
            return response.json(), None
        return None, "Search failed"
    except Exception as e:
        return None, str(e)


def get_stats():
    try:
        response = requests.get(f"{API_URL}/stats", timeout=30)
        if response.status_code == 200:
            return response.json(), None
        return None, "Failed to fetch stats"
    except Exception as e:
        return None, str(e)


def render_entities(entities):
    if not entities:
        st.info("No entities found")
        return
    html = ""
    for entity in entities:
        entity_type = entity["type"]
        css_class = entity_type if entity_type in ["PER", "ORG", "LOC"] else "ORG"
        html += f'<span class="entity-tag {css_class}">{entity["text"]} [{entity_type}]</span> '
    st.markdown(html, unsafe_allow_html=True)

# ---- Pages ----

if page == "📄 Analyze Document":
    st.markdown('<div class="main-header">🔍 NLP Document Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze invoices, emails, support tickets, PDFs or any text document</div>', unsafe_allow_html=True)

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📝 Paste Text", "📎 Upload File"],
        horizontal=True
    )

    text_input = ""

    if input_method == "📝 Paste Text":
        text_input = st.text_area(
            "Paste your document here:",
            height=200,
            placeholder="Paste any invoice, email, support ticket, news article or any text..."
        )

    elif input_method == "📎 Upload File":
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["txt", "pdf"],
            help="Supports PDF and TXT files"
        )

        if uploaded_file is not None:
            file_type = uploaded_file.name.split(".")[-1].lower()

            if file_type == "txt":
                text_input = uploaded_file.read().decode("utf-8", errors="ignore")
                st.success(f"✅ Text file loaded — {len(text_input)} characters")
                st.text_area("Extracted Text Preview:", text_input[:500], height=150)

            elif file_type == "pdf":
                import pdfplumber
                import io
                with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                    text_input = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_input += page_text + "\n"

                if text_input.strip():
                    st.success(f"✅ PDF loaded — {len(pdf.pages)} pages, {len(text_input)} characters")
                    st.text_area("Extracted Text Preview:", text_input[:500], height=150)
                else:
                    st.error("Could not extract text from PDF. It may be a scanned image PDF.")

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
    with col2:
        st.markdown("")

    if analyze_btn:
        if not text_input.strip():
            st.error("Please enter or upload a document to analyze")
        else:
            with st.spinner("Analyzing document... this may take 10-20 seconds"):
                result, error = analyze_document(text_input)

            if error:
                st.error(f"Error: {error}")
            else:
                st.success("Document analyzed successfully!")

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Document Type", result["doc_type"])
                with col2:
                    st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                with col3:
                    st.metric("Entities Found", len(result["entities"]))
                with col4:
                    st.metric("Document ID", result["doc_id"])

                st.markdown("---")

                # Extracted Fields based on document type
                if result.get("extracted_fields"):
                    fields = result["extracted_fields"]
                    doc_type = result["doc_type"]

                    if doc_type == "Invoice":
                        st.markdown("### 🧾 Invoice Details")
                        cols = st.columns(2)
                        field_items = list(fields.items())
                        for i, (key, value) in enumerate(field_items):
                            with cols[i % 2]:
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")

                    elif doc_type == "Email":
                        st.markdown("### 📧 Email Details")
                        cols = st.columns(2)
                        field_items = list(fields.items())
                        for i, (key, value) in enumerate(field_items):
                            with cols[i % 2]:
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")

                    elif doc_type == "Support Ticket":
                        st.markdown("### 🎫 Ticket Details")
                        cols = st.columns(2)
                        field_items = list(fields.items())
                        for i, (key, value) in enumerate(field_items):
                            with cols[i % 2]:
                                label = key.replace('_', ' ').title()
                                if key == "priority":
                                    color = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "Critical": "🔴"}.get(value, "⚪")
                                    st.markdown(f"**{label}:** {color} {value}")
                                else:
                                    st.markdown(f"**{label}:** {value}")

                    st.markdown("---")

                # Named Entities
                st.markdown("### 🏷️ Named Entities")
                render_entities(result["entities"])

                st.markdown("---")

                # Summary
                st.markdown("### 📝 Summary")
                st.info(result["summary"])

                st.markdown("---")

                # Similar Documents
                st.markdown("### 🔗 Similar Documents")
                if result["similar_docs"]:
                    for doc in result["similar_docs"]:
                        with st.expander(f"📄 {doc['doc_type']} document"):
                            st.write(doc["text"])
                            st.caption(f"Summary: {doc['summary']}")
                else:
                    st.info("No similar documents found yet. Analyze more documents to enable similarity search.")

                # Export Results
                st.markdown("---")
                st.markdown("### 📥 Export Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=json.dumps(result, indent=2),
                        file_name=f"analysis_{result['doc_id']}.json",
                        mime="application/json"
                    )
                with col2:
                    import csv
                    import io
                    csv_buffer = io.StringIO()
                    writer = csv.writer(csv_buffer)
                    writer.writerow(["Field", "Value"])
                    writer.writerow(["Document ID", result["doc_id"]])
                    writer.writerow(["Document Type", result["doc_type"]])
                    writer.writerow(["Confidence", f"{result['confidence']*100:.1f}%"])
                    writer.writerow(["Summary", result["summary"]])
                    for entity in result["entities"]:
                        writer.writerow([f"Entity ({entity['type']})", entity["text"]])
                    if result.get("extracted_fields"):
                        for k, v in result["extracted_fields"].items():
                            writer.writerow([k.replace("_", " ").title(), v])
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"analysis_{result['doc_id']}.csv",
                        mime="text/csv"
                    )

                # Raw JSON
                with st.expander("🔧 Raw JSON Response"):
                    st.json(result)

# Page 2 — Document History
elif page == "📚 Document History":
    st.markdown("## 📚 Document History")
    st.markdown("All previously analyzed documents stored in SQLite")

    if st.button("🔄 Refresh", type="primary"):
        st.rerun()

    data, error = get_documents()

    if error:
        st.error(f"Error: {error}")
    elif not data or data["total"] == 0:
        st.info("No documents analyzed yet. Go to Analyze Document to get started!")
    else:
        st.metric("Total Documents", data["total"])
        st.markdown("---")

        for doc in data["documents"]:
            with st.expander(f"📄 [{doc['doc_type']}] {doc['text_preview'][:80]}..."):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Type:** {doc['doc_type']}")
                with col2:
                    st.markdown(f"**Confidence:** {doc['confidence']*100:.1f}%")
                with col3:
                    st.markdown(f"**Date:** {doc['created_at']}")

                st.markdown("**Entities:**")
                render_entities(doc["entities"])

                st.markdown("**Summary:**")
                st.info(doc["summary"])


# Page 3 — Semantic Search
elif page == "🔎 Semantic Search":
    st.markdown("## 🔎 Semantic Search")
    st.markdown("Search across all analyzed documents by meaning, not just keywords")

    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g. company earnings, sports results, technology news..."
    )

    n_results = st.slider("Number of results", min_value=1, max_value=10, value=3)

    if st.button("🔍 Search", type="primary"):
        if not query.strip():
            st.error("Please enter a search query")
        else:
            with st.spinner("Searching..."):
                results, error = search_documents(query, n_results)

            if error:
                st.error(f"Error: {error}")
            elif not results or results["total"] == 0:
                st.info("No results found. Analyze more documents first!")
            else:
                st.success(f"Found {results['total']} results for: '{query}'")
                st.markdown("---")

                for i, doc in enumerate(results["results"]):
                    with st.expander(f"Result {i+1} — {doc['doc_type']}"):
                        st.markdown("**Preview:**")
                        st.write(doc["text_preview"])
                        st.markdown("**Summary:**")
                        st.info(doc["summary"])


# Page 4 — Stats
elif page == "📊 Stats":
    st.markdown("## 📊 Pipeline Statistics")

    if st.button("🔄 Refresh Stats", type="primary"):
        st.rerun()

    stats, error = get_stats()

    if error:
        st.error(f"Error: {error}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Documents Analyzed", stats["total_documents"])
        with col2:
            st.metric("Vectors in ChromaDB", stats["vector_store_count"])

        st.markdown("---")
        st.markdown("### Documents by Type")

        if stats["documents_by_type"]:
            for doc_type, count in stats["documents_by_type"].items():
                st.progress(
                    count / stats["total_documents"],
                    text=f"{doc_type}: {count} documents"
                )
        else:
            st.info("No documents analyzed yet!")

        st.markdown("---")
        st.markdown("### Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**NER Model (DistilBERT)**")
            st.progress(0.9263, text="F1 Score: 92.63%")
        with col2:
            st.markdown("**Classifier (BERT-base)**")
            st.progress(0.9277, text="F1 Score: 92.77%")