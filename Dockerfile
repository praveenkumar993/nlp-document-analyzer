FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY src/ ./src/

RUN mkdir -p models data

# Download models from HuggingFace Hub at build time
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='praveends/doculens-ner', local_dir='models/ner_model'); \
snapshot_download(repo_id='praveends/doculens-classifier', local_dir='models/classifier_model'); \
print('Models downloaded!')"

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]