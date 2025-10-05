FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('deepvk/USER-bge-m3')"
RUN python -c "from natasha import NewsEmbedding, NewsMorphTagger, NewsNERTagger; \
    emb = NewsEmbedding(); \
    morph = NewsMorphTagger(emb); \
    ner = NewsNERTagger(emb)"
COPY . .

RUN mkdir -p /app/rss_parser/output

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]