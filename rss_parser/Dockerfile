FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей для парсинга
RUN apt-get update && apt-get install -y \
    gcc \
    libxml2-dev \
    libxslt1-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Обновление сертификатов
RUN update-ca-certificates

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rss_parser.py .
COPY rss_scheduler.py .
COPY parsers/ ./parsers/

RUN mkdir -p input output

ENV PYTHONUNBUFFERED=1

CMD ["python", "rss_scheduler.py"]