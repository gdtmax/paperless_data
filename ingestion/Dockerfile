FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run full ingestion pipeline
CMD ["sh", "-c", "python ingest_iam.py && python ingest_squad.py && python augment_iam.py"]
