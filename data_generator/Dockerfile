FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the live traffic generator against the integrated Paperless stack.
# Callers must pass --paperless-token <token> or set $PAPERLESS_TOKEN.
# The stub API is still shipped in this image; to run it instead, override
# the CMD:  `docker run ... data-generator uvicorn stub_api:app --host 0.0.0.0 --port 8000`
ENTRYPOINT ["python", "generator.py"]
CMD ["--rate", "2.0", "--duration", "300"]
