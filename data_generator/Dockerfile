FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make loop.sh executable
RUN chmod +x /app/loop.sh

# Default: one-shot generator run via python generator.py.
#   docker run ... -e PAPERLESS_TOKEN=... data-generator --rate 1.0 --duration 600
#
# For continuous unattended traffic, override the entrypoint:
#   docker run ... --entrypoint /app/loop.sh -e PAPERLESS_URL=... -e PAPERLESS_TOKEN=... data-generator
# The compose.yml does this for you.
ENTRYPOINT ["python", "generator.py"]
CMD ["--rate", "2.0", "--duration", "300"]
