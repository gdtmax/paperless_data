COMPOSE = docker compose -f docker/docker-compose.yaml

.PHONY: up down restart status logs clean
.PHONY: up-db up-storage up-stream up-vector
.PHONY: check-pg check-minio check-redpanda check-qdrant
.PHONY: ingest ingest-iam ingest-squad augment-iam

# ── Lifecycle ─────────────────────────────────

up:
	$(COMPOSE) up -d
	@echo ""
	@echo "  MinIO Console   ->  http://localhost:9001  (admin / paperless_minio)"
	@echo "  Adminer (PG)    ->  http://localhost:5050  (user / paperless_postgres)"
	@echo "  Redpanda UI     ->  http://localhost:8090"
	@echo "  Qdrant          ->  http://localhost:6333/dashboard"

down:
	$(COMPOSE) down

restart: down up

status:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f

clean:
	$(COMPOSE) down -v

# ── Bring up individual layers ────────────────

up-db:
	$(COMPOSE) up -d postgres adminer

up-storage:
	$(COMPOSE) up -d minio minio-init

up-stream:
	$(COMPOSE) up -d redpanda redpanda-init redpanda-console

up-vector:
	$(COMPOSE) up -d qdrant

# ── Ingestion Pipeline ────────────────────────

ingest: ingest-iam ingest-squad augment-iam
	@echo "Full ingestion pipeline complete."

ingest-iam:
	docker build -t paperless-ingest ./ingestion
	docker run --rm --network docker_default \
		-e MINIO_ENDPOINT=minio:9000 \
		-e MINIO_ACCESS_KEY=admin \
		-e MINIO_SECRET_KEY=paperless_minio \
		paperless-ingest python ingest_iam.py

ingest-squad:
	docker build -t paperless-ingest ./ingestion
	docker run --rm --network docker_default \
		-e MINIO_ENDPOINT=minio:9000 \
		-e MINIO_ACCESS_KEY=admin \
		-e MINIO_SECRET_KEY=paperless_minio \
		paperless-ingest python ingest_squad.py

augment-iam:
	docker build -t paperless-ingest ./ingestion
	docker run --rm --network docker_default \
		-e MINIO_ENDPOINT=minio:9000 \
		-e MINIO_ACCESS_KEY=admin \
		-e MINIO_SECRET_KEY=paperless_minio \
		paperless-ingest python augment_iam.py

# ── Health checks ─────────────────────────────

check-pg:
	docker exec postgres pg_isready -U user -d paperless

check-minio:
	curl -sf http://localhost:9000/minio/health/live && echo "MinIO OK"

check-redpanda:
	docker exec redpanda rpk cluster health

check-qdrant:
	curl -sf http://localhost:6333/healthz && echo "Qdrant OK"
