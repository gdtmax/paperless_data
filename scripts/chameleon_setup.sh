#!/bin/bash
# Run once after SSH into the Chameleon VM:
#   bash scripts/chameleon_setup.sh

set -e

echo "==> Installing Docker..."
curl -sSL https://get.docker.com/ | sudo sh
sudo groupadd -f docker
sudo usermod -aG docker $USER

echo "==> Cloning repo..."
# Replace with your actual repo URL
git clone https://github.com/YOUR_ORG/paperless_data ~/paperless_data
cd ~/paperless_data

echo "==> Starting data platform..."
sg docker -c "make up"

echo ""
IP=$(curl -sf https://checkip.amazonaws.com || echo "<floating-ip>")
echo "  MinIO Console   ->  http://${IP}:9001  (admin / paperless_minio)"
echo "  Adminer         ->  http://${IP}:5050"
echo "  Redpanda UI     ->  http://${IP}:8090"
echo "  Qdrant          ->  http://${IP}:6333/dashboard"
