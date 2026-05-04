#!/bin/bash
set -euo pipefail

PROJECT="submission"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
  echo ""
  echo "Parando containers..."
  docker compose -p "$PROJECT" -f "$SCRIPT_DIR/docker-compose.yml" down --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Build e start dos containers (aguarda healthchecks) ==="
docker compose -p "$PROJECT" -f "$SCRIPT_DIR/docker-compose.yml" up --build --wait
echo ""

echo "=== Testando POST /fraud-score na porta 9999 ==="
PAYLOAD='{
  "id": "test-001",
  "transaction": {"amount": 100.0, "installments": 1, "requested_at": "2026-05-04T12:00:00Z"},
  "customer": {"avg_amount": 150.0, "tx_count_24h": 2, "known_merchants": ["merchant-abc"]},
  "merchant": {"id": "merchant-abc", "mcc": "5411", "avg_amount": 120.0},
  "terminal": {"is_online": false, "card_present": true, "km_from_home": 1.5},
  "last_transaction": {"timestamp": "2026-05-04T11:00:00Z", "km_from_current": 0.5}
}'

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:9999/fraud-score \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -1)

echo "HTTP $HTTP_CODE"
echo "Body: $BODY"

if [ "$HTTP_CODE" = "200" ]; then
  echo ""
  echo "PASSOU - container funcionando corretamente"
else
  echo ""
  echo "FALHOU - HTTP $HTTP_CODE inesperado"
  echo ""
  echo "=== Logs dos containers ==="
  docker compose -p "$PROJECT" -f "$SCRIPT_DIR/docker-compose.yml" logs
  exit 1
fi
