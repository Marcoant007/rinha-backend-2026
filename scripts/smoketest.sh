#!/usr/bin/env bash
# smoketest.sh — validates the API is responding correctly.
# Usage: ./scripts/smoketest.sh [host:port]   default: localhost:9999
set -euo pipefail

HOST="${1:-localhost:9999}"
BASE="http://$HOST"
PASS=0
FAIL=0

green() { printf "\033[32m✓\033[0m %s\n" "$*"; }
red()   { printf "\033[31m✗\033[0m %s\n" "$*"; }

check() {
  local desc="$1"
  local expected="$2"
  local actual="$3"
  if echo "$actual" | grep -q "$expected"; then
    green "$desc"
    PASS=$((PASS+1))
  else
    red "$desc — expected '$expected' in: $actual"
    FAIL=$((FAIL+1))
  fi
}

echo "=== Smoke test → $BASE ==="

# 1. /ready should return 200 (or 503 if still loading)
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/ready" || echo "000")
if [ "$STATUS" = "200" ] || [ "$STATUS" = "503" ]; then
  green "/ready → HTTP $STATUS (OK)"
  PASS=$((PASS+1))
else
  red "/ready → HTTP $STATUS (unexpected)"
  FAIL=$((FAIL+1))
fi

# Wait for the index to load (poll /ready up to 30 s)
echo "Waiting for index to load..."
for i in $(seq 1 30); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/ready" || echo "000")
  if [ "$STATUS" = "200" ]; then
    green "Index ready after ${i}s"
    break
  fi
  sleep 1
done

# 2. Normal transaction (low amount, known merchant) → should be approved
LEGIT_BODY='{
  "id":"tx-001",
  "transaction":{"amount":50.0,"installments":1,"requested_at":"2024-06-10T14:00:00Z"},
  "customer":{"avg_amount":60.0,"tx_count_24h":1,"known_merchants":["merchant-xyz"]},
  "merchant":{"id":"merchant-xyz","mcc":"5411","avg_amount":55.0},
  "terminal":{"is_online":false,"card_present":true,"km_from_home":2.0},
  "last_transaction":{"timestamp":"2024-06-10T13:30:00Z","km_from_current":1.0}
}'
RESP=$(curl -s -X POST -H "Content-Type: application/json" -d "$LEGIT_BODY" "$BASE/fraud-score")
check "Legit tx has 'approved'" '"approved"' "$RESP"
check "Legit tx has 'fraud_score'" '"fraud_score"' "$RESP"

# 3. Suspicious transaction (gambling MCC, high amount, unknown merchant, far from home)
FRAUD_BODY='{
  "id":"tx-002",
  "transaction":{"amount":9500.0,"installments":1,"requested_at":"2024-06-10T03:00:00Z"},
  "customer":{"avg_amount":60.0,"tx_count_24h":15,"known_merchants":[]},
  "merchant":{"id":"casino-999","mcc":"7995","avg_amount":8000.0},
  "terminal":{"is_online":true,"card_present":false,"km_from_home":950.0},
  "last_transaction":{"timestamp":"2024-06-10T02:00:00Z","km_from_current":900.0}
}'
RESP=$(curl -s -X POST -H "Content-Type: application/json" -d "$FRAUD_BODY" "$BASE/fraud-score")
check "Fraud tx has 'approved'" '"approved"' "$RESP"
check "Fraud tx has 'fraud_score'" '"fraud_score"' "$RESP"

# 4. Malformed JSON → must return HTTP 200 with fallback (not 4xx/5xx)
MALFORMED='{"bad json'
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d "$MALFORMED" "$BASE/fraud-score")
if [ "$HTTP_CODE" = "200" ]; then
  green "Malformed JSON → HTTP 200 (fallback, not error)"
  PASS=$((PASS+1))
else
  red "Malformed JSON → HTTP $HTTP_CODE (should be 200)"
  FAIL=$((FAIL+1))
fi

# 5. GET /fraud-score → 405 Method Not Allowed
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/fraud-score")
if [ "$HTTP_CODE" = "405" ]; then
  green "GET /fraud-score → HTTP 405 (correct)"
  PASS=$((PASS+1))
else
  red "GET /fraud-score → HTTP $HTTP_CODE (expected 405)"
  FAIL=$((FAIL+1))
fi

# 6. Unknown path → 404
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/unknown")
if [ "$HTTP_CODE" = "404" ]; then
  green "GET /unknown → HTTP 404 (correct)"
  PASS=$((PASS+1))
else
  red "GET /unknown → HTTP $HTTP_CODE (expected 404)"
  FAIL=$((FAIL+1))
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
