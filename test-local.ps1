$PROJECT = "submission"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

function Cleanup {
    Write-Host ""
    Write-Host "Parando containers..."
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    docker compose -p $PROJECT -f "$SCRIPT_DIR\docker-compose.yml" down --remove-orphans
    $ErrorActionPreference = $prev
}

Write-Host "=== Build e start dos containers (aguarda healthchecks) ==="
docker compose -p $PROJECT -f "$SCRIPT_DIR\docker-compose.yml" up --build --wait
if ($LASTEXITCODE -ne 0) {
    Write-Host "FALHOU: docker compose up retornou exit code $LASTEXITCODE"
    Cleanup
    exit 1
}

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "=== Testando POST /fraud-score na porta 9999 ==="

$payload = '{"id":"test-001","transaction":{"amount":100.0,"installments":1,"requested_at":"2026-05-04T12:00:00Z"},"customer":{"avg_amount":150.0,"tx_count_24h":2,"known_merchants":["merchant-abc"]},"merchant":{"id":"merchant-abc","mcc":"5411","avg_amount":120.0},"terminal":{"is_online":false,"card_present":true,"km_from_home":1.5},"last_transaction":{"timestamp":"2026-05-04T11:00:00Z","km_from_current":0.5}}'

$wc = New-Object System.Net.WebClient
$wc.Headers["Content-Type"] = "application/json"
try {
    $body = $wc.UploadString("http://127.0.0.1:9999/fraud-score", $payload)
    $httpCode = 200
} catch [System.Net.WebException] {
    $httpCode = [int]$_.Exception.Response.StatusCode
    $body = $_.Exception.Message
} catch {
    Write-Host "FALHOU ao conectar: $_"
    Write-Host ""
    Write-Host "=== Logs dos containers ==="
    docker compose -p $PROJECT -f "$SCRIPT_DIR\docker-compose.yml" logs
    Cleanup
    exit 1
}

Write-Host "HTTP $httpCode"
Write-Host "Body: $body"

if ($httpCode -eq 200) {
    Write-Host ""
    Write-Host "PASSOU - container funcionando corretamente"
    Cleanup
} else {
    Write-Host ""
    Write-Host "FALHOU - HTTP $httpCode inesperado"
    Write-Host ""
    Write-Host "=== Logs dos containers ==="
    docker compose -p $PROJECT -f "$SCRIPT_DIR\docker-compose.yml" logs
    Cleanup
    exit 1
}
