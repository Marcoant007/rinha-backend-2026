# Rinha de Backend 2026 — Detecção de Fraudes (Go)

Implementação em Go para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026), cujo tema é detecção de fraudes em transações financeiras usando busca vetorial KNN.

## Arquitetura

```
Cliente → Nginx :9999 (load balancer)
               ├── api1 :8081
               ├── api2 :8081
               └── knn  :8080 (busca vetorial)
```

| Serviço | CPU    | Memória |
|---------|--------|---------|
| knn     | 0.50   | 220 MB  |
| api1    | 0.225  | 60 MB   |
| api2    | 0.225  | 60 MB   |
| nginx   | 0.05   | 10 MB   |
| **Total** | **1.0** | **350 MB** |

## Como funciona

1. A API recebe um `POST /fraud-score` com os dados da transação
2. Transforma os dados em um vetor de **14 dimensões** (normalizado)
3. Envia o vetor ao serviço KNN
4. O KNN busca os **5 vizinhos mais próximos** entre ~3 milhões de referências (distância euclidiana)
5. `fraud_score = fraudes_encontradas / 5`
6. Retorna `{ approved: fraud_score < 0.6, fraud_score: X }`

### Vetor de 14 dimensões

| Índice | Feature                        | Normalização     |
|--------|--------------------------------|------------------|
| 0      | Valor da transação             | max: R$ 10.000   |
| 1      | Número de parcelas             | max: 12          |
| 2      | Razão valor / média do cliente | max: 10x         |
| 3      | Hora do dia                    | 0–23 → 0–1       |
| 4      | Dia da semana                  | 0–6 → 0–1        |
| 5      | Minutos desde última transação | max: 1.440 min   |
| 6      | Km desde última transação      | max: 1.000 km    |
| 7      | Km de casa                     | max: 1.000 km    |
| 8      | Transações nas últimas 24h     | max: 20          |
| 9      | Terminal online                | 0 ou 1           |
| 10     | Cartão presente                | 0 ou 1           |
| 11     | Merchant desconhecido          | 0 ou 1           |
| 12     | Risco do MCC                   | lookup table     |
| 13     | Valor médio do merchant        | max: R$ 10.000   |

## API

### `GET /ready`
Health check — retorna `200` quando o serviço estiver pronto.

### `POST /fraud-score`
```json
{
  "transaction": { "amount": 150.00, "installments": 1, "requested_at": "2025-11-17T10:30:00Z" },
  "customer": { "avg_amount": 120.00, "tx_count_24h": 2, "known_merchants": ["abc123"] },
  "merchant": { "id": "abc123", "mcc": "5411", "avg_amount": 200.00 },
  "terminal": { "is_online": true, "card_present": true, "km_from_home": 2.5 },
  "last_transaction": { "requested_at": "2025-11-17T08:00:00Z", "km_from_last": 1.2 }
}
```

**Resposta:**
```json
{ "approved": true, "fraud_score": 0.2 }
```

## Rodando localmente

**Pré-requisitos:** Docker e Docker Compose

```bash
# O arquivo references.json.gz (~50MB) precisa estar em resources/
# Ele não está no git por ser grande — baixe do repositório oficial

docker compose up --build
```

O serviço sobe na porta `9999`.

## Estrutura do projeto

```
├── cmd/
│   ├── api/main.go       # Serviço HTTP da API
│   └── knn/main.go       # Serviço de busca KNN
├── internal/
│   ├── models/           # Structs de request/response
│   └── vectorize/        # Transformação transação → vetor
├── nginx/nginx.conf       # Configuração do load balancer
├── resources/
│   ├── mcc_risk.json      # Tabela de risco por categoria
│   ├── normalization.json # Constantes de normalização
│   └── references.json.gz # ~3M vetores de referência (não versionado)
├── Dockerfile.api
├── Dockerfile.knn
└── docker-compose.yml
```
