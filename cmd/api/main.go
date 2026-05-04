package main

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/Marcoant007/rinha-2026/internal/models"
	"github.com/Marcoant007/rinha-2026/internal/vectorize"
)

var knnServiceURL string
var httpClient = &http.Client{Timeout: 30 * time.Second}

func handleReady(w http.ResponseWriter, r *http.Request) {
	resp, err := httpClient.Get(knnServiceURL + "/ready")
	if err != nil || resp.StatusCode >= 300 {
		http.Error(w, "knn-service not ready", http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
}

func handleFraudScore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req models.TransactionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	vec := vectorize.Vectorize(&req)

	knnReq := models.KNNRequest{Vector: vec}
	body, _ := json.Marshal(knnReq)

	resp, err := httpClient.Post(knnServiceURL+"/knn", "application/json", bytes.NewReader(body))
	if err != nil {
		log.Printf("Erro ao chamar knn-service: %v\n", err)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models.FraudResponse{Approved: true, FraudScore: 0.0})
		return
	}
	defer resp.Body.Close()

	var knnResp models.KNNResponse
	if err := json.NewDecoder(resp.Body).Decode(&knnResp); err != nil {
		log.Printf("Erro ao decodificar resposta do knn-service: %v\n", err)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models.FraudResponse{Approved: true, FraudScore: 0.0})
		return
	}

	fraudScore := 0.0
	if knnResp.Total > 0 {
		fraudScore = float64(knnResp.FraudCount) / float64(knnResp.Total)
	}
	approved := fraudScore < 0.6

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models.FraudResponse{
		Approved:   approved,
		FraudScore: fraudScore,
	})
}

func main() {
	knnServiceURL = os.Getenv("KNN_SERVICE_URL")
	if knnServiceURL == "" {
		knnServiceURL = "http://knn:8080"
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8081"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ready", handleReady)
	mux.HandleFunc("/fraud-score", handleFraudScore)

	log.Printf("api escutando na porta %s\n", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}
