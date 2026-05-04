package main

import (
	"compress/gzip"
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/Marcoant007/rinha-2026/internal/models"
)

const dims = 14

var (
	vectors []float32
	labels  []bool
	numRefs int
	ready   bool
)

type refJSON struct {
	Vector [dims]float64 `json:"vector"`
	Label  string        `json:"label"`
}

func loadReferences(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	log.Println("Carregando vetores de referência...")

	dec := json.NewDecoder(gz)

	if _, err := dec.Token(); err != nil {
		return err
	}

	vectors = make([]float32, 0, 3_000_000*dims)
	labels = make([]bool, 0, 3_000_000)

	var r refJSON
	for dec.More() {
		if err := dec.Decode(&r); err != nil {
			return err
		}
		for _, v := range r.Vector {
			vectors = append(vectors, float32(v))
		}
		labels = append(labels, r.Label == "fraud")
	}

	numRefs = len(labels)
	ready = true
	log.Printf("Carregados %d vetores de referência.\n", numRefs)
	return nil
}

type neighbor struct {
	dist    float32
	isFraud bool
}

func knn5(query []float64) [5]neighbor {
	var heap [5]neighbor
	heapSize := 0

	for i := 0; i < numRefs; i++ {
		base := i * dims

		var sq float32
		for j := 0; j < dims; j++ {
			d := float32(query[j]) - vectors[base+j]
			sq += d * d
		}

		if heapSize < 5 {
			heap[heapSize] = neighbor{sq, labels[i]}
			heapSize++
			if heapSize == 5 {
				buildMaxHeap(&heap)
			}
		} else if sq < heap[0].dist {
			heap[0] = neighbor{sq, labels[i]}
			siftDown(&heap, 0)
		}
	}

	return heap
}

func buildMaxHeap(h *[5]neighbor) {
	for i := 1; i < 5; i++ {
		siftUp(h, i)
	}
}

func siftUp(h *[5]neighbor, i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h[parent].dist >= h[i].dist {
			break
		}
		h[parent], h[i] = h[i], h[parent]
		i = parent
	}
}

func siftDown(h *[5]neighbor, i int) {
	for {
		largest := i
		l, r := 2*i+1, 2*i+2
		if l < 5 && h[l].dist > h[largest].dist {
			largest = l
		}
		if r < 5 && h[r].dist > h[largest].dist {
			largest = r
		}
		if largest == i {
			break
		}
		h[i], h[largest] = h[largest], h[i]
		i = largest
	}
}

func handleKNN(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req models.KNNRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	if len(req.Vector) != dims {
		http.Error(w, "vector must have 14 dimensions", http.StatusBadRequest)
		return
	}

	neighbors := knn5(req.Vector)

	fraudCount := 0
	for _, n := range neighbors {
		if n.isFraud {
			fraudCount++
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models.KNNResponse{
		FraudCount: fraudCount,
		Total:      5,
	})
}

func handleReady(w http.ResponseWriter, r *http.Request) {
	if !ready {
		http.Error(w, "loading", http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
}

func main() {
	refsPath := os.Getenv("REFS_PATH")
	if refsPath == "" {
		refsPath = "/data/references.json.gz"
	}

	if err := loadReferences(refsPath); err != nil {
		log.Fatalf("Erro ao carregar referências: %v", err)
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/knn", handleKNN)
	mux.HandleFunc("/ready", handleReady)

	log.Printf("knn-service escutando na porta %s\n", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}
