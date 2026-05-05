package main

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"sync/atomic"

	"github.com/Marcoant007/rinha-2026/internal/models"
	"github.com/Marcoant007/rinha-2026/internal/vectorize"
)

const dims = 14

var (
	vectors   []uint8
	labels    []uint8
	numRefs   int
	dataReady int32
	sem       = make(chan struct{}, 1)
)

func encodeFloat(v float64) uint8 {
	return uint8((v + 1.0) * 127.5)
}

func loadReferences(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	log.Println("Carregando vetores de referência (binário)...")

	var n uint32
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return err
	}

	vectors = make([]uint8, int(n)*dims)
	if _, err := io.ReadFull(f, vectors); err != nil {
		return err
	}

	labels = make([]uint8, n)
	if _, err := io.ReadFull(f, labels); err != nil {
		return err
	}

	numRefs = int(n)
	atomic.StoreInt32(&dataReady, 1)
	log.Printf("Carregados %d vetores.\n", numRefs)
	return nil
}

type neighbor struct {
	dist    uint32
	isFraud bool
}

func knn5(query []float64) int {
	var q [dims]uint8
	for i, v := range query {
		q[i] = encodeFloat(v)
	}

	var heap [5]neighbor
	heapSize := 0

	for i := 0; i < numRefs; i++ {
		base := i * dims
		var sq uint32
		for j := 0; j < dims; j++ {
			d := int32(q[j]) - int32(vectors[base+j])
			sq += uint32(d * d)
		}
		if heapSize < 5 {
			heap[heapSize] = neighbor{sq, labels[i] == 1}
			heapSize++
			if heapSize == 5 {
				buildMaxHeap(&heap)
			}
		} else if sq < heap[0].dist {
			heap[0] = neighbor{sq, labels[i] == 1}
			siftDown(&heap, 0)
		}
	}

	fraudCount := 0
	for _, n := range heap {
		if n.isFraud {
			fraudCount++
		}
	}
	return fraudCount
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

func handleReady(w http.ResponseWriter, r *http.Request) {
	if atomic.LoadInt32(&dataReady) == 0 {
		http.Error(w, "loading", http.StatusServiceUnavailable)
		return
	}
	w.WriteHeader(http.StatusOK)
}

func handleFraudScore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if atomic.LoadInt32(&dataReady) == 0 {
		http.Error(w, "loading", http.StatusServiceUnavailable)
		return
	}

	var req models.TransactionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	vec := vectorize.Vectorize(&req)
	sem <- struct{}{}
	fraudCount := knn5(vec)
	<-sem
	fraudScore := float64(fraudCount) / 5.0

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(models.FraudResponse{
		Approved:   fraudScore < 0.6,
		FraudScore: fraudScore,
	})
}

func main() {
	refsPath := os.Getenv("REFS_PATH")
	if refsPath == "" {
		refsPath = "/data/references.bin"
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8081"
	}

	go func() {
		if err := loadReferences(refsPath); err != nil {
			log.Fatalf("Erro ao carregar referências: %v", err)
		}
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("/ready", handleReady)
	mux.HandleFunc("/fraud-score", handleFraudScore)

	log.Printf("api escutando na porta %s\n", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}
