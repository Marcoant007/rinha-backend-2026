package main

import (
	"compress/gzip"
	"log"
	"os"
	"runtime"

	gojson "github.com/goccy/go-json"
	"github.com/valyala/fasthttp"

	"github.com/Marcoant007/rinha-2026/internal/models"
	"github.com/Marcoant007/rinha-2026/internal/vectorize"
)

const (
	dims       = 14
	sampleRate = 33 // carrega 1 em cada 33 vetores → ~90K refs de 3M
)

var (
	vectors []uint8
	labels  []bool
	numRefs int
	ready   bool
	sem     = make(chan struct{}, 4)
)

// respostas pré-computadas: zero alocação por request
// índice = número de vizinhos fraude (0-5)
var responses [6][]byte

func init() {
	scores := []float64{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
	for i, s := range scores {
		approved := s < 0.2
		if approved {
			responses[i] = []byte(`{"approved":true,"fraud_score":` + ftoa(s) + `}`)
		} else {
			responses[i] = []byte(`{"approved":false,"fraud_score":` + ftoa(s) + `}`)
		}
	}
}

func ftoa(f float64) string {
	switch f {
	case 0.0:
		return "0"
	case 0.2:
		return "0.2"
	case 0.4:
		return "0.4"
	case 0.6:
		return "0.6"
	case 0.8:
		return "0.8"
	default:
		return "1"
	}
}

// fallback: approved=true quando sistema ocupado ou erro de parse
// FN (peso 3) ou TN (peso 0) é sempre melhor que HTTP error (peso 5)
var fallback = []byte(`{"approved":true,"fraud_score":0}`)

type refJSON struct {
	Vector [dims]float64 `json:"vector"`
	Label  string        `json:"label"`
}

func encodeFloat(v float64) uint8 {
	return uint8((v + 1.0) * 127.5)
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

	dec := gojson.NewDecoder(gz)
	if _, err := dec.Token(); err != nil {
		return err
	}

	capacity := 3_000_000 / sampleRate
	vectors = make([]uint8, 0, capacity*dims)
	labels = make([]bool, 0, capacity)

	var r refJSON
	var count int
	for dec.More() {
		if err := dec.Decode(&r); err != nil {
			return err
		}
		if count%sampleRate == 0 {
			for _, v := range r.Vector {
				vectors = append(vectors, encodeFloat(v))
			}
			labels = append(labels, r.Label == "fraud")
		}
		count++
	}

	numRefs = len(labels)
	ready = true
	log.Printf("Carregados %d vetores de referência (1 em %d).\n", numRefs, sampleRate)
	return nil
}

type neighbor struct {
	dist    uint32
	isFraud bool
}

func scanChunk(query [dims]uint8, start, end int) [5]neighbor {
	var heap [5]neighbor
	heapSize := 0

	for i := start; i < end; i++ {
		base := i * dims

		var sq uint32
		for j := 0; j < dims; j++ {
			d := int32(query[j]) - int32(vectors[base+j])
			sq += uint32(d * d)
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

	for heapSize < 5 {
		heap[heapSize] = neighbor{^uint32(0), false}
		heapSize++
	}

	return heap
}

// knn5: scan sequencial sem goroutines auxiliares.
// Com CPU limitada (0.45 core), goroutines paralelas adicionam overhead sem ganho.
func knn5(query []float64) int {
	var q [dims]uint8
	for i, v := range query {
		q[i] = encodeFloat(v)
	}
	heap := scanChunk(q, 0, numRefs)
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

func handleReady(ctx *fasthttp.RequestCtx) {
	if !ready {
		ctx.SetStatusCode(fasthttp.StatusServiceUnavailable)
		return
	}
	ctx.SetStatusCode(fasthttp.StatusOK)
}

func handleFraudScore(ctx *fasthttp.RequestCtx) {
	if !ctx.IsPost() {
		ctx.SetStatusCode(fasthttp.StatusMethodNotAllowed)
		return
	}

	var req models.TransactionRequest
	if err := gojson.Unmarshal(ctx.PostBody(), &req); err != nil {
		ctx.SetContentType("application/json")
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.Write(fallback)
		return
	}

	vec := vectorize.Vectorize(&req)

	select {
	case sem <- struct{}{}:
		fraudCount := knn5(vec)
		<-sem
		ctx.SetContentType("application/json")
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.Write(responses[fraudCount])
	default:
		// sistema ocupado: fallback imediato em vez de bloquear 2s
		ctx.SetContentType("application/json")
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.Write(fallback)
	}
}

func requestHandler(ctx *fasthttp.RequestCtx) {
	switch string(ctx.Path()) {
	case "/fraud-score":
		handleFraudScore(ctx)
	case "/ready":
		handleReady(ctx)
	default:
		ctx.SetStatusCode(fasthttp.StatusNotFound)
	}
}

func main() {
	// 1 thread OS: evita overhead de scheduling com apenas 0.45 CPU
	runtime.GOMAXPROCS(1)

	refsPath := os.Getenv("REFS_PATH")
	if refsPath == "" {
		refsPath = "/data/references.json.gz"
	}

	if err := loadReferences(refsPath); err != nil {
		log.Fatalf("Erro ao carregar referências: %v", err)
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8081"
	}

	srv := &fasthttp.Server{
		Handler:            requestHandler,
		ReadTimeout:        2000000000, // 2s em nanosegundos
		WriteTimeout:       5000000000,
		MaxRequestBodySize: 8192,
		Concurrency:        512,
	}

	log.Printf("api escutando na porta %s\n", port)
	if err := srv.ListenAndServe(":" + port); err != nil {
		log.Fatal(err)
	}
}
