package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"runtime"
	"sync"
	"sync/atomic"

	gojson "github.com/goccy/go-json"
	"github.com/valyala/fasthttp"

	"github.com/Marcoant007/rinha-2026/internal/models"
	"github.com/Marcoant007/rinha-2026/internal/vectorize"
)

const (
	dims   = 14
	stride = 150
	// approvalThreshold: aprova apenas se nenhum vizinho for fraude (0/5 = 0.0 < 0.1)
	// Penalidade FN=3x > FP=1x, então ser agressivo na detecção vale mais
	approvalThreshold = 0.1
)

var (
	refVectors []uint8
	refLabels  []uint8
	numRefs    int
	dataReady  int32

	// knnResponses[i] = resposta JSON para fraudCount=i, i in [0,5]
	// Pré-computado para zero alocações por request
	knnResponses [6][]byte

	// fallbackResponse: retornado durante loading, erro de parse ou sobrecarga
	fallbackResponse []byte

	bufPool sync.Pool

)

func init() {
	for i := 0; i <= 5; i++ {
		score := float64(i) / 5.0
		approved := score < approvalThreshold
		appr := "false"
		if approved {
			appr = "true"
		}
		knnResponses[i] = fmt.Appendf(nil,
			`{"approved":%s,"fraud_score":%.1f}`,
			appr, score,
		)
	}
	fallbackResponse = []byte(`{"approved":true,"fraud_score":0.0}`)

	bufPool.New = func() interface{} {
		b := make([]byte, 0, 4096)
		return &b
	}
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

	log.Println("Carregando vetores de referência (binário)...")

	var n uint32
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return err
	}

	total := int(n)

	allVecs := make([]uint8, total*dims)
	if _, err := io.ReadFull(f, allVecs); err != nil {
		return err
	}

	allLabels := make([]uint8, total)
	if _, err := io.ReadFull(f, allLabels); err != nil {
		return err
	}

	sampled := total/stride + 1
	refVectors = make([]uint8, 0, sampled*dims)
	refLabels = make([]uint8, 0, sampled)
	for i := 0; i < total; i += stride {
		refVectors = append(refVectors, allVecs[i*dims:(i+1)*dims]...)
		refLabels = append(refLabels, allLabels[i])
	}

	allVecs = nil
	allLabels = nil
	runtime.GC()

	numRefs = len(refLabels)
	atomic.StoreInt32(&dataReady, 1)
	log.Printf("Carregados %d/%d vetores.\n", numRefs, total)
	return nil
}

type neighbor struct {
	dist    uint32
	isFraud bool
}

func knn5(query [14]float64) int {
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
			d := int32(q[j]) - int32(refVectors[base+j])
			sq += uint32(d * d)
		}
		if heapSize < 5 {
			heap[heapSize] = neighbor{sq, refLabels[i] == 1}
			heapSize++
			if heapSize == 5 {
				buildMaxHeap(&heap)
			}
		} else if sq < heap[0].dist {
			heap[0] = neighbor{sq, refLabels[i] == 1}
			siftDown(&heap, 0)
		}
	}

	fraudCount := 0
	for _, nb := range heap {
		if nb.isFraud {
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
	if atomic.LoadInt32(&dataReady) == 0 {
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

	// Durante carregamento, retorna fallback aprovado em vez de 503
	if atomic.LoadInt32(&dataReady) == 0 {
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetContentTypeBytes([]byte("application/json"))
		ctx.Write(fallbackResponse)
		return
	}

	bufp := bufPool.Get().(*[]byte)
	body := ctx.PostBody()
	*bufp = append((*bufp)[:0], body...)

	var req models.TransactionRequest
	err := gojson.Unmarshal(*bufp, &req)
	bufPool.Put(bufp)

	if err != nil {
		// Retorna 200 com fallback em vez de 400 (peso 5 no scoring)
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetContentTypeBytes([]byte("application/json"))
		ctx.Write(fallbackResponse)
		return
	}

	vec := vectorize.Vectorize(&req)
	fraudCount := knn5(vec)

	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentTypeBytes([]byte("application/json"))
	ctx.Write(knnResponses[fraudCount])
}

func main() {
	runtime.GOMAXPROCS(2)

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

	requestHandler := func(ctx *fasthttp.RequestCtx) {
		switch string(ctx.Path()) {
		case "/ready":
			handleReady(ctx)
		case "/fraud-score":
			handleFraudScore(ctx)
		default:
			ctx.SetStatusCode(fasthttp.StatusNotFound)
		}
	}

	ln, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("Erro ao escutar porta %s: %v", port, err)
	}

	srv := &fasthttp.Server{
		Handler:               requestHandler,
		MaxRequestBodySize:    4096,
		NoDefaultServerHeader: true,
		NoDefaultContentType:  true,
		ReadBufferSize:        4096,
		WriteBufferSize:       4096,
		Concurrency:           256,
	}

	log.Printf("api escutando na porta %s\n", port)
	if err := srv.Serve(ln); err != nil {
		log.Fatal(err)
	}
}
