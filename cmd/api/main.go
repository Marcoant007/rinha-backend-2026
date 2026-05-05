package main

import (
	"fmt"
	"log"
	"net"
	"os"
	"runtime"
	"sync"
	"sync/atomic"

	gojson "github.com/goccy/go-json"
	"github.com/valyala/fasthttp"

	"github.com/Marcoant007/rinha-2026/internal/kdtree"
	"github.com/Marcoant007/rinha-2026/internal/models"
	"github.com/Marcoant007/rinha-2026/internal/vectorize"
)

// approvalThreshold: aprova somente se nenhum dos 5 vizinhos for fraude (0/5 = 0.0)
// FN custa 3x mais que FP, então ser agressivo na detecção vale mais
const approvalThreshold = 0.1

var (
	idx       *kdtree.Index
	dataReady int32

	// knnResponses[i] = JSON para fraudCount=i (i in 0..5), zero alloc por request
	knnResponses [6][]byte

	// fallbackResponse: retornado durante loading ou erro de parse
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
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetContentTypeBytes([]byte("application/json"))
		ctx.Write(fallbackResponse)
		return
	}

	vec64 := vectorize.Vectorize(&req)
	var query [kdtree.Dims]float32
	for i, v := range vec64 {
		query[i] = float32(v)
	}

	fraudCount := idx.Search(query, kdtree.VisitCap)

	ctx.SetStatusCode(fasthttp.StatusOK)
	ctx.SetContentTypeBytes([]byte("application/json"))
	ctx.Write(knnResponses[fraudCount])
}

func main() {
	runtime.GOMAXPROCS(1)

	indexPath := os.Getenv("INDEX_PATH")
	if indexPath == "" {
		indexPath = "/data/index.bin"
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "8081"
	}

	go func() {
		log.Println("Carregando K-D Tree...")
		var err error
		idx, err = kdtree.Load(indexPath)
		if err != nil {
			log.Fatalf("Erro ao carregar índice: %v", err)
		}
		atomic.StoreInt32(&dataReady, 1)
		log.Println("K-D Tree pronta!")
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
