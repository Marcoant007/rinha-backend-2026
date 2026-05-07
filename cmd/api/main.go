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

	"github.com/Marcoant007/rinha-2026/internal/ivf"
	"github.com/Marcoant007/rinha-2026/internal/models"
	"github.com/Marcoant007/rinha-2026/internal/vectorize"
)

// approvalThreshold: approve only if 0 of 5 neighbours are fraud.
// FN costs 3× more than FP, so be aggressive detecting fraud.
const approvalThreshold = 0.1

var (
	idx       *ivf.Index
	dataReady int32

	// knnResponses[i] = pre-built JSON for fraudCount=i (0..5), zero alloc.
	knnResponses [6][]byte

	// fallbackResponse: returned while loading or on parse errors.
	fallbackResponse []byte

	// bodyPool reduces allocations for request body copying.
	bodyPool sync.Pool

	// reqPool reuses TransactionRequest structs, including KnownMerchants slice.
	reqPool sync.Pool
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

	bodyPool.New = func() interface{} {
		b := make([]byte, 0, 4096)
		return &b
	}
	reqPool.New = func() interface{} {
		return new(models.TransactionRequest)
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

	// Serve a safe fallback while the index is loading.
	if atomic.LoadInt32(&dataReady) == 0 {
		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetContentTypeBytes([]byte("application/json"))
		ctx.Write(fallbackResponse)
		return
	}

	// Copy body via pooled buffer then unmarshal into a pooled request struct.
	bufp := bodyPool.Get().(*[]byte)
	body := ctx.PostBody()
	*bufp = append((*bufp)[:0], body...)

	req := reqPool.Get().(*models.TransactionRequest)
	err := gojson.Unmarshal(*bufp, req)
	bodyPool.Put(bufp)

	if err != nil {
		merchants := req.Customer.KnownMerchants[:0]
		*req = models.TransactionRequest{}
		req.Customer.KnownMerchants = merchants
		reqPool.Put(req)

		ctx.SetStatusCode(fasthttp.StatusOK)
		ctx.SetContentTypeBytes([]byte("application/json"))
		ctx.Write(fallbackResponse)
		return
	}

	vec64 := vectorize.Vectorize(req)

	// Return request to pool, preserving the KnownMerchants backing array.
	merchants := req.Customer.KnownMerchants[:0]
	*req = models.TransactionRequest{}
	req.Customer.KnownMerchants = merchants
	reqPool.Put(req)

	var query [ivf.Dims]float32
	for i, v := range vec64 {
		query[i] = float32(v)
	}

	fraudCount := idx.Search(query)

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

	// Determine listen address: Unix socket (when INSTANCE_ID is set) or TCP.
	instanceID := os.Getenv("INSTANCE_ID")
	var ln net.Listener
	var listenDesc string

	if instanceID != "" {
		sockPath := fmt.Sprintf("/tmp/sockets/api%s.sock", instanceID)
		os.Remove(sockPath)
		var err error
		ln, err = net.Listen("unix", sockPath)
		if err != nil {
			log.Fatalf("unix listen %s: %v", sockPath, err)
		}
		os.Chmod(sockPath, 0777)
		listenDesc = "unix:" + sockPath
	} else {
		port := os.Getenv("PORT")
		if port == "" {
			port = "8081"
		}
		var err error
		ln, err = net.Listen("tcp", ":"+port)
		if err != nil {
			log.Fatalf("tcp listen :%s: %v", port, err)
		}
		listenDesc = "tcp::" + port
	}

	// Load the IVF index in the background; serve fallback until ready.
	go func() {
		log.Println("Loading IVF index...")
		var err error
		idx, err = ivf.Load(indexPath)
		if err != nil {
			log.Fatalf("failed to load IVF index: %v", err)
		}
		atomic.StoreInt32(&dataReady, 1)
		log.Println("IVF index ready!")
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

	// No Concurrency cap: fasthttp queues connections gracefully under load.
	srv := &fasthttp.Server{
		Handler:               requestHandler,
		MaxRequestBodySize:    8192,
		NoDefaultServerHeader: true,
		NoDefaultContentType:  true,
		ReadBufferSize:        4096,
		WriteBufferSize:       4096,
	}

	log.Printf("API listening on %s\n", listenDesc)
	if err := srv.Serve(ln); err != nil {
		log.Fatal(err)
	}
}
