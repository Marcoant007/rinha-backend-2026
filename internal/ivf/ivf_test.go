package ivf

import (
	"bufio"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// buildTestIndex writes a minimal IVF binary with synthetic data and returns the file path.
// Layout: 2 clusters, 20 vectors each (10 fraud + 10 legit per cluster).
func buildTestIndex(t testing.TB) string {
	t.Helper()

	const nlist = 2
	const n = 40 // 20 per cluster

	// Centroids: cluster 0 near (0,0,...), cluster 1 near (0.5,0.5,...)
	centroids := make([]float32, nlist*Dims)
	for d := 0; d < Dims; d++ {
		centroids[0*Dims+d] = 0.0
		centroids[1*Dims+d] = 0.5
	}

	// Offsets: cluster 0 = [0, 20), cluster 1 = [20, 40)
	offsets := []uint32{0, 20, 40}

	// Vectors: cluster 0 near 0, cluster 1 near 0.5 (quantized × 10000)
	vectors := make([]int16, n*Dims)
	for i := 0; i < n; i++ {
		base := float32(0)
		if i >= 20 {
			base = 0.5
		}
		for d := 0; d < Dims; d++ {
			// Add slight noise per vector
			noise := float32(i%5) * 0.001
			vectors[i*Dims+d] = int16((base+noise)*Scale + 0.5)
		}
	}

	// Labels: first 10 of each cluster = fraud, last 10 = legit
	labels := make([]byte, (n+7)/8)
	for i := 0; i < n; i++ {
		posInCluster := i % 20
		if posInCluster < 10 { // fraud
			labels[i/8] |= 1 << uint(i%8)
		}
	}

	f, err := os.CreateTemp("", "ivf_test_*.bin")
	if err != nil {
		t.Fatalf("create temp file: %v", err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	binary.Write(w, binary.LittleEndian, Magic)
	binary.Write(w, binary.LittleEndian, uint32(nlist))
	binary.Write(w, binary.LittleEndian, uint32(n))
	binary.Write(w, binary.LittleEndian, centroids)
	binary.Write(w, binary.LittleEndian, offsets)
	binary.Write(w, binary.LittleEndian, vectors)
	w.Write(labels)
	w.Flush()

	return f.Name()
}

func TestLoad(t *testing.T) {
	path := buildTestIndex(t)
	defer os.Remove(path)

	idx, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if idx.n != 40 {
		t.Errorf("n: got %d, want 40", idx.n)
	}
	if idx.nlist != 2 {
		t.Errorf("nlist: got %d, want 2", idx.nlist)
	}
}

func TestSearch_FraudCluster(t *testing.T) {
	path := buildTestIndex(t)
	defer os.Remove(path)

	idx, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Query very close to cluster 0 fraud vectors (near origin)
	var q [Dims]float32
	for d := 0; d < Dims; d++ {
		q[d] = 0.001 // very close to cluster 0, which has fraud in first 10 vectors
	}
	count := idx.Search(q)
	// Should find mostly fraud neighbours (cluster 0, first 10 = fraud)
	if count == 0 {
		t.Errorf("expected fraud neighbours near fraud cluster, got 0")
	}
}

func TestSearch_LegitCluster(t *testing.T) {
	path := buildTestIndex(t)
	defer os.Remove(path)

	idx, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Query very close to cluster 0 legit vectors (slight offset)
	var q [Dims]float32
	for d := 0; d < Dims; d++ {
		q[d] = 0.016 // within cluster 0, closer to legit vectors (indices 10-19)
	}
	count := idx.Search(q)
	// Result should be in [0, K]
	if count < 0 || count > K {
		t.Errorf("fraud count %d out of range [0, %d]", count, K)
	}
}

func TestSearch_ReturnsBounded(t *testing.T) {
	path := buildTestIndex(t)
	defer os.Remove(path)

	idx, _ := Load(path)

	// Multiple random-ish queries; result must always be in [0, K].
	queries := [][Dims]float32{
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
		{-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
	for _, q := range queries {
		c := idx.Search(q)
		if c < 0 || c > K {
			t.Errorf("Search(%v) = %d, want [0, %d]", q, c, K)
		}
	}
}

// --- Benchmark ---

func BenchmarkSearch(b *testing.B) {
	path := buildTestIndex(b)
	defer os.Remove(path)

	idx, err := Load(path)
	if err != nil {
		b.Fatalf("Load: %v", err)
	}

	var q [Dims]float32
	for d := 0; d < Dims; d++ {
		q[d] = 0.3
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(q)
	}
}

// BenchmarkSearchLarger tests with a bigger synthetic index (~500 vectors)
// to better approximate real-world performance.
func BenchmarkSearchLarger(b *testing.B) {
	const nlist = Nlist // 256
	const perCluster = 2000
	const n = nlist * perCluster

	centroids := make([]float32, nlist*Dims)
	for i := 0; i < nlist; i++ {
		v := float32(i) / float32(nlist)
		for d := 0; d < Dims; d++ {
			centroids[i*Dims+d] = v
		}
	}

	offsets := make([]uint32, nlist+1)
	for i := 0; i <= nlist; i++ {
		offsets[i] = uint32(i * perCluster)
	}

	vectors := make([]int16, n*Dims)
	for i := 0; i < n; i++ {
		clust := i / perCluster
		base := float32(clust) / float32(nlist)
		for d := 0; d < Dims; d++ {
			vectors[i*Dims+d] = int16(base * Scale)
		}
	}

	labels := make([]byte, (n+7)/8)
	for i := 0; i < n; i += 2 {
		labels[i/8] |= 1 << uint(i%8)
	}

	f, _ := os.CreateTemp("", "ivf_bench_*.bin")
	w := bufio.NewWriter(f)
	binary.Write(w, binary.LittleEndian, Magic)
	binary.Write(w, binary.LittleEndian, uint32(nlist))
	binary.Write(w, binary.LittleEndian, uint32(n))
	binary.Write(w, binary.LittleEndian, centroids)
	binary.Write(w, binary.LittleEndian, offsets)
	binary.Write(w, binary.LittleEndian, vectors)
	w.Write(labels)
	w.Flush()
	f.Close()
	defer os.Remove(f.Name())

	idx, _ := Load(f.Name())

	var q [Dims]float32
	for d := 0; d < Dims; d++ {
		q[d] = 0.3
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(q)
	}
}

// Ensure math import doesn't cause issues
var _ = math.MaxFloat32
