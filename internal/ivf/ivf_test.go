package ivf

import (
	"bufio"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// writeIVF2 writes a valid IVF2 binary file given per-cluster vectors and labels.
// vecs[c][i] is a float32 vector in QuantScale space; lbls[c][i] is 1=fraud, 0=legit.
func writeIVF2(t testing.TB, vecs [][][Dims]float32, lbls [][]uint8) string {
	t.Helper()
	K := len(vecs)

	// Count blocks per cluster (block = up to BlockLanes vectors).
	blocksPerCluster := make([]int, K)
	totalBlocks := 0
	totalN := 0
	for c, cv := range vecs {
		n := len(cv)
		totalN += n
		b := (n + BlockLanes - 1) / BlockLanes
		if b == 0 && n > 0 {
			b = 1
		}
		blocksPerCluster[c] = b
		totalBlocks += b
	}

	// Centroids: mean of each cluster (in QuantScale space).
	centroids := make([]float32, K*Dims)
	for c, cv := range vecs {
		if len(cv) == 0 {
			continue
		}
		for _, v := range cv {
			for d := 0; d < Dims; d++ {
				centroids[c*Dims+d] += v[d]
			}
		}
		inv := float32(1) / float32(len(cv))
		for d := 0; d < Dims; d++ {
			centroids[c*Dims+d] *= inv
		}
	}

	// Offsets in block units (K+1 entries).
	offsets := make([]uint32, K+1)
	for c := 0; c < K; c++ {
		offsets[c+1] = offsets[c] + uint32(blocksPerCluster[c])
	}

	// BboxMin / BboxMax: K * BboxLanes int16s.
	bboxMin := make([]int16, K*BboxLanes)
	bboxMax := make([]int16, K*BboxLanes)
	for i := range bboxMin {
		bboxMin[i] = math.MaxInt16
		bboxMax[i] = math.MinInt16
	}
	for c, cv := range vecs {
		base := c * BboxLanes
		for _, v := range cv {
			for d := 0; d < Dims; d++ {
				q := quantize16(v[d])
				if q < bboxMin[base+d] {
					bboxMin[base+d] = q
				}
				if q > bboxMax[base+d] {
					bboxMax[base+d] = q
				}
			}
		}
		// Zero-pad unused BboxLanes dimensions.
		for d := Dims; d < BboxLanes; d++ {
			bboxMin[base+d] = 0
			bboxMax[base+d] = 0
		}
	}
	// For empty clusters, reset to 0.
	for c, cv := range vecs {
		if len(cv) == 0 {
			base := c * BboxLanes
			for d := 0; d < BboxLanes; d++ {
				bboxMin[base+d] = 0
				bboxMax[base+d] = 0
			}
		}
	}

	// Labels and BlockData in block-major layout.
	labels := make([]uint8, totalBlocks*BlockLanes)
	blockData := make([]int16, totalBlocks*BlockStride)

	blockIdx := 0
	for c, cv := range vecs {
		for b := 0; b < blocksPerCluster[c]; b++ {
			lb := (blockIdx + b) * BlockLanes
			bb := (blockIdx + b) * BlockStride
			for lane := 0; lane < BlockLanes; lane++ {
				vi := b*BlockLanes + lane
				if vi < len(cv) {
					labels[lb+lane] = lbls[c][vi]
					for d := 0; d < Dims; d++ {
						blockData[bb+d*BlockLanes+lane] = quantize16(cv[vi][d])
					}
				}
				// phantom lanes: label=0, data=0 (zero value).
			}
		}
		blockIdx += blocksPerCluster[c]
	}

	f, err := os.CreateTemp("", "ivf_test_*.bin")
	if err != nil {
		t.Fatalf("create temp: %v", err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	// Header: magic(4)+version(4)+N(4)+K(4)+Blocks(4)+pad(12) = 32 bytes.
	w.Write(BinaryMagic())
	binary.Write(w, binary.LittleEndian, uint32(1)) // version
	binary.Write(w, binary.LittleEndian, uint32(totalN))
	binary.Write(w, binary.LittleEndian, uint32(K))
	binary.Write(w, binary.LittleEndian, uint32(totalBlocks))
	w.Write(make([]byte, 12)) // padding
	binary.Write(w, binary.LittleEndian, centroids)
	binary.Write(w, binary.LittleEndian, offsets)
	binary.Write(w, binary.LittleEndian, bboxMin)
	binary.Write(w, binary.LittleEndian, bboxMax)
	w.Write(labels)
	binary.Write(w, binary.LittleEndian, blockData)
	w.Flush()

	return f.Name()
}

// quantize16 rounds a float32 (QuantScale space) to int16.
func quantize16(v float32) int16 {
	if v < 0 {
		return int16(v - 0.5)
	}
	return int16(v + 0.5)
}

// makeQuery builds qf (float32) and qi (int16) query arrays from a single scalar value.
func makeQuery(val float32) (*[Dims]float32, *[Dims]int16) {
	var qf [Dims]float32
	var qi [Dims]int16
	for d := 0; d < Dims; d++ {
		qf[d] = val
		qi[d] = quantize16(val)
	}
	return &qf, &qi
}

// buildTestIndex creates a small test index with 2 clusters, 20 vectors each.
// Cluster 0: fraud vectors near 0, legit vectors near 3000.
// Cluster 1: fraud vectors near 5000, legit vectors near 8000.
func buildTestIndex(t testing.TB) string {
	t.Helper()

	const nClusters = 2
	const perCluster = 20

	vecs := make([][][Dims]float32, nClusters)
	lbls := make([][]uint8, nClusters)

	bases := [nClusters][2]float32{
		{0, 3000},    // cluster 0: fraud base, legit base
		{5000, 8000}, // cluster 1: fraud base, legit base
	}

	for c := 0; c < nClusters; c++ {
		vecs[c] = make([][Dims]float32, perCluster)
		lbls[c] = make([]uint8, perCluster)
		for i := 0; i < perCluster; i++ {
			var base float32
			if i < 10 {
				base = bases[c][0] // fraud
				lbls[c][i] = 1
			} else {
				base = bases[c][1] // legit
			}
			noise := float32(i%10) * 10.0
			for d := 0; d < Dims; d++ {
				vecs[c][i][d] = base + noise
			}
		}
	}

	return writeIVF2(t, vecs, lbls)
}

// --- Tests ---

func TestLoad(t *testing.T) {
	path := buildTestIndex(t)
	defer os.Remove(path)

	idx, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if idx.N != 40 {
		t.Errorf("N: got %d, want 40", idx.N)
	}
	if idx.K != 2 {
		t.Errorf("K: got %d, want 2", idx.K)
	}
}

func TestSearch_FraudCluster(t *testing.T) {
	path := buildTestIndex(t)
	defer os.Remove(path)

	idx, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	scratch := NewScratch()
	// Query near cluster 0 fraud vectors (base=0, values 0..90).
	qf, qi := makeQuery(5.0)
	count := idx.Search(qf, qi, scratch)
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

	scratch := NewScratch()
	// Query near cluster 0 legit vectors (base=3000, values 3000..3090).
	qf, qi := makeQuery(3005.0)
	count := idx.Search(qf, qi, scratch)
	if count != 0 {
		t.Errorf("expected 0 fraud neighbours near legit cluster, got %d", count)
	}
}

func TestSearch_ReturnsBounded(t *testing.T) {
	path := buildTestIndex(t)
	defer os.Remove(path)

	idx, _ := Load(path)
	scratch := NewScratch()

	testVals := []float32{0, 10000, 5000, -10000, 3000, 8000}
	for _, val := range testVals {
		qf, qi := makeQuery(val)
		c := idx.Search(qf, qi, scratch)
		if c < 0 || c > KNN {
			t.Errorf("Search(val=%.0f) = %d, want [0, %d]", val, c, KNN)
		}
	}
}

// --- Benchmarks ---

func BenchmarkSearch(b *testing.B) {
	path := buildTestIndex(b)
	defer os.Remove(path)

	idx, err := Load(path)
	if err != nil {
		b.Fatalf("Load: %v", err)
	}

	scratch := NewScratch()
	qf, qi := makeQuery(3000.0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(qf, qi, scratch)
	}
}

// BenchmarkSearchLarger tests with a bigger synthetic index to better approximate
// real-world performance.
func BenchmarkSearchLarger(b *testing.B) {
	const nClusters = 64
	const perCluster = 2000

	vecs := make([][][Dims]float32, nClusters)
	lbls := make([][]uint8, nClusters)
	for c := 0; c < nClusters; c++ {
		base := float32(c) / float32(nClusters) * QuantScale
		vecs[c] = make([][Dims]float32, perCluster)
		lbls[c] = make([]uint8, perCluster)
		for i := 0; i < perCluster; i++ {
			for d := 0; d < Dims; d++ {
				vecs[c][i][d] = base
			}
			if i%2 == 0 {
				lbls[c][i] = 1
			}
		}
	}

	path := writeIVF2(b, vecs, lbls)
	defer os.Remove(path)

	idx, err := Load(path)
	if err != nil {
		b.Fatalf("Load: %v", err)
	}

	scratch := NewScratch()
	qf, qi := makeQuery(3000.0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(qf, qi, scratch)
	}
}

// Ensure math import is used.
var _ = math.MaxFloat32
