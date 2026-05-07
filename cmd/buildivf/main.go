// cmd/buildivf builds an IVF (Inverted File Index) from references.json.gz.
// Usage: buildivf <src.json.gz> <dst.bin>
//
// The output binary format is:
//   [magic: uint32] [nlist: uint32] [n: uint32]
//   [centroids: nlist*14*float32]
//   [offsets: (nlist+1)*uint32]
//   [vectors: n*14*int16]   (quantized: value * 10000)
//   [labels: ceil(n/8) bytes] (bit-packed, 1=fraud)
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"

	"github.com/Marcoant007/rinha-2026/internal/ivf"
)

const (
	stride = 6 // sample 1 of every 6 vectors → ~500K from 3M
	iters  = 25
)

type refEntry struct {
	Vector [ivf.Dims]float64 `json:"vector"`
	Label  string            `json:"label"`
}

type point struct {
	vec   [ivf.Dims]float32
	fraud bool
}

func main() {
	src := "resources/references.json.gz"
	dst := "index.bin"
	if len(os.Args) >= 3 {
		src = os.Args[1]
		dst = os.Args[2]
	}

	log.Printf("Loading points from %s (stride=%d)...", src, stride)
	points, err := loadPoints(src)
	if err != nil {
		log.Fatalf("error loading points: %v", err)
	}
	log.Printf("Loaded %d points", len(points))

	rng := rand.New(rand.NewSource(42))

	log.Printf("Initializing %d centroids (k-means++)...", ivf.Nlist)
	centroids := kmeansppInit(points, ivf.Nlist, rng)

	log.Printf("Running %d Lloyd iterations...", iters)
	t0 := time.Now()
	assignments := make([]int, len(points))
	for iter := 0; iter < iters; iter++ {
		changed := assignStep(points, centroids, assignments)
		updateStep(points, centroids, assignments, rng)
		log.Printf("  iter %2d: %d points moved", iter+1, changed)
		if changed < len(points)/500 { // < 0.2% → converged
			break
		}
	}
	log.Printf("K-means done in %v", time.Since(t0))

	// Final assignment
	assignStep(points, centroids, assignments)

	// Sort points by cluster
	type indexed struct {
		p  point
		cl int
	}
	indexed2 := make([]indexed, len(points))
	for i, p := range points {
		indexed2[i] = indexed{p, assignments[i]}
	}
	sort.Slice(indexed2, func(i, j int) bool {
		return indexed2[i].cl < indexed2[j].cl
	})

	// Compute cluster offsets
	n := len(indexed2)
	offsets := make([]uint32, ivf.Nlist+1)
	for _, x := range indexed2 {
		offsets[x.cl+1]++
	}
	for i := 1; i <= ivf.Nlist; i++ {
		offsets[i] += offsets[i-1]
	}

	// Quantize vectors to int16 and pack labels
	vectors := make([]int16, n*ivf.Dims)
	labelBytes := make([]byte, (n+7)/8)
	for i, x := range indexed2 {
		off := i * ivf.Dims
		for d := 0; d < ivf.Dims; d++ {
			v := float64(x.p.vec[d]) * ivf.Scale
			if v < 0 {
				vectors[off+d] = int16(v - 0.5)
			} else {
				vectors[off+d] = int16(v + 0.5)
			}
		}
		if x.p.fraud {
			labelBytes[i/8] |= 1 << uint(i%8)
		}
	}

	// Flatten centroids
	centroidsFlat := make([]float32, ivf.Nlist*ivf.Dims)
	for i, c := range centroids {
		copy(centroidsFlat[i*ivf.Dims:], c[:])
	}

	log.Printf("Writing index to %s...", dst)
	if err := writeIndex(dst, centroidsFlat, offsets, vectors, labelBytes, n); err != nil {
		log.Fatalf("error writing index: %v", err)
	}

	fraudTotal := 0
	for _, x := range indexed2 {
		if x.p.fraud {
			fraudTotal++
		}
	}
	log.Printf("Done: %d vectors, %d clusters, %.1f%% fraud",
		n, ivf.Nlist, 100*float64(fraudTotal)/float64(n))
}

// kmeansppInit initializes centroids with k-means++ using a subsample for
// distance computation to keep startup fast (O(k × sample) instead of O(k × n)).
func kmeansppInit(points []point, k int, rng *rand.Rand) [][ivf.Dims]float32 {
	n := len(points)
	centroids := make([][ivf.Dims]float32, 0, k)

	// First centroid: random
	centroids = append(centroids, points[rng.Intn(n)].vec)

	// Subsample for faster distance probability computation
	sampleN := min(8000, n)
	sample := make([]point, sampleN)
	for i := range sample {
		sample[i] = points[rng.Intn(n)]
	}
	dists := make([]float32, sampleN)

	for len(centroids) < k {
		var total float64
		for i, p := range sample {
			d := float32(math.MaxFloat32)
			for _, c := range centroids {
				if dd := sqDist(p.vec, c); dd < d {
					d = dd
				}
			}
			dists[i] = d
			total += float64(d)
		}

		r := rng.Float64() * total
		var cumsum float64
		chosen := sample[0].vec
		for i, d := range dists {
			cumsum += float64(d)
			if cumsum >= r {
				chosen = sample[i].vec
				break
			}
		}
		centroids = append(centroids, chosen)
	}
	return centroids
}

// assignStep assigns every point to its nearest centroid.
// Returns how many points changed assignment.
func assignStep(points []point, centroids [][ivf.Dims]float32, assignments []int) int {
	changed := 0
	for i, p := range points {
		best := 0
		bestDist := float32(math.MaxFloat32)
		for j, c := range centroids {
			// Early exit: compute first 4 dims and compare before finishing
			d0 := p.vec[0] - c[0]
			d1 := p.vec[1] - c[1]
			d2 := p.vec[2] - c[2]
			d3 := p.vec[3] - c[3]
			partial := d0*d0 + d1*d1 + d2*d2 + d3*d3
			if partial >= bestDist {
				continue
			}
			d4 := p.vec[4] - c[4]
			d5 := p.vec[5] - c[5]
			d6 := p.vec[6] - c[6]
			d7 := p.vec[7] - c[7]
			partial += d4*d4 + d5*d5 + d6*d6 + d7*d7
			if partial >= bestDist {
				continue
			}
			d8 := p.vec[8] - c[8]
			d9 := p.vec[9] - c[9]
			d10 := p.vec[10] - c[10]
			d11 := p.vec[11] - c[11]
			d12 := p.vec[12] - c[12]
			d13 := p.vec[13] - c[13]
			d := partial + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13
			if d < bestDist {
				bestDist = d
				best = j
			}
		}
		if assignments[i] != best {
			assignments[i] = best
			changed++
		}
	}
	return changed
}

// updateStep recomputes each centroid as the mean of its assigned points.
// Empty clusters are reinitialized to a random point.
func updateStep(points []point, centroids [][ivf.Dims]float32, assignments []int, rng *rand.Rand) {
	k := len(centroids)
	sums := make([][ivf.Dims]float64, k)
	counts := make([]int, k)

	for i, p := range points {
		c := assignments[i]
		counts[c]++
		for d := 0; d < ivf.Dims; d++ {
			sums[c][d] += float64(p.vec[d])
		}
	}

	for j := range centroids {
		if counts[j] == 0 {
			centroids[j] = points[rng.Intn(len(points))].vec
		} else {
			inv := 1.0 / float64(counts[j])
			for d := 0; d < ivf.Dims; d++ {
				centroids[j][d] = float32(sums[j][d] * inv)
			}
		}
	}
}

func sqDist(a, b [ivf.Dims]float32) float32 {
	var s float32
	for i := 0; i < ivf.Dims; i++ {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func loadPoints(path string) ([]point, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	dec := json.NewDecoder(gz)
	if _, err := dec.Token(); err != nil {
		return nil, fmt.Errorf("expected '[': %w", err)
	}

	pts := make([]point, 0, 500_000)
	var entry refEntry
	i := 0
	for dec.More() {
		if err := dec.Decode(&entry); err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		if i%stride == 0 {
			var p point
			for d := 0; d < ivf.Dims; d++ {
				p.vec[d] = float32(entry.Vector[d])
			}
			p.fraud = entry.Label == "fraud"
			pts = append(pts, p)
		}
		i++
	}
	return pts, nil
}

func writeIndex(path string, centroids []float32, offsets []uint32, vectors []int16, labels []byte, n int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 8<<20)
	binary.Write(w, binary.LittleEndian, ivf.Magic)
	binary.Write(w, binary.LittleEndian, uint32(ivf.Nlist))
	binary.Write(w, binary.LittleEndian, uint32(n))
	binary.Write(w, binary.LittleEndian, centroids)
	binary.Write(w, binary.LittleEndian, offsets)
	binary.Write(w, binary.LittleEndian, vectors)
	w.Write(labels)
	return w.Flush()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
