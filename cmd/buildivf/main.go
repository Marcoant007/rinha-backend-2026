// cmd/buildivf builds an IVF2 index from references.json.gz.
// Usage: buildivf <src.json.gz> <dst.bin>
//
// Strategy:
//   1. Stream-parse ALL 3M reference vectors (custom scanner, no json.Decoder).
//   2. Train k-means on a 65536-point sample (6 Lloyd iterations).
//   3. Assign all 3M vectors to their nearest centroid.
//   4. Build block-major index with AABB bounding boxes per cluster.
//   5. Write IVF2 binary format.
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"
	"unsafe"

	"github.com/Marcoant007/rinha-2026/internal/ivf"
)

const (
	sampleSize = 65536 // k-means training sample
	maxIters   = 6     // Lloyd iterations
	rngSeed    = 42
)

// point holds a quantised vector and its fraud label.
type point struct {
	vec   [ivf.Dims]int16
	fraud bool
}

func main() {
	src := "resources/references.json.gz"
	dst := "index.bin"
	if len(os.Args) >= 3 {
		src, dst = os.Args[1], os.Args[2]
	}

	t0 := time.Now()
	log.Printf("Loading all points from %s ...", src)
	points, err := loadAllPoints(src)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	n := len(points)
	fraudTotal := 0
	for _, p := range points {
		if p.fraud {
			fraudTotal++
		}
	}
	log.Printf("Loaded %d points (%.1f%% fraud) in %v", n, 100*float64(fraudTotal)/float64(n), time.Since(t0))

	rng := rand.New(rand.NewSource(rngSeed))

	// Draw float32 sample for k-means (int16 → float32, still in QuantScale space).
	t1 := time.Now()
	sample := drawSample(points, sampleSize, rng)
	log.Printf("Drew %d-point sample in %v", sampleSize, time.Since(t1))

	// K-means++ init + Lloyd iterations on the sample.
	t2 := time.Now()
	log.Printf("Initialising %d centroids (k-means++)...", ivf.Nclusters)
	centroids := kmeansppInit(sample, rng)
	log.Printf("Running %d Lloyd iterations...", maxIters)
	for iter := 0; iter < maxIters; iter++ {
		assign := assignSample(sample, &centroids)
		updateCentroids(sample, assign, &centroids)
		log.Printf("  iter %d done", iter+1)
	}
	log.Printf("K-means done in %v", time.Since(t2))

	// Assign all 3M points to their nearest centroid.
	t3 := time.Now()
	log.Printf("Assigning %d vectors to nearest centroid...", n)
	assignment := assignAll(points, &centroids)
	log.Printf("AssignAll done in %v", time.Since(t3))

	// Build the index.
	t4 := time.Now()
	log.Printf("Building block-major index...")
	idx := buildIndex(points, assignment, &centroids)
	log.Printf("Build done in %v: %d blocks, %d clusters", time.Since(t4), idx.Blocks, idx.K)

	// Write.
	t5 := time.Now()
	log.Printf("Writing %s...", dst)
	if err := writeIndex(dst, idx); err != nil {
		log.Fatalf("write: %v", err)
	}
	info, _ := os.Stat(dst)
	log.Printf("Wrote %.1f MB in %v", float64(info.Size())/(1<<20), time.Since(t5))
	log.Printf("Total: %v", time.Since(t0))
}

// ---------------------------------------------------------------------------
// Streaming JSON parser
// ---------------------------------------------------------------------------

func loadAllPoints(path string) ([]point, error) {
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

	br := bufio.NewReaderSize(gz, 1<<20)

	pts := make([]point, 0, 3_100_000)
	const vecKey = `"vector":[`
	const lblKey = `,"label":"`

	for {
		if err := skipUntil(br, vecKey); err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("scan vector key (entry %d): %w", len(pts), err)
		}

		var p point
		for i := 0; i < ivf.Dims; i++ {
			v, last, err := readFloat(br)
			if err != nil {
				return nil, fmt.Errorf("dim %d (entry %d): %w", i, len(pts), err)
			}
			p.vec[i] = quantize(v)
			if i < ivf.Dims-1 && last == ']' {
				return nil, fmt.Errorf("unexpected ']' at dim %d (entry %d)", i, len(pts))
			}
		}

		if err := skipUntil(br, lblKey); err != nil {
			return nil, fmt.Errorf("scan label key (entry %d): %w", len(pts), err)
		}
		isFraud, err := readLabel(br)
		if err != nil {
			return nil, fmt.Errorf("label (entry %d): %w", len(pts), err)
		}
		p.fraud = isFraud
		pts = append(pts, p)
	}
	return pts, nil
}

func quantize(v float64) int16 {
	const sentinel = -1.0
	if v <= sentinel+1e-6 && v >= sentinel-1e-6 {
		return -10000
	}
	if v <= 0 {
		return 0
	}
	if v >= 1 {
		return 10000
	}
	return int16(v*10000 + 0.5)
}

func skipUntil(br *bufio.Reader, needle string) error {
	nlen := len(needle)
	matched := 0
	for {
		b, err := br.ReadByte()
		if err != nil {
			return err
		}
		if b == needle[matched] {
			matched++
			if matched == nlen {
				return nil
			}
			continue
		}
		matched = 0
		if b == needle[0] {
			matched = 1
		}
	}
}

func readFloat(br *bufio.Reader) (val float64, last byte, err error) {
	var v float64
	var sign float64 = 1
	state := 0 // 0=start 1=int 2=frac 3=expsign 4=exp
	var frac float64 = 0.1
	expSign := 1
	exp := 0
	for {
		b, rerr := br.ReadByte()
		if rerr != nil {
			return 0, 0, rerr
		}
		switch {
		case b == '-' && state == 0:
			sign = -1
		case b >= '0' && b <= '9':
			switch state {
			case 0, 1:
				state = 1
				v = v*10 + float64(b-'0')
			case 2:
				v += float64(b-'0') * frac
				frac *= 0.1
			case 3:
				expSign = 1
				exp = int(b - '0')
				state = 4
			case 4:
				exp = exp*10 + int(b-'0')
			}
		case b == '.':
			state = 2
		case b == 'e' || b == 'E':
			state = 3
		case b == '+' && state == 3:
			expSign = 1
		case b == '-' && state == 3:
			expSign = -1
		default:
			if exp != 0 {
				pow := 1.0
				for i := 0; i < exp; i++ {
					pow *= 10
				}
				if expSign < 0 {
					v /= pow
				} else {
					v *= pow
				}
			}
			return v * sign, b, nil
		}
	}
}

func readLabel(br *bufio.Reader) (bool, error) {
	var buf [5]byte
	if _, err := io.ReadFull(br, buf[:]); err != nil {
		return false, err
	}
	next, err := br.ReadByte()
	if err != nil {
		return false, err
	}
	if next != '"' {
		return false, fmt.Errorf("expected '\"' after label, got %q", next)
	}
	switch buf {
	case [5]byte{'f', 'r', 'a', 'u', 'd'}:
		return true, nil
	case [5]byte{'l', 'e', 'g', 'i', 't'}:
		return false, nil
	}
	return false, fmt.Errorf("unknown label %q", buf)
}

// ---------------------------------------------------------------------------
// K-means
// ---------------------------------------------------------------------------

// drawSample picks sampleSize random points and returns them as float32 in
// QuantScale space (int16 → float32 direct cast).
func drawSample(points []point, size int, rng *rand.Rand) []float32 {
	n := len(points)
	if size > n {
		size = n
	}
	sample := make([]float32, size*ivf.Dims)
	for i := 0; i < size; i++ {
		idx := rng.Intn(n)
		base := i * ivf.Dims
		for d := 0; d < ivf.Dims; d++ {
			sample[base+d] = float32(points[idx].vec[d])
		}
	}
	return sample
}

type centroidArr [ivf.Nclusters][ivf.Dims]float32

// kmeansppInit seeds Nclusters centroids using k-means++.
func kmeansppInit(sample []float32, rng *rand.Rand) centroidArr {
	n := len(sample) / ivf.Dims
	var centroids centroidArr

	first := rng.Intn(n)
	for d := 0; d < ivf.Dims; d++ {
		centroids[0][d] = sample[first*ivf.Dims+d]
	}

	d2 := make([]float64, n)
	for i := 0; i < n; i++ {
		d2[i] = sqDistSample(sample, i, centroids[0][:])
	}

	for k := 1; k < ivf.Nclusters; k++ {
		total := 0.0
		for _, v := range d2 {
			total += v
		}
		r := rng.Float64() * total
		acc := 0.0
		pick := n - 1
		for i, v := range d2 {
			acc += v
			if acc >= r {
				pick = i
				break
			}
		}
		for d := 0; d < ivf.Dims; d++ {
			centroids[k][d] = sample[pick*ivf.Dims+d]
		}
		// Update nearest distances.
		for i := 0; i < n; i++ {
			dd := sqDistSample(sample, i, centroids[k][:])
			if dd < d2[i] {
				d2[i] = dd
			}
		}
	}
	return centroids
}

func sqDistSample(sample []float32, i int, c []float32) float64 {
	base := i * ivf.Dims
	var s float64
	for d := 0; d < ivf.Dims; d++ {
		diff := float64(sample[base+d]) - float64(c[d])
		s += diff * diff
	}
	return s
}

func assignSample(sample []float32, centroids *centroidArr) []uint16 {
	n := len(sample) / ivf.Dims
	out := make([]uint16, n)
	for i := 0; i < n; i++ {
		base := i * ivf.Dims
		best := 0
		bestD := math.MaxFloat64
		for c := 0; c < ivf.Nclusters; c++ {
			var s float64
			for d := 0; d < ivf.Dims; d++ {
				diff := float64(sample[base+d]) - float64(centroids[c][d])
				s += diff * diff
				if s >= bestD {
					goto nextCentroid
				}
			}
			if s < bestD {
				bestD = s
				best = c
			}
		nextCentroid:
		}
		out[i] = uint16(best)
	}
	return out
}

func updateCentroids(sample []float32, assign []uint16, centroids *centroidArr) {
	var sums [ivf.Nclusters][ivf.Dims]float64
	var counts [ivf.Nclusters]uint32

	n := len(assign)
	for i := 0; i < n; i++ {
		c := assign[i]
		base := i * ivf.Dims
		for d := 0; d < ivf.Dims; d++ {
			sums[c][d] += float64(sample[base+d])
		}
		counts[c]++
	}
	for c := 0; c < ivf.Nclusters; c++ {
		if counts[c] == 0 {
			continue
		}
		inv := 1.0 / float64(counts[c])
		for d := 0; d < ivf.Dims; d++ {
			centroids[c][d] = float32(sums[c][d] * inv)
		}
	}
}

// assignAll assigns each of the n reference vectors to its nearest centroid.
// Uses partial-distance early exit (4 dims, then 8 dims, then all 14).
func assignAll(points []point, centroids *centroidArr) []uint16 {
	out := make([]uint16, len(points))
	for i, p := range points {
		best := 0
		bestD := float32(math.MaxFloat32)
		for c := 0; c < ivf.Nclusters; c++ {
			d0 := float32(p.vec[0]) - centroids[c][0]
			d1 := float32(p.vec[1]) - centroids[c][1]
			d2 := float32(p.vec[2]) - centroids[c][2]
			d3 := float32(p.vec[3]) - centroids[c][3]
			partial := d0*d0 + d1*d1 + d2*d2 + d3*d3
			if partial >= bestD {
				continue
			}
			d4 := float32(p.vec[4]) - centroids[c][4]
			d5 := float32(p.vec[5]) - centroids[c][5]
			d6 := float32(p.vec[6]) - centroids[c][6]
			d7 := float32(p.vec[7]) - centroids[c][7]
			partial += d4*d4 + d5*d5 + d6*d6 + d7*d7
			if partial >= bestD {
				continue
			}
			d8 := float32(p.vec[8]) - centroids[c][8]
			d9 := float32(p.vec[9]) - centroids[c][9]
			d10 := float32(p.vec[10]) - centroids[c][10]
			d11 := float32(p.vec[11]) - centroids[c][11]
			d12 := float32(p.vec[12]) - centroids[c][12]
			d13 := float32(p.vec[13]) - centroids[c][13]
			total := partial + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13
			if total < bestD {
				bestD = total
				best = c
			}
		}
		out[i] = uint16(best)
	}
	return out
}

// ---------------------------------------------------------------------------
// Index build
// ---------------------------------------------------------------------------

type builtIndex struct {
	N         uint32
	K         uint32
	Blocks    uint32
	Centroids []float32 // AoS: [c*Dims+d]
	Offsets   []uint32  // length K+1 (in blocks)
	BboxMin   []int16   // [c*BboxLanes+d]
	BboxMax   []int16
	Labels    []uint8  // [block*BlockLanes+lane]
	BlockData []int16  // [block*BlockStride + d*BlockLanes + lane]
}

func buildIndex(points []point, assignment []uint16, centroids *centroidArr) *builtIndex {
	n := len(points)
	K := ivf.Nclusters

	// 1. Bucket vector ids by cluster.
	buckets := make([][]uint32, K)
	for i := range buckets {
		buckets[i] = make([]uint32, 0, n/K+8)
	}
	for i, c := range assignment {
		buckets[c] = append(buckets[c], uint32(i))
	}

	// 2. Sort each bucket by distance to centroid (closest first for early-exit benefit).
	for c := 0; c < K; c++ {
		ids := buckets[c]
		if len(ids) < 2 {
			continue
		}
		dists := make([]float32, len(ids))
		for i, id := range ids {
			var sq float32
			for d := 0; d < ivf.Dims; d++ {
				diff := float32(points[id].vec[d]) - centroids[c][d]
				sq += diff * diff
			}
			dists[i] = sq
		}
		sort.Slice(ids, func(a, b int) bool { return dists[a] < dists[b] })
	}

	// 3. Compute offsets (in blocks).
	offsets := make([]uint32, K+1)
	var totalBlocks uint32
	for c := 0; c < K; c++ {
		offsets[c] = totalBlocks
		blocks := uint32((len(buckets[c]) + ivf.BlockLanes - 1) / ivf.BlockLanes)
		totalBlocks += blocks
	}
	offsets[K] = totalBlocks

	// 4. Flatten centroids (AoS).
	centFlat := make([]float32, K*ivf.Dims)
	for c := 0; c < K; c++ {
		copy(centFlat[c*ivf.Dims:], centroids[c][:])
	}

	// 5. AABB per cluster.
	bboxMin := make([]int16, K*ivf.BboxLanes)
	bboxMax := make([]int16, K*ivf.BboxLanes)
	for c := 0; c < K; c++ {
		base := c * ivf.BboxLanes
		if len(buckets[c]) == 0 {
			// Empty cluster: unreachable bbox.
			for d := 0; d < ivf.BboxLanes; d++ {
				bboxMin[base+d] = 32767
				bboxMax[base+d] = 32767
			}
			continue
		}
		for d := 0; d < ivf.Dims; d++ {
			bboxMin[base+d] = 32767
			bboxMax[base+d] = -32768
		}
		for _, id := range buckets[c] {
			for d := 0; d < ivf.Dims; d++ {
				v := points[id].vec[d]
				if v < bboxMin[base+d] {
					bboxMin[base+d] = v
				}
				if v > bboxMax[base+d] {
					bboxMax[base+d] = v
				}
			}
		}
	}

	// 6. Block-major layout + labels.
	labels := make([]uint8, int(totalBlocks)*ivf.BlockLanes)
	blockData := make([]int16, int(totalBlocks)*ivf.BlockStride)

	for c := 0; c < K; c++ {
		ids := buckets[c]
		blockStart := int(offsets[c])
		for i, id := range ids {
			b := blockStart + i/ivf.BlockLanes
			lane := i % ivf.BlockLanes
			bb := b * ivf.BlockStride
			for d := 0; d < ivf.Dims; d++ {
				blockData[bb+d*ivf.BlockLanes+lane] = points[id].vec[d]
			}
			if points[id].fraud {
				labels[b*ivf.BlockLanes+lane] = 1
			}
		}
		// Pad phantom lanes with int16 max (distance will be huge).
		used := len(ids) % ivf.BlockLanes
		if used != 0 {
			b := blockStart + len(ids)/ivf.BlockLanes
			bb := b * ivf.BlockStride
			for lane := used; lane < ivf.BlockLanes; lane++ {
				for d := 0; d < ivf.Dims; d++ {
					blockData[bb+d*ivf.BlockLanes+lane] = 32767
				}
				// label stays 0 (non-fraud) — phantom lane never beats top-5.
			}
		}
	}

	return &builtIndex{
		N:         uint32(n),
		K:         uint32(K),
		Blocks:    totalBlocks,
		Centroids: centFlat,
		Offsets:   offsets,
		BboxMin:   bboxMin,
		BboxMax:   bboxMax,
		Labels:    labels,
		BlockData: blockData,
	}
}

// ---------------------------------------------------------------------------
// Binary write
// ---------------------------------------------------------------------------

// Header: magic(4) + version(4) + N(4) + K(4) + Blocks(4) + pad(12) = 32 bytes
const headerSize = 32

func writeIndex(path string, idx *builtIndex) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 8<<20)

	var hdr [headerSize]byte
	copy(hdr[0:4], ivf.BinaryMagic())
	binary.LittleEndian.PutUint32(hdr[4:8], 1) // version
	binary.LittleEndian.PutUint32(hdr[8:12], idx.N)
	binary.LittleEndian.PutUint32(hdr[12:16], idx.K)
	binary.LittleEndian.PutUint32(hdr[16:20], idx.Blocks)
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}

	if err := writeF32(w, idx.Centroids); err != nil {
		return err
	}
	if err := writeU32(w, idx.Offsets); err != nil {
		return err
	}
	if err := writeI16(w, idx.BboxMin); err != nil {
		return err
	}
	if err := writeI16(w, idx.BboxMax); err != nil {
		return err
	}
	if _, err := w.Write(idx.Labels); err != nil {
		return err
	}
	if err := writeI16(w, idx.BlockData); err != nil {
		return err
	}
	return w.Flush()
}

func writeF32(w *bufio.Writer, s []float32) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 4*len(s))
	_, err := w.Write(b)
	return err
}

func writeU32(w *bufio.Writer, s []uint32) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 4*len(s))
	_, err := w.Write(b)
	return err
}

func writeI16(w *bufio.Writer, s []int16) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 2*len(s))
	_, err := w.Write(b)
	return err
}
