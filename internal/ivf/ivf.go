package ivf

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"os"
	"unsafe"
)

const (
	Dims       = 14
	Nclusters  = 4096
	KNN        = 5
	QuantScale = float32(10000)

	FastNProbe  = 2
	BboxLanes   = 16 // 14 real dims + 2 zero-padding
	BlockLanes  = 8
	BlockStride = Dims * BlockLanes // 112 int16s per block
)

// magicV2 identifies the IVF2 binary format.
const magicV2 = "IVF2"

// BinaryMagic returns the 4-byte magic used in the IVF2 file header.
// Exported so cmd/buildivf can write a consistent header.
func BinaryMagic() []byte { return []byte(magicV2) }

// ExtremeWorstThreshold[c] is the i64 squared-distance above which a fast-tier
// result with fraudCount==c triggers a full sweep. 0 = disabled (trust fast tier).
// Calibrate with cmd/calibrate after building a new index.
var ExtremeWorstThreshold = [KNN + 1]int64{
	/* count=0 */ 92128083, // escalate if neighbors are far (uncertain legit)
	/* count=1 */ 4592068,  // escalate if neighbors are far (uncertain near-legit)
	/* count=2 */ 0,        // always escalate via count gate
	/* count=3 */ 0,        // always escalate via count gate
	/* count=4 */ 0,        // always escalate via count gate
	/* count=5 */ 91459358, // escalate if neighbors are far (uncertain fraud)
}

// Index is the runtime IVF index loaded from the binary produced by cmd/buildivf.
type Index struct {
	N         uint32
	K         uint32    // == Nclusters
	Blocks    uint32    // total block count across all clusters
	Centroids []float32 // AoS: Centroids[c*Dims+d], in QuantScale space
	Offsets   []uint32  // length K+1, in block units
	BboxMin   []int16   // cluster-major: BboxMin[c*BboxLanes+d]
	BboxMax   []int16   // same layout as BboxMin
	Radii     []float32 // length K, computed at load (not serialised)
	Labels    []uint8   // Labels[block*BlockLanes+lane], 1=fraud
	BlockData []int16   // BlockData[block*BlockStride + d*BlockLanes + lane]
}

// Scratch holds per-request reusable buffers. Allocate one from a sync.Pool.
type Scratch struct {
	CentroidDists [Nclusters]float32
	Picked        [FastNProbe]uint16
	Scanned       [Nclusters / 64]uint64
	top           top5
}

// NewScratch returns a freshly initialised Scratch for use in a sync.Pool.
func NewScratch() *Scratch { return &Scratch{} }

// ---------------------------------------------------------------------------
// top-5 max-heap (worst distance at index 0)
// ---------------------------------------------------------------------------

type top5 struct {
	dist  [KNN]int64
	label [KNN]uint8
	count int
}

func (t *top5) reset() {
	t.count = 0
	for i := range t.dist {
		t.dist[i] = math.MaxInt64
	}
}

func (t *top5) worst() int64 {
	if t.count < KNN {
		return math.MaxInt64
	}
	return t.dist[0]
}

func (t *top5) worstF32() float32 {
	w := t.worst()
	if w == math.MaxInt64 {
		return math.MaxFloat32
	}
	return float32(w)
}

func (t *top5) insert(d int64, label uint8) {
	if t.count < KNN {
		t.dist[t.count] = d
		t.label[t.count] = label
		t.count++
		if t.count == KNN {
			for i := KNN/2 - 1; i >= 0; i-- {
				top5SiftDown(t, i)
			}
		}
		return
	}
	if d >= t.dist[0] {
		return
	}
	t.dist[0] = d
	t.label[0] = label
	top5SiftDown(t, 0)
}

func top5SiftDown(t *top5, i int) {
	for {
		l, r, largest := 2*i+1, 2*i+2, i
		if l < KNN && t.dist[l] > t.dist[largest] {
			largest = l
		}
		if r < KNN && t.dist[r] > t.dist[largest] {
			largest = r
		}
		if largest == i {
			break
		}
		t.dist[i], t.dist[largest] = t.dist[largest], t.dist[i]
		t.label[i], t.label[largest] = t.label[largest], t.label[i]
		i = largest
	}
}

func (t *top5) fraudCount() uint8 {
	var c uint8
	for i := 0; i < t.count; i++ {
		c += t.label[i]
	}
	return c
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

// Load reads an IVF index from the binary file produced by cmd/buildivf.
func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 8<<20)

	// Header: magic(4) + version(4) + N(4) + K(4) + Blocks(4) + pad(12) = 32 bytes
	var hdr [32]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	if string(hdr[0:4]) != magicV2 {
		return nil, fmt.Errorf("bad magic %q (want %q)", hdr[0:4], magicV2)
	}
	// hdr[4:8] = version (unused for now)
	idx := &Index{
		N:      le32(hdr[8:]),
		K:      le32(hdr[12:]),
		Blocks: le32(hdr[16:]),
	}

	K := int(idx.K)
	blocks := int(idx.Blocks)

	idx.Centroids = make([]float32, K*Dims)
	if err := readF32(r, idx.Centroids); err != nil {
		return nil, fmt.Errorf("centroids: %w", err)
	}
	idx.Offsets = make([]uint32, K+1)
	if err := readU32(r, idx.Offsets); err != nil {
		return nil, fmt.Errorf("offsets: %w", err)
	}
	idx.BboxMin = make([]int16, K*BboxLanes)
	if err := readI16(r, idx.BboxMin); err != nil {
		return nil, fmt.Errorf("bboxMin: %w", err)
	}
	idx.BboxMax = make([]int16, K*BboxLanes)
	if err := readI16(r, idx.BboxMax); err != nil {
		return nil, fmt.Errorf("bboxMax: %w", err)
	}
	idx.Labels = make([]uint8, blocks*BlockLanes)
	if _, err := io.ReadFull(r, idx.Labels); err != nil {
		return nil, fmt.Errorf("labels: %w", err)
	}
	idx.BlockData = make([]int16, blocks*BlockStride)
	if err := readI16(r, idx.BlockData); err != nil {
		return nil, fmt.Errorf("blockdata: %w", err)
	}

	computeRadii(idx)
	return idx, nil
}

func computeRadii(idx *Index) {
	K := int(idx.K)
	idx.Radii = make([]float32, K)
	for c := 0; c < K; c++ {
		cBase := c * Dims
		start := idx.Offsets[c]
		end := idx.Offsets[c+1]
		if start == end {
			continue
		}
		var maxSq float32
		for b := start; b < end; b++ {
			bb := int(b) * BlockStride
			for lane := 0; lane < BlockLanes; lane++ {
				var sq float32
				for d := 0; d < Dims; d++ {
					v := float32(idx.BlockData[bb+d*BlockLanes+lane])
					diff := v - idx.Centroids[cBase+d]
					sq += diff * diff
				}
				if sq > maxSq {
					maxSq = sq
				}
			}
		}
		idx.Radii[c] = float32(math.Sqrt(float64(maxSq)))
	}
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

// Search returns the number of fraud neighbors (0..KNN) among the KNN nearest
// reference vectors found via IVF search with class-conditional escalation.
//
// qf must be the float32 query in QuantScale (10000) space.
// qi must be the int16 quantization of the same vector.
// scratch must not be shared across goroutines.
func (idx *Index) Search(qf *[Dims]float32, qi *[Dims]int16, scratch *Scratch) int {
	K := int(idx.K)

	// 1. Score all K centroids in float32 (QuantScale) space.
	for c := 0; c < K; c++ {
		base := c * Dims
		var d float32
		for dd := 0; dd < Dims; dd++ {
			x := qf[dd] - idx.Centroids[base+dd]
			d += x * x
		}
		scratch.CentroidDists[c] = d
	}

	// 2. Pick the top FastNProbe closest centroids.
	pickTop2(scratch.CentroidDists[:K], &scratch.Picked)

	// 3. Reset.
	scratch.top.reset()
	for i := range scratch.Scanned {
		scratch.Scanned[i] = 0
	}

	// 4. Fast tier: scan the closest FastNProbe clusters.
	for i := 0; i < FastNProbe; i++ {
		c := scratch.Picked[i]
		if c == ^uint16(0) {
			break
		}
		scanCluster(c, qi, idx, &scratch.top)
		scratch.Scanned[c/64] |= 1 << (c % 64)
	}

	// 5. Escalation check.
	count := scratch.top.fraudCount()
	needSweep := count == 2 || count == 3 || count == 4
	if !needSweep {
		thr := ExtremeWorstThreshold[count]
		if thr > 0 && scratch.top.worst() > thr {
			needSweep = true
		}
	}
	if !needSweep {
		return int(count)
	}

	// 6. Full sweep: triangle-inequality LB + AABB LB prune, then scan.
	for c := uint16(0); c < uint16(K); c++ {
		if scratch.Scanned[c/64]&(1<<(c%64)) != 0 {
			continue
		}
		// Triangle inequality: dist(q,x) >= (sqrt(centDist) - radius)²
		cd := scratch.CentroidDists[c]
		if cd > 0 {
			gap := float32(math.Sqrt(float64(cd))) - idx.Radii[c]
			if gap > 0 && gap*gap >= scratch.top.worstF32() {
				continue
			}
		}
		// AABB lower bound.
		if aabbLB(qi, idx.BboxMin, idx.BboxMax, int(c)) >= scratch.top.worst() {
			continue
		}
		scanCluster(c, qi, idx, &scratch.top)
	}

	return int(scratch.top.fraudCount())
}

func pickTop2(dists []float32, out *[FastNProbe]uint16) {
	out[0] = ^uint16(0)
	out[1] = ^uint16(0)
	d0, d1 := float32(math.MaxFloat32), float32(math.MaxFloat32)
	for i, d := range dists {
		if d < d0 {
			d1, out[1] = d0, out[0]
			d0, out[0] = d, uint16(i)
		} else if d < d1 {
			d1, out[1] = d, uint16(i)
		}
	}
}

func aabbLB(qi *[Dims]int16, bboxMin, bboxMax []int16, c int) int64 {
	base := c * BboxLanes
	var lb int64
	for d := 0; d < Dims; d++ {
		q := int64(qi[d])
		mn := int64(bboxMin[base+d])
		mx := int64(bboxMax[base+d])
		if q < mn {
			diff := mn - q
			lb += diff * diff
		} else if q > mx {
			diff := q - mx
			lb += diff * diff
		}
	}
	return lb
}

func scanCluster(c uint16, qi *[Dims]int16, idx *Index, top *top5) {
	start := int(idx.Offsets[c])
	end := int(idx.Offsets[c+1])

	for b := start; b < end; b++ {
		bb := b * BlockStride
		lb := b * BlockLanes
		worst := top.worst()

		for lane := 0; lane < BlockLanes; lane++ {
			// Early exit on first 7 dims.
			var ssd int64
			for d := 0; d < 7; d++ {
				diff := int64(qi[d]) - int64(idx.BlockData[bb+d*BlockLanes+lane])
				ssd += diff * diff
			}
			if ssd >= worst {
				continue
			}
			for d := 7; d < Dims; d++ {
				diff := int64(qi[d]) - int64(idx.BlockData[bb+d*BlockLanes+lane])
				ssd += diff * diff
			}
			if ssd < worst {
				top.insert(ssd, idx.Labels[lb+lane])
				worst = top.worst()
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Binary I/O helpers
// ---------------------------------------------------------------------------

func le32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func readF32(r io.Reader, s []float32) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 4*len(s))
	_, err := io.ReadFull(r, b)
	return err
}

func readU32(r io.Reader, s []uint32) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 4*len(s))
	_, err := io.ReadFull(r, b)
	return err
}

func readI16(r io.Reader, s []int16) error {
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), 2*len(s))
	_, err := io.ReadFull(r, b)
	return err
}
