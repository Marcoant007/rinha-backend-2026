package ivf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

const (
	Dims   = 14
	Nlist  = 256 // number of clusters
	Nprobe = 20  // clusters to search per query
	K      = 7   // nearest neighbours
	Scale  = 10000.0
	Magic  = uint32(0x49564649) // "IVFI"
)

// Index holds the clustered reference vectors.
type Index struct {
	nlist     int
	centroids []float32 // nlist * Dims
	offsets   []uint32  // nlist + 1
	vectors   []int16   // n * Dims, sorted by cluster
	labels    []byte    // bit-packed fraud labels, sorted by cluster
	n         int
}

// Load reads an IVF index from the binary file produced by cmd/buildivf.
func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 8<<20)

	var m uint32
	if err := binary.Read(r, binary.LittleEndian, &m); err != nil {
		return nil, err
	}
	if m != Magic {
		return nil, fmt.Errorf("invalid IVF index: bad magic %08x", m)
	}

	var nl, n uint32
	binary.Read(r, binary.LittleEndian, &nl)
	binary.Read(r, binary.LittleEndian, &n)

	centroids := make([]float32, int(nl)*Dims)
	binary.Read(r, binary.LittleEndian, centroids)

	offsets := make([]uint32, int(nl)+1)
	binary.Read(r, binary.LittleEndian, offsets)

	vectors := make([]int16, int(n)*Dims)
	binary.Read(r, binary.LittleEndian, vectors)

	labelCount := (int(n) + 7) / 8
	labels := make([]byte, labelCount)
	binary.Read(r, binary.LittleEndian, labels)

	return &Index{
		nlist:     int(nl),
		centroids: centroids,
		offsets:   offsets,
		vectors:   vectors,
		labels:    labels,
		n:         int(n),
	}, nil
}

// Search returns the number of fraud neighbors (0..K) among the K nearest
// vectors found via IVF approximate search.
func (idx *Index) Search(query [Dims]float32) int {
	// 1. Find top Nprobe centroids (linear scan, tiny: 256 × 14 floats)
	var topCells [Nprobe]int
	var topDists [Nprobe]float32
	for i := range topDists {
		topDists[i] = math.MaxFloat32
	}

	for i := 0; i < idx.nlist; i++ {
		off := i * Dims
		d0 := query[0] - idx.centroids[off+0]
		d1 := query[1] - idx.centroids[off+1]
		d2 := query[2] - idx.centroids[off+2]
		d3 := query[3] - idx.centroids[off+3]
		d4 := query[4] - idx.centroids[off+4]
		d5 := query[5] - idx.centroids[off+5]
		d6 := query[6] - idx.centroids[off+6]
		d7 := query[7] - idx.centroids[off+7]
		d8 := query[8] - idx.centroids[off+8]
		d9 := query[9] - idx.centroids[off+9]
		d10 := query[10] - idx.centroids[off+10]
		d11 := query[11] - idx.centroids[off+11]
		d12 := query[12] - idx.centroids[off+12]
		d13 := query[13] - idx.centroids[off+13]
		d := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
			d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13

		if d < topDists[Nprobe-1] {
			pos := Nprobe - 1
			for pos > 0 && d < topDists[pos-1] {
				topDists[pos] = topDists[pos-1]
				topCells[pos] = topCells[pos-1]
				pos--
			}
			topDists[pos] = d
			topCells[pos] = i
		}
	}

	// 2. Quantize query to int16
	var q [Dims]int16
	for i, v := range query {
		if v < 0 {
			q[i] = int16(v*Scale - 0.5)
		} else {
			q[i] = int16(v*Scale + 0.5)
		}
	}

	// 3. Scan top clusters, maintain K-nearest heap (max-dist at index 0)
	var topDist [K]uint64
	var topIdx [K]int
	for i := range topDist {
		topDist[i] = math.MaxUint64
	}

	for _, cellID := range topCells {
		start := int(idx.offsets[cellID])
		end := int(idx.offsets[cellID+1])

		for i := start; i < end; i++ {
			off := i * Dims

			// Compute first 7 dims; early exit before full distance
			v0 := int32(idx.vectors[off+0]) - int32(q[0])
			v1 := int32(idx.vectors[off+1]) - int32(q[1])
			v2 := int32(idx.vectors[off+2]) - int32(q[2])
			v3 := int32(idx.vectors[off+3]) - int32(q[3])
			v4 := int32(idx.vectors[off+4]) - int32(q[4])
			v5 := int32(idx.vectors[off+5]) - int32(q[5])
			v6 := int32(idx.vectors[off+6]) - int32(q[6])
			dist := uint64(v0*v0) + uint64(v1*v1) + uint64(v2*v2) + uint64(v3*v3) +
				uint64(v4*v4) + uint64(v5*v5) + uint64(v6*v6)

			if dist >= topDist[K-1] {
				continue
			}

			v7 := int32(idx.vectors[off+7]) - int32(q[7])
			v8 := int32(idx.vectors[off+8]) - int32(q[8])
			v9 := int32(idx.vectors[off+9]) - int32(q[9])
			v10 := int32(idx.vectors[off+10]) - int32(q[10])
			v11 := int32(idx.vectors[off+11]) - int32(q[11])
			v12 := int32(idx.vectors[off+12]) - int32(q[12])
			v13 := int32(idx.vectors[off+13]) - int32(q[13])
			dist += uint64(v7*v7) + uint64(v8*v8) + uint64(v9*v9) + uint64(v10*v10) +
				uint64(v11*v11) + uint64(v12*v12) + uint64(v13*v13)

			if dist < topDist[K-1] {
				pos := K - 1
				for pos > 0 && dist < topDist[pos-1] {
					topDist[pos] = topDist[pos-1]
					topIdx[pos] = topIdx[pos-1]
					pos--
				}
				topDist[pos] = dist
				topIdx[pos] = i
			}
		}
	}

	// 4. Count fraud among K nearest
	fraudCount := 0
	for i := 0; i < K; i++ {
		if topDist[i] == math.MaxUint64 {
			continue
		}
		vi := topIdx[i]
		if (idx.labels[vi/8] & (1 << uint(vi%8))) != 0 {
			fraudCount++
		}
	}
	return fraudCount
}
