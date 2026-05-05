package kdtree

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
)

const (
	Dims       = 14
	BucketSize = 32
	K          = 5
	VisitCap   = 256
	magic      = uint32(0x4B445449) // "KDTI"
)

// Point é um vetor de referência com seu label de fraude.
type Point struct {
	Vec   [Dims]float32
	Fraud bool
}

// node é um nó da K-D Tree. Nós internos têm Dim >= 0; folhas têm Dim == -1.
type node struct {
	Dim      int32
	SplitVal float32
	Left     int32
	Right    int32
	Start    int32
	End      int32
}

// Index é a K-D Tree pronta para busca.
type Index struct {
	nodes  []node
	points []Point
}

// Build constrói a K-D Tree a partir de um slice de pontos.
func Build(points []Point) *Index {
	idx := &Index{
		points: make([]Point, len(points)),
		nodes:  make([]node, 0, len(points)/BucketSize*3),
	}
	copy(idx.points, points)
	idx.buildNode(0, len(idx.points))
	return idx
}

func (idx *Index) buildNode(lo, hi int) int32 {
	pos := int32(len(idx.nodes))
	idx.nodes = append(idx.nodes, node{})

	if hi-lo <= BucketSize {
		idx.nodes[pos] = node{Dim: -1, Start: int32(lo), End: int32(hi)}
		return pos
	}

	dim := widestDim(idx.points[lo:hi])

	pts := idx.points[lo:hi]
	sort.Slice(pts, func(i, j int) bool {
		return pts[i].Vec[dim] < pts[j].Vec[dim]
	})

	mid := (lo + hi) / 2
	splitVal := idx.points[mid].Vec[dim]

	left := idx.buildNode(lo, mid)
	right := idx.buildNode(mid, hi)

	idx.nodes[pos] = node{
		Dim:      int32(dim),
		SplitVal: splitVal,
		Left:     left,
		Right:    right,
	}
	return pos
}

func widestDim(pts []Point) int {
	var minV, maxV [Dims]float32
	for i := range minV {
		minV[i] = math.MaxFloat32
		maxV[i] = -math.MaxFloat32
	}
	for _, p := range pts {
		for d := 0; d < Dims; d++ {
			if p.Vec[d] < minV[d] {
				minV[d] = p.Vec[d]
			}
			if p.Vec[d] > maxV[d] {
				maxV[d] = p.Vec[d]
			}
		}
	}
	best, bestSpread := 0, float32(0)
	for d := 0; d < Dims; d++ {
		spread := maxV[d] - minV[d]
		if spread > bestSpread {
			bestSpread = spread
			best = d
		}
	}
	return best
}

// Search faz busca aproximada dos K vizinhos mais próximos e retorna a contagem de fraudes.
// visitCap limita quantas folhas são visitadas (troca precisão por velocidade).
func (idx *Index) Search(query [Dims]float32, visitCap int) int {
	var h knnHeap
	visited := 0
	idx.searchNode(0, query, &h, &visited, visitCap)
	return h.fraudCount()
}

func (idx *Index) searchNode(ni int32, query [Dims]float32, h *knnHeap, visited *int, visitCap int) {
	if *visited >= visitCap {
		return
	}
	n := idx.nodes[ni]

	if n.Dim == -1 {
		*visited++
		for i := n.Start; i < n.End; i++ {
			d := sqDist(idx.points[i].Vec, query)
			h.push(d, idx.points[i].Fraud)
		}
		return
	}

	diff := query[n.Dim] - n.SplitVal
	var near, far int32
	if diff <= 0 {
		near, far = n.Left, n.Right
	} else {
		near, far = n.Right, n.Left
	}

	idx.searchNode(near, query, h, visited, visitCap)

	if diff*diff < h.maxDistSq() {
		idx.searchNode(far, query, h, visited, visitCap)
	}
}

func sqDist(a, b [Dims]float32) float32 {
	var s float32
	for i := 0; i < Dims; i++ {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// knnHeap é um max-heap de K vizinhos por distância.
type knnHeap struct {
	items [K]heapItem
	size  int
}

type heapItem struct {
	distSq float32
	fraud  bool
}

func (h *knnHeap) push(distSq float32, fraud bool) {
	if h.size < K {
		h.items[h.size] = heapItem{distSq, fraud}
		h.size++
		if h.size == K {
			for i := K/2 - 1; i >= 0; i-- {
				h.siftDown(i)
			}
		}
	} else if distSq < h.items[0].distSq {
		h.items[0] = heapItem{distSq, fraud}
		h.siftDown(0)
	}
}

func (h *knnHeap) siftDown(i int) {
	for {
		largest := i
		l, r := 2*i+1, 2*i+2
		if l < h.size && h.items[l].distSq > h.items[largest].distSq {
			largest = l
		}
		if r < h.size && h.items[r].distSq > h.items[largest].distSq {
			largest = r
		}
		if largest == i {
			break
		}
		h.items[i], h.items[largest] = h.items[largest], h.items[i]
		i = largest
	}
}

func (h *knnHeap) maxDistSq() float32 {
	if h.size < K {
		return math.MaxFloat32
	}
	return h.items[0].distSq
}

func (h *knnHeap) fraudCount() int {
	n := 0
	for i := 0; i < h.size; i++ {
		if h.items[i].fraud {
			n++
		}
	}
	return n
}

// Save serializa o índice para um arquivo binário.
func (idx *Index) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 4<<20)

	binary.Write(w, binary.LittleEndian, magic)
	binary.Write(w, binary.LittleEndian, uint32(len(idx.nodes)))
	binary.Write(w, binary.LittleEndian, uint32(len(idx.points)))

	for i := range idx.nodes {
		n := &idx.nodes[i]
		binary.Write(w, binary.LittleEndian, n.Dim)
		binary.Write(w, binary.LittleEndian, n.SplitVal)
		binary.Write(w, binary.LittleEndian, n.Left)
		binary.Write(w, binary.LittleEndian, n.Right)
		binary.Write(w, binary.LittleEndian, n.Start)
		binary.Write(w, binary.LittleEndian, n.End)
	}

	for i := range idx.points {
		p := &idx.points[i]
		binary.Write(w, binary.LittleEndian, p.Vec)
		var fraud uint8
		if p.Fraud {
			fraud = 1
		}
		w.WriteByte(fraud)
	}

	return w.Flush()
}

// Load carrega o índice de um arquivo binário.
func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 4<<20)

	var m uint32
	if err := binary.Read(r, binary.LittleEndian, &m); err != nil {
		return nil, err
	}
	if m != magic {
		return nil, fmt.Errorf("índice inválido: magic incorreto")
	}

	var numNodes, numPoints uint32
	binary.Read(r, binary.LittleEndian, &numNodes)
	binary.Read(r, binary.LittleEndian, &numPoints)

	nodes := make([]node, numNodes)
	for i := range nodes {
		n := &nodes[i]
		binary.Read(r, binary.LittleEndian, &n.Dim)
		binary.Read(r, binary.LittleEndian, &n.SplitVal)
		binary.Read(r, binary.LittleEndian, &n.Left)
		binary.Read(r, binary.LittleEndian, &n.Right)
		binary.Read(r, binary.LittleEndian, &n.Start)
		binary.Read(r, binary.LittleEndian, &n.End)
	}

	points := make([]Point, numPoints)
	for i := range points {
		p := &points[i]
		binary.Read(r, binary.LittleEndian, &p.Vec)
		b, _ := r.ReadByte()
		p.Fraud = b == 1
	}

	return &Index{nodes: nodes, points: points}, nil
}
