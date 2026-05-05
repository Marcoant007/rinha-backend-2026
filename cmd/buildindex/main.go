package main

import (
	"compress/gzip"
	"encoding/json"
	"io"
	"log"
	"os"

	"github.com/Marcoant007/rinha-2026/internal/kdtree"
)

// stride=6 → ~500K pontos de 3M (16% do dataset)
// mais pontos = melhor acurácia, K-D Tree mantém busca rápida
const stride = 6

type refEntry struct {
	Vector [kdtree.Dims]float64 `json:"vector"`
	Label  string               `json:"label"`
}

func main() {
	src := "resources/references.json.gz"
	dst := "index.bin"
	if len(os.Args) >= 3 {
		src = os.Args[1]
		dst = os.Args[2]
	}

	log.Printf("Lendo %s (stride=%d)...", src, stride)
	points, err := loadPoints(src)
	if err != nil {
		log.Fatalf("Erro ao ler referências: %v", err)
	}
	log.Printf("%d pontos carregados", len(points))

	log.Println("Construindo K-D Tree...")
	idx := kdtree.Build(points)

	log.Printf("Salvando índice em %s...", dst)
	if err := idx.Save(dst); err != nil {
		log.Fatalf("Erro ao salvar índice: %v", err)
	}
	log.Println("K-D Tree pronta!")
}

func loadPoints(path string) ([]kdtree.Point, error) {
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
		return nil, err
	}

	points := make([]kdtree.Point, 0, 500_000)
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
			var p kdtree.Point
			for d := 0; d < kdtree.Dims; d++ {
				p.Vec[d] = float32(entry.Vector[d])
			}
			p.Fraud = entry.Label == "fraud"
			points = append(points, p)
		}
		i++
	}
	return points, nil
}
