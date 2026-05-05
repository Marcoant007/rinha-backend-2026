package main

import (
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"log"
	"os"
)

const dims = 14

type refJSON struct {
	Vector [dims]float64 `json:"vector"`
	Label  string        `json:"label"`
}

func encodeFloat(v float64) uint8 {
	return uint8((v + 1.0) * 127.5)
}

func main() {
	if len(os.Args) != 3 {
		log.Fatal("uso: convert <entrada.json.gz> <saída.bin>")
	}

	in, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	defer in.Close()

	gz, err := gzip.NewReader(in)
	if err != nil {
		log.Fatal(err)
	}
	defer gz.Close()

	dec := json.NewDecoder(gz)
	if _, err := dec.Token(); err != nil {
		log.Fatal(err)
	}

	const keepEvery = 10 // usa 1/10 dos vetores → ~300K

	vecs := make([]uint8, 0, 300_000*dims)
	lbls := make([]uint8, 0, 300_000)

	var r refJSON
	idx := 0
	for dec.More() {
		if err := dec.Decode(&r); err != nil {
			log.Fatal(err)
		}
		if idx%keepEvery == 0 {
			for _, v := range r.Vector {
				vecs = append(vecs, encodeFloat(v))
			}
			lbl := uint8(0)
			if r.Label == "fraud" {
				lbl = 1
			}
			lbls = append(lbls, lbl)
		}
		idx++
	}

	out, err := os.Create(os.Args[2])
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()

	n := uint32(len(lbls))
	binary.Write(out, binary.LittleEndian, n)
	out.Write(vecs)
	out.Write(lbls)

	log.Printf("Convertidos %d vetores → %s\n", n, os.Args[2])
}
