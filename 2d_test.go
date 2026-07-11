package fft

import (
	"fmt"
	"testing"
)

func TestFFT2D(t *testing.T) {
	in := subdivideslice(make([]float32, 16), 4)
	in[0][0] = 0.5
	in[1][1] = 0.5
	in[2][2] = 0.5
	in[3][3] = 0.5

	s := make([]complex64, 16)

	_, e := Compute2D(in, s)
	if e != nil {
		t.Errorf("Well, it should have worked, but didnt (%v)", e)
	}

	//TODO actually test the return
}

func TestTranspose(t *testing.T) {
	a := [][]int{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
		{9, 10, 11, 12},
		{13, 14, 15, 16},
	}

	transpose(a, 4)

	al := fmt.Sprint(a)

	if al != "[[1 5 9 13] [2 6 10 14] [3 7 11 15] [4 8 12 16]]" {
		t.Errorf("unexpected results: %v", al)
	}
}
