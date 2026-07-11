package fft

import (
	"fmt"
	"math/cmplx"
	"testing"

	dspfft "github.com/mjibson/go-dsp/fft"
)

func TestFFT2D(t *testing.T) {
	in := mk2dTestData[float32]()
	in64 := mk2dTestData[float64]()

	s := make([]complex64, 16)

	resdsp := dspfft.FFT2Real(in64) //we just trust go-dsp did its homework, lol
	res, e := Compute2D(in, s)
	if e != nil {
		t.Errorf("Well, it should have worked, but didnt (%v)", e)
	}

	for x := range 8 {
		for y := range 5 {
			ours := complex128(res[y][x])
			theirs := resdsp[x][y]

			abs := cmplx.Abs(ours - theirs)

			if abs > 0.0001 {
				t.Error("Too much difference")
			}
		}
	}
}

func mk2dTestData[T float32 | float64]() [][]T {
	in := subdivideslice(make([]T, 256), 8)
	in[0][0] = 0.5
	in[0][1] = 0.5
	in[1][0] = 0.5
	in[1][1] = 0.5
	in[2][2] = 0.5
	in[3][3] = 0.5
	in[4][4] = 0.5

	in[5][2] = 0.5

	return in
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

//TODO benchmarks
