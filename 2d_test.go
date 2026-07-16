package fft

import (
	"fmt"
	"math/cmplx"
	"testing"

	dspfft "github.com/mjibson/go-dsp/fft"
)

func TestReal2DFFT(t *testing.T) {
	len := 16
	in := mk2dTestData[float32](len)
	in64 := mk2dTestData[float64](len)

	resdsp := dspfft.FFT2Real(in64) //we just trust go-dsp did its homework, lol
	res, e := ComputeReal2D(in)
	if e != nil {
		t.Errorf("Well, it should have worked, but didnt (%v)", e)
	}

	for x := range len {
		for y := range len/2 + 1 {
			ours := complex128(res[y][x])
			theirs := resdsp[x][y]

			abs := cmplx.Abs(ours - theirs)

			if abs > 0.0001 {
				t.Error("Too much difference")
			}
		}
	}
}

func mk2dTestData[T float32 | float64](N int) [][]T {
	in := subdivideslice(make([]T, N*N), N)
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

	naiveTranspose(a)

	al := fmt.Sprint(a)

	if al != "[[1 5 9 13] [2 6 10 14] [3 7 11 15] [4 8 12 16]]" {
		t.Errorf("unexpected results for small transpose: %v", al)
	}

	bigOne := ([256 * 256]float64)(floatRand(256 * 256))
	bigTwo := bigOne

	testa := subdivideslice(bigOne[:], 256)
	naiveTranspose(testa)

	testb := subdivideslice(bigTwo[:], 256)
	transpose(testb)

	if bigOne != bigTwo {
		t.Errorf("naive and optimized transpose gave different results!")
	}
}

func BenchmarkReal2DFFT(b *testing.B) {
	var s RealFFT2D
	s.Init(8)
	d := mk2dTestData[float32](8)

	b.Run("Small", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			s.Compute(d)
		}
	})

	s.Init(512)
	d = mk2dTestData[float32](512)
	b.Run("Big", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			s.Compute(d)
		}
	})

}
