package fft

import (
	"math"
	"math/bits"
	"math/cmplx"
	"math/rand"
	"runtime"
	"sync/atomic"
	"testing"

	ktyefft "github.com/ktye/fft"
	dspfft "github.com/mjibson/go-dsp/fft"
	gonumfft "gonum.org/v1/gonum/dsp/fourier"
	scientificfft "scientificgo.org/fft"
)

// Slow is the simplest and slowest FFT transform, for testing purposes
func slowFFT(x []complex128) []complex128 {
	N := len(x)
	y := make([]complex128, N)
	for k := 0; k < N; k++ {
		for n := 0; n < N; n++ {
			phi := -2.0 * math.Pi * float64(k*n) / float64(N)
			s, c := math.Sincos(phi)
			y[k] += x[n] * complex(c, s)
		}
	}
	return y
}

func floatRand(N int) []float64 {
	x := make([]float64, N)
	for i := 0; i < N; i++ {
		x[i] = rand.NormFloat64()
	}
	return x
}

func complexRand(N int) []complex128 {
	x := make([]complex128, N)
	for i := 0; i < N; i++ {
		x[i] = complex(rand.NormFloat64(), rand.NormFloat64())
	}
	return x
}

func copyVector(v []complex128) []complex128 {
	y := make([]complex128, len(v))
	copy(y, v)
	return y
}

func copyToCplx64(v []complex128) []complex64 {
	y := make([]complex64, len(v))
	for k, v := range v {
		y[k] = complex64(v)
	}
	return y
}

func TestPrepare(t *testing.T) {
	// Test Prepare of non-powers of 2 returns InputSizeError
	checkIsInputSizeError(t, "Prepare(17)", Prepare(17))
	// Test Prepare for power of 2 up to 2^30
	for N := 2; N < (1 << 31); N <<= 1 {
		if err := Prepare(N); err != nil {
			t.Errorf("Prepare error on power of 2: %v", err)
		}
	}
}

func TestFFT(t *testing.T) {
	// Test FFT of non-powers of 2 returns InputSizeError
	checkIsInputSizeError(t, "FFT(complexRand(17))", Compute64(complexRand(17)))
	// Test FFT(x) == slowFFT(x) for power of 2 up to 2^10
	for N := 2; N < (1 << 11); N <<= 1 {
		x := complexRand(N)

		y1 := slowFFT(copyVector(x))
		y2 := copyVector(x)
		err := Compute64(y2)
		if err != nil {
			t.Errorf("FFT error: %v", err)
		}
		for i := 0; i < N; i++ {
			if e := cmplx.Abs(y1[i] - y2[i]); e > 1e-9 {
				t.Errorf("slowFFT and FFT differ: i=%d N=%d y1[%d]=%v y2[%d]=%v diff=%v\n", i, N, i, y1[i], i, y2[i], e)
			}
		}
	}
}

func TestFFTFloat32(t *testing.T) {
	nl := []struct {
		size   int
		elimit float64
	}{
		{1, 1e-6},
		{2, 1e-6},
		{4, 1e-6},
		{8, 1e-5},
		{128, 1e-4},
		{1024, 1e-3},
		{32768, 0.1},
	}

	for _, tt := range nl {
		N := tt.size
		xd := complexRand(N)
		xs := copyToCplx64(xd)

		Compute64(xd)
		err := Compute(xs)
		if err != nil {
			t.Fatalf("FFT error: %v", err)
		}

		var ee, emin, emax float64
		emin = 1e99
		for i := range N {
			vv := complex128(xs[i])
			e := cmplx.Abs(vv - xd[i])
			ee += e
			emin = min(emin, e)
			emax = max(emax, e)
		}
		ee /= float64(N)
		if ee > tt.elimit {
			t.Errorf("Compute and Compute64 differ: N=%d, avg_error=%v, min=%v, max%v (limit:%v)\n", N, ee, emin, emax, tt.elimit)
		}
	}
}

func TestIFFT(t *testing.T) {
	// Test IFFT of non-powers of 2 returns InputSizeError
	checkIsInputSizeError(t, "IFFT(complexRand(17))", InvCompute64(complexRand(17)))
	// Test FFT(IFFT(x)) == x for power of 2 up to 2^10
	for N := 2; N < (1 << 11); N <<= 1 {
		x := complexRand(N)
		y := copyVector(x)
		err := Compute64(y)
		if err != nil {
			t.Errorf("FFT error: %v", err)
		}
		err = InvCompute64(y)
		if err != nil {
			t.Errorf("IFFT error: %v", err)
		}
		for i := range x {
			if e := cmplx.Abs(x[i] - y[i]); e > 1e-9 {
				t.Errorf("inverse differs %d: %v %v\n", i, x[i], y[i])
			}
		}
	}
}

func TestPermute(t *testing.T) {
	shift := uint64(64)
	for n := 1; n < (1 << 11); n <<= 1 {
		x := complexRand(n)
		y := make([]complex128, n)
		copy(y, x)
		permute(x)
		for i := 0; i < n; i++ {
			ind := int(bits.Reverse64(uint64(i)) >> shift)
			if x[i] != y[ind] {
				t.Errorf("%d expected: x[%d] = %v, got: %v\n", n, i, y[ind], x[i])
			}
		}
		shift--
	}
}

var (
	benchmarks = []struct {
		size int
		name string
	}{
		{4, "Tiny (4)"},
		{128, "Small (128)"},
		{4096, "Medium (4096)"},
		{131072, "Large (131072)"},
		{4194304, "Huge (4194304)"},
		{16777216, "Massive (16777216)"},
	}
)

func BenchmarkSlowFFT(b *testing.B) {
	for _, bm := range benchmarks {
		if bm.size > 10000 {
			// Don't run sizes too big for slow
			continue
		}
		x := complexRand(bm.size)

		b.Run(bm.name, func(b *testing.B) {
			b.SetBytes(int64(bm.size * 16))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				slowFFT(x)
			}
		})
	}
}

func BenchmarkKtyeFFT(b *testing.B) {
	for _, bm := range benchmarks {
		if bm.size > 1048576 {
			// Max size for ktye's fft
			continue
		}
		f, err := ktyefft.New(bm.size)
		if err != nil {
			b.Errorf("fft.New error: %v", err)
		}
		x := complexRand(bm.size)

		b.Run(bm.name, func(b *testing.B) {
			b.SetBytes(int64(bm.size * 16))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				f.Transform(x)
			}
		})
	}
}

func BenchmarkGoDSPFFT(b *testing.B) {
	for _, bm := range benchmarks {
		dspfft.EnsureRadix2Factors(bm.size)
		x := complexRand(bm.size)

		b.Run(bm.name, func(b *testing.B) {
			b.SetBytes(int64(bm.size * 16))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				dspfft.FFT(x)
			}
		})
	}
}

func BenchmarkGonumFFT(b *testing.B) {
	for _, bm := range benchmarks {
		fft := gonumfft.NewCmplxFFT(bm.size)
		x := complexRand(bm.size)

		b.Run(bm.name, func(b *testing.B) {
			b.SetBytes(int64(bm.size * 16))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fft.Coefficients(x, x)
			}
		})
	}
}

func BenchmarkScientificFFT(b *testing.B) {
	for _, bm := range benchmarks {
		x := complexRand(bm.size)

		b.Run(bm.name, func(b *testing.B) {
			b.SetBytes(int64(bm.size * 16))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				scientificfft.Fft(x, false)
			}
		})
	}
}

func BenchmarkFFT(b *testing.B) {
	for _, bm := range benchmarks {
		x := complexRand(bm.size)

		b.Run(bm.name, func(b *testing.B) {
			b.SetBytes(int64(bm.size * 16))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Compute64(x)
			}
		})
	}
}

func BenchmarkFFTFloat32(b *testing.B) {
	for _, bm := range benchmarks {
		x128 := complexRand(bm.size)
		x := copyToCplx64(x128)

		b.Run(bm.name, func(b *testing.B) {
			b.SetBytes(int64(bm.size * 16))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Compute(x)
			}
		})
	}
}

func BenchmarkFFTParallel(b *testing.B) {
	for _, bm := range benchmarks {
		procs := runtime.GOMAXPROCS(0)
		x := complexRand(bm.size * procs)

		b.Run(bm.name, func(b *testing.B) {
			var idx uint64
			b.SetBytes(int64(bm.size * 16))
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				i := int(atomic.AddUint64(&idx, 1) - 1)
				y := x[i*bm.size : (i+1)*bm.size]
				for pb.Next() {
					Compute64(y)
				}
			})
		})
	}
}

func BenchmarkIFFT(b *testing.B) {
	for _, bm := range benchmarks {
		x := complexRand(bm.size)

		b.Run(bm.name, func(b *testing.B) {
			b.SetBytes(int64(bm.size * 16))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				InvCompute64(x)
			}
		})
	}
}

func BenchmarkIFFTParallel(b *testing.B) {
	for _, bm := range benchmarks {
		procs := runtime.GOMAXPROCS(0)
		x := complexRand(bm.size * procs)

		b.Run(bm.name, func(b *testing.B) {
			var idx uint64
			b.SetBytes(int64(bm.size * 16))
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				i := int(atomic.AddUint64(&idx, 1) - 1)
				y := x[i*bm.size : (i+1)*bm.size]
				for pb.Next() {
					InvCompute64(y)
				}
			})
		})
	}
}

func BenchmarkPermute(b *testing.B) {
	for _, bm := range benchmarks {
		x := complexRand(bm.size)
		b.Run(bm.name, func(b *testing.B) {
			b.SetBytes(int64(bm.size * 16))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				permute(x)
			}
		})
	}
}
