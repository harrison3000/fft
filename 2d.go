package fft

type RealFFT2D struct {
	n         int
	scratch   []complex64
	workspace [][]complex64
	perms     []int
}

func (f *RealFFT2D) Init(N int) error {
	if e := checkLength("FFT 2D dimension", N); e != nil {
		return e
	}
	Nsq := N * N

	f.n = N
	f.scratch = make([]complex64, Nsq)
	f.workspace = subdivideslice(f.scratch, N)
	f.perms = make([]int, N)
	for i := range N {
		f.perms[i] = i
	}
	permute(f.perms)

	return nil
}

func ComputeReal2D(input [][]float32) ([][]complex64, error) {
	var s RealFFT2D
	if e := s.Init(len(input)); e != nil {
		return nil, e
	}
	c := s.Compute(input).Result()
	return c, nil
}

func (f *RealFFT2D) Compute(input [][]float32) *RealFFT2D {
	N := len(input)
	if f.n != N {
		panic("bad usage")
	}

	for r := range N {
		s := f.workspace[r]
		is := input[r]
		for x := range N {
			sp := f.perms[x]
			s[sp] = complex(is[x], 0)
		}
		fft64(s, false)
	}

	permute(f.workspace)
	transpose(f.workspace, N)
	N2p1 := N/2 + 1

	for r := range N2p1 {
		s := f.workspace[r]
		fft64(s, false)
	}

	return f
}

// Result returns... well.... the result!
// it doesn't include de symetric redundant data, having only N/2 + 1 rows
// so if the input is for example 64x64 the result will be 33x64
func (f *RealFFT2D) Result() [][]complex64 {
	N2p1 := f.n/2 + 1
	return f.workspace[:N2p1]
}

func subdivideslice[T any](scratch []T, N int) [][]T {
	r := make([][]T, 0, N)
	for i := 0; i < N*N; i += N {
		r = append(r, scratch[i:i+N])
	}
	return r
}

func transpose[T any](s [][]T, N int) {

	for y := range N {
		for x := range N {
			if y < x {
				s[x][y], s[y][x] = s[y][x], s[x][y]
			}
		}
	}
}
