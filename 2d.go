package fft

type RealFFT2D struct {
	n         int
	scratch   []complex64
	workspace [][]complex64
}

func (f *RealFFT2D) Init(N int) error {
	if e := checkLength("FFT 2D dimension", N); e != nil {
		return e
	}
	Nsq := N * N
	if len(f.scratch) < Nsq {
		f.scratch = make([]complex64, Nsq)
		f.workspace = subdivideslice(f.scratch, N)
	}
	f.n = N

	return nil
}

func ComputeReal2D(input [][]float32) ([][]complex64, error) {
	var s RealFFT2D
	if e := s.Init(len(input)); e != nil {
		return nil, e
	}
	c := s.Compute(input).FastResult() //usar fullresult
	return c, nil
}

func (f *RealFFT2D) Compute(input [][]float32) *RealFFT2D {
	N := len(input)
	if f.n != N {
		panic("bad usage")
	}

	for y := range N {
		s := f.workspace[y]
		is := input[y]
		for x := range N {
			//TODO possible optimization: do the shuffling here
			s[x] = complex(is[x], 0)
		}
		Compute(s) //no need to check the error, already made the right size
	}

	//TODO possible optimization: do the shuffling here, per line
	transpose(f.workspace, N)
	N2p1 := N/2 + 1

	for y := range N2p1 {
		s := f.workspace[y]
		Compute(s)
	}

	return f
}

func (f *RealFFT2D) FastResult() [][]complex64 {
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
