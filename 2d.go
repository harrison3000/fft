package fft

func Compute2D(input [][]float32, scratch []complex64) ([][]complex64, error) {
	N := len(input)
	Nsq := N * N
	if e := checkLength("FFT 2D dimension", N); e != nil {
		return nil, e
	}

	if len(scratch) < Nsq {
		scratch = make([]complex64, Nsq)
	}

	wrkspc := subdivideslice(scratch, N)

	for y := range N {
		s := wrkspc[y]
		is := input[y]
		for x := range N {
			//TODO possible optimization: do the shuffling here
			s[x] = complex(is[x], 0)
		}
		Compute(s) //no need to check the error, already made the right size
	}

	//TODO possible optimization: do the shuffling here, per line
	transpose(wrkspc, N)
	N2p1 := N/2 + 1

	for y := range N2p1 {
		s := wrkspc[y]
		Compute(s)
	}

	return wrkspc[:N2p1], nil
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
