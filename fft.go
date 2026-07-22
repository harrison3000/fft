// Package fft provides a fast discrete Fourier transformation algorithm.
//
// Implemented is the 1-dimensional DFT of complex input data
// for with input lengths which are powers of 2.
//
// The algorithm is non-recursive, works in-place overwriting
// the input array, and requires O(1) additional space.
package fft

import (
	"math/bits"
	"math/cmplx"
)

// Prepare precomputes values used for FFT on a vector of length N.
// N must be a perfect power of 2, otherwise this will return an error.
//
// Deprecated: This no longer has any functionality
func Prepare(N int) error {
	return checkLength("FFT Input", N)
}

// Compute64 is the precision fast Fourier transform.
// This is done in-place (modifying the input array).
// Requires O(1) additional memory.
// len(x) must be a perfect power of 2, otherwise this will return an error.
func Compute64(x []complex128) error {
	if err := checkLength("FFT Input", len(x)); err != nil {
		return err
	}
	fft(x)
	return nil
}

// Compute implements the fast Fourier transform. In Float32 Format
// This is done in-place (modifying the input array).
// Requires O(1) additional memory.
// len(x) must be a perfect power of 2, otherwise this will return an error.
func Compute(x []complex64) error {
	if err := checkLength("FFT Input", len(x)); err != nil {
		return err
	}

	fft64(x)
	return nil
}

// ComputeFramesOverlap computes windowed FFT frames with overlap and averaging - with moving, windowing, copying and averaging
func ComputeFramesOverlap(x []complex64, overlapRatio float32, fftSize int) ([]complex64, error) {
	if err := checkLength("FFT Input", len(x)); err != nil {
		return nil, err
	}
	final := make([]complex64, fftSize)
	// Compute the number of frames
	overlapSamples := fftSize - int(float32(fftSize)*overlapRatio)
	numFrames := (len(x) / overlapSamples) - (fftSize / overlapSamples)
	// Compute the FFT for each frame
	for i := 0; i < numFrames; i++ {
		// Compute the start and end indices for the frame
		start := i * overlapSamples
		end := start + fftSize
		// Copy the frame to a new slice
		frame := make([]complex64, fftSize)
		copy(frame, x[start:end])
		// Apply the window function
		ApplyWindow64(frame, Hanning)
		// Compute the FFT
		Compute(frame)
		// Final FFT is Average of all frames
		for j := 0; j < fftSize; j++ {
			final[j] += complex(real(frame[j])/float32(numFrames), imag(frame[j])/float32(numFrames))
		}
	}

	// Normalize Power Spectrum of the final result
	for i := 0; i < fftSize; i++ {
		final[i] = complex(real(final[i])/float32(fftSize), imag(final[i])/float32(fftSize))
		final[i] = complex64(10 * cmplx.Log10(complex128(final[i])))
	}
	return final, nil
}

// InvCompute64 implements the inverse fast Fourier transform.
// This is done in-place (modifying the input array).
// Requires O(1) additional memory.
// len(x) must be a perfect power of 2, otherwise this will return an error.
func InvCompute64(x []complex128) error {
	if err := checkLength("IFFT Input", len(x)); err != nil {
		return err
	}
	ifft(x)
	return nil
}

// InvCompute mplements the inverse fast Fourier transform.
// This is done in-place (modifying the input array).
// Requires O(1) additional memory.
// len(x) must be a perfect power of 2, otherwise this will return an error.
func InvCompute(x []complex64) error {
	if err := checkLength("IFFT Input", len(x)); err != nil {
		return err
	}
	ifft64(x)
	return nil
}

var sqrt64cache [256]complex64
var sqrt128cache [256]complex128

func init() {
	w := complex(0, -1)
	for n := range 64 {
		w = cmplx.Sqrt(w)
		sqrt128cache[n] = w
		sqrt64cache[n] = complex64(w)
	}
}

// fft does the actual work for FFT
func fft(x []complex128) {
	N := len(x)
	// Handle small N quickly
	switch N {
	case 1:
		return
	case 2:
		x[0], x[1] = x[0]+x[1], x[0]-x[1]
		return
	case 4:
		f := complex(imag(x[1])-imag(x[3]), real(x[3])-real(x[1]))
		x[0], x[1], x[2], x[3] = x[0]+x[1]+x[2]+x[3], x[0]-x[2]+f, x[0]-x[1]+x[2]-x[3], x[0]-x[2]-f
		return
	}
	// Reorder the input array.
	permute(x)
	// Butterfly
	// First 2 steps
	for i := 0; i < N; i += 4 {
		f := complex(imag(x[i+2])-imag(x[i+3]), real(x[i+3])-real(x[i+2]))
		x[i], x[i+1], x[i+2], x[i+3] = x[i]+x[i+1]+x[i+2]+x[i+3], x[i]-x[i+1]+f, x[i]-x[i+2]+x[i+1]-x[i+3], x[i]-x[i+1]-f
	}
	// Remaining steps
	sn := 0
	for n := 4; n < N; n <<= 1 {
		w := sqrt128cache[uint8(sn)]
		sn++
		for o := 0; o < N; o += (n << 1) {
			wj := complex(1, 0)
			for k := 0; k < n; k++ {
				i := k + o
				f := wj * x[i+n]
				x[i], x[i+n] = x[i]+f, x[i]-f
				wj *= w
			}
		}
	}
}

// ifft does the actual work for IFFT
func ifft(x []complex128) {
	N := len(x)
	// Reverse the input vector
	for i := 1; i < N/2; i++ {
		j := N - i
		x[i], x[j] = x[j], x[i]
	}

	// Do the transform.
	fft(x)

	// Scale the output by 1/N
	invN := complex(1.0/float64(N), 0)
	for i := 0; i < N; i++ {
		x[i] *= invN
	}
}

// ifft does the actual work for IFFT
func ifft64(x []complex64) {
	N := len(x)
	// Reverse the input vector
	for i := 1; i < N/2; i++ {
		j := N - i
		x[i], x[j] = x[j], x[i]
	}

	// Do the transform.
	fft64(x)

	// Scale the output by 1/N
	invN := complex(1.0/float32(N), 0)
	for i := 0; i < N; i++ {
		x[i] *= invN
	}
}

// permutate permutes the input vector using bit reversal.
// Uses an in-place algorithm that runs in O(N) time and O(1) additional space.
func permute[T any](x []T) {
	N := len(x)
	// Handle small N quickly
	switch N {
	case 1, 2:
		return
	case 4:
		x[1], x[2] = x[2], x[1]
		return
	case 8:
		x[1], x[4] = x[4], x[1]
		x[3], x[6] = x[6], x[3]
		return
	}
	shift := 64 - uint64(bits.Len64(uint64(N-1)))
	N2 := N >> 1
	for i := 0; i < N; i += 2 {
		ind := int(bits.Reverse64(uint64(i)) >> shift)
		// Skip cases where low bit isn't set while high bit is
		// This eliminates 25% of iterations
		if i < N2 && ind > i {
			x[i], x[ind] = x[ind], x[i]
		}
		ind |= N2 // Fast way to get int(bits.Reverse64(uint64(i+1)) >> shift) here
		if ind > i+1 {
			x[i+1], x[ind] = x[ind], x[i+1]
		}
	}
}

// fft does the actual work for FFT
func fft64(x []complex64) {
	N := len(x)
	// Handle small N quickly
	switch N {
	case 1:
		return
	case 2:
		x[0], x[1] = x[0]+x[1], x[0]-x[1]
		return
	case 4:
		f := complex(imag(x[1])-imag(x[3]), real(x[3])-real(x[1]))
		x[0], x[1], x[2], x[3] = x[0]+x[1]+x[2]+x[3], x[0]-x[2]+f, x[0]-x[1]+x[2]-x[3], x[0]-x[2]-f
		return
	}
	// Reorder the input array.
	permute(x)
	// Butterfly
	// First 2 steps
	for i := 0; i < N; i += 4 {
		f := complex(float32(imag(x[i+2])-imag(x[i+3])), float32(real(x[i+3])-real(x[i+2])))
		x[i], x[i+1], x[i+2], x[i+3] = x[i]+x[i+1]+x[i+2]+x[i+3], x[i]-x[i+1]+f, x[i]-x[i+2]+x[i+1]-x[i+3], x[i]-x[i+1]-f
	}
	// Remaining steps
	sn := 0
	for n := 4; n < N; n <<= 1 {
		w := sqrt64cache[uint8(sn)]
		sn++
		for o := 0; o < N; o += (n << 1) {
			wj := complex(float32(1), float32(0))
			for k := 0; k < n; k++ {
				i := k + o
				f := cplxMult(wj, x[i+n])
				x[i], x[i+n] = x[i]+f, x[i]-f
				wj = cplxMult(wj, w)
			}
		}
	}
}

// cplxMult multiplies 2 complex64 values
// why reimplement it manually? because the go compiler promotes complex64 multiplications to complex128 internally
// doing it manually gives like a 50% speedup on bigger transforms
func cplxMult(x, y complex64) complex64 {
	a, b := real(x), imag(x)
	c, d := real(y), imag(y)

	rr := a*c - b*d
	ri := a*d + b*c

	return complex(rr, ri)
}
