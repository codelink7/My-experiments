import numpy as np
import time
import matplotlib.pyplot as plt

def poly_multiply_naive(a, b):
    m, n = len(a), len(b)
    res = np.zeros(m + n - 1)
    for i in range(m):
        for j in range(n):
            res[i + j] += a[i] * b[j]
    return res

def fft_radix2(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft_radix2(x[0::2])
    odd = fft_radix2(x[1::2])
    W = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate([even + W * odd, even - W * odd])

def poly_multiply_fft(a, b, fft_func):
    N = 1
    while N < len(a) + len(b) - 1:
        N *= 2
    A = fft_func(np.pad(a, (0, N - len(a))))
    B = fft_func(np.pad(b, (0, N - len(b))))
    C = A * B
    c = np.conj(fft_func(np.conj(C))) / N
    return np.real(c[:len(a) + len(b) - 1])

fft2_times = []
for N in sizes:
    a, b = np.random.rand(N), np.random.rand(N)
    start = time.time()
    res_fft2 = poly_multiply_fft(a, b, fft_radix2)
    fft2_times.append(time.time() - start)
    print(f"N={N} | Radix-2 FFT result (first 5): {np.round(res_fft2[:5], 4)}")

plt.loglog(sizes, naive_times, 'o-', label='Naive O(N²)')
plt.loglog(sizes, fft2_times, 's-', label='Radix-2 FFT')
plt.xlabel('N'); plt.ylabel('Time (s, log scale)')
plt.title('Naive vs Radix-2 FFT Multiplication')
plt.legend(); plt.show()
