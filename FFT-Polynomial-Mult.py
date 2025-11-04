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

def fft_radix4(x):
    N = len(x)
    if N == 1:
        return x
    if N % 4 != 0:
        return fft_radix2(x)
    x0 = fft_radix4(x[0::4])
    x1 = fft_radix4(x[1::4])
    x2 = fft_radix4(x[2::4])
    x3 = fft_radix4(x[3::4])
    W_N = np.exp(-2j * np.pi * np.arange(N) / N)
    X = np.zeros(N, dtype=complex)
    for k in range(N // 4):
        Wk, W2k, W3k = W_N[k], W_N[2*k], W_N[3*k]
        t0, t1, t2, t3 = x0[k], Wk*x1[k], W2k*x2[k], W3k*x3[k]
        X[k] = t0 + t1 + t2 + t3
        X[k + N//4] = t0 - 1j*t1 - t2 + 1j*t3
        X[k + N//2] = t0 - t1 + t2 - t3
        X[k + 3*N//4] = t0 + 1j*t1 - t2 - 1j*t3
    return X

def fft_dif_iterative(x):
    N = len(x)
    X = np.array(x, dtype=complex)
    stages = int(np.log2(N))
    for s in range(stages):
        step = 2 ** (s + 1)
        half = step // 2
        for k in range(0, N, step):
            for n in range(half):
                t = X[k + n] - X[k + n + half]
                X[k + n] = X[k + n] + X[k + n + half]
                X[k + n + half] = t * np.exp(-2j * np.pi * n / step)
    return X

def fft_split_radix(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    if N == 1:
        return x
    if N == 2:
        return np.array([x[0] + x[1], x[0] - x[1]], dtype=complex)
    if N % 4 != 0:
        return fft_radix2(x)
    X_even = fft_split_radix(x[0::2])
    X_odd1 = fft_split_radix(x[1::4])
    X_odd3 = fft_split_radix(x[3::4])
    k = np.arange(N // 4)
    Wk, W3k = np.exp(-2j * np.pi * k / N), np.exp(-2j * np.pi * 3 * k / N)
    X = np.zeros(N, dtype=complex)
    for i in range(N // 4):
        t1, t3 = Wk[i]*X_odd1[i], W3k[i]*X_odd3[i]
        A0, A1 = X_even[i], X_even[i + N//4]
        X[i] = A0 + t1 + t3
        X[i + N//2] = A0 - (t1 + t3)
        X[i + N//4] = A1 + 1j*(t1 - t3)
        X[i + 3*N//4] = A1 - 1j*(t1 - t3)
    return X

fft4_times, dif_times, split_times = [], [], []
for N in sizes:
    a, b = np.random.rand(N), np.random.rand(N)

    start = time.time(); poly_multiply_fft(a, b, fft_radix4); fft4_times.append(time.time()-start)
    start = time.time(); poly_multiply_fft(a, b, fft_dif_iterative); dif_times.append(time.time()-start)
    start = time.time(); poly_multiply_fft(a, b, fft_split_radix); split_times.append(time.time()-start)

plt.loglog(sizes, naive_times, 'o-', label='Naive O(N²)')
plt.loglog(sizes, fft2_times, 's-', label='Radix-2 FFT')
plt.loglog(sizes, fft4_times, 'd-', label='Radix-4 FFT')
plt.loglog(sizes, dif_times, 'x-', label='DIF Iterative FFT')
plt.loglog(sizes, split_times, '^-', label='Split Radix FFT')
plt.xlabel('N'); plt.ylabel('Time (s, log scale)')
plt.title('Full FFT Comparison')
plt.legend(); plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.show()

