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

sizes = [2**i for i in range(3, 12)]
naive_times = []
for N in sizes:
    a, b = np.random.rand(N), np.random.rand(N)
    start = time.time()
    res = poly_multiply_naive(a, b)
    naive_times.append(time.time() - start)
    print(f"N={N} | Naive result (first 5): {np.round(res[:5], 4)}")

plt.loglog(sizes, naive_times, 'o-', label='Naive O(N²)')
plt.xlabel('N')
plt.ylabel('Time (s, log scale)')
plt.title('Version 1: Naive Polynomial Multiplication')
plt.legend(); plt.show()
