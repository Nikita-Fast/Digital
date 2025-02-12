import numpy as np
import scipy
from matplotlib import pyplot as plt

n_fft = 16
opora = np.random.randint(-5, 5, 10)
s_opora = scipy.fft.fft(opora, n=n_fft)
signal = np.random.randint(-5, 5, 100)
s = np.random.randint(-5, 5, 10)

n = len(opora)
m = n_fft - n + 1

s1 = signal[:m]
ss1 = scipy.fft.fft(s1, n=n_fft)
x1 = scipy.fft.ifft(ss1 * s_opora, n=n_fft)

s2_start = m
s2 = signal[s2_start: s2_start+m]
ss2 = scipy.fft.fft(s2, n=n_fft)
x2 = scipy.fft.ifft(ss2 * s_opora, n=n_fft)


plt.plot(x1, label="x1")
plt.plot(np.pad(x2, (s2_start,0)), label="x2")
plt.plot(np.convolve(signal, opora, "valid"), label="expected")
plt.legend()
plt.show()
print(1)


