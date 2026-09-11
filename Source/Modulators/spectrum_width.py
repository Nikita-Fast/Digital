import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

np.random.seed(666)

qam4_constellation = np.ravel(np.array([-1, 1]) + 1j*np.array([1, -1]).reshape((-1, 1)))

idx = np.random.randint(0, 4, 100)
symbols = qam4_constellation[idx]

sps = 8
signal = np.zeros(len(symbols) * sps, dtype=complex)
signal[::sps] = symbols
# plt.plot(np.real(signal))
# plt.show()

# Raised Cosine Filter
beta = 0.25
num_taps = 101
fs = 1 # частота дискретизации
T = sps / fs # длительность символа
t = np.arange(num_taps) - (num_taps - 1) // 2

# Точки где возникает неопределенность 0/0 (предел неопределенности можно найти по правилу Лопиталя)
singular = np.isclose(np.abs(t), T / (2 * beta))
regular = ~singular

h = np.empty_like(t, dtype=float)

h[regular] = np.sinc(t[regular]/T) * np.cos(np.pi * beta * t[regular] / T) / (1 - (2 * beta * t[regular] / T)**2)
h[singular] = np.pi / 4 * np.sinc(1 / (2 * beta))
# E_h = np.sum(np.abs(h)**2)
# h = h / np.sqrt(E_h)
# h = h / np.sum(h)
# print(np.sum(h))

# plt.plot(t, h, label="h(t) приподнятого косинуса")
# plt.show()

# Применение фильтра
filtered = lfilter(h, 1, signal)

mask_re = np.isclose(np.abs(np.real(filtered)), 1)
mask_im = np.isclose(np.abs(np.imag(filtered)), 1)
re, im = np.real(filtered), np.imag(filtered)
plt.plot(re)
plt.plot(im)
plt.plot(np.flatnonzero(mask_re), re[mask_re], "*", label="symbols re")
plt.plot(np.flatnonzero(mask_im), im[mask_im], "*", label="symbols im")
plt.legend()
plt.show()

# S = np.fft.fftshift(np.fft.fft(filtered))
# S = S / np.max(S)
# f = np.fft.fftshift(np.fft.fftfreq(len(S)))
# plt.plot(f, 10*np.log10(np.abs(S)))
# plt.axvline(0.5 / 8, linestyle="--", alpha=0.3, color="red")
# plt.axvline(-0.5 / 8, linestyle="--", alpha=0.3, color="red")
# plt.show()

iq_samples = filtered[51::8]
plt.title(f"Сэмплирование QAM-4 в неправильный момент времени\nRoot Raised filter (beta={beta:.2f})")
plt.scatter(np.real(iq_samples), np.imag(iq_samples))
plt.show()

rx_idx = np.argmin(np.abs(iq_samples.reshape(-1, 1) - qam4_constellation), axis=1).ravel()
symbols_received = len(rx_idx)
symbol_errors = idx[:symbols_received] != rx_idx
print(f"Символов приянто:", symbols_received)
print(f"Символьных ошибок:", np.count_nonzero(symbol_errors))
