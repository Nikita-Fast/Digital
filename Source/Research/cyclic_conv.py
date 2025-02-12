import numpy as np
from scipy.fft import fft, ifft
from matplotlib import pyplot as plt


opora = np.random.randint(-5, 5, 10)
signal = np.random.randint(-5, 5, 100)


def valid_from_full_conv(s, opora):
    start = len(opora)-1
    end = len(s)
    # full_conv = np.convolve(s, opora)
    n_fft = len(s) + len(opora) - 1
    full_conv = ifft(fft(s, n=n_fft) * fft(opora, n=n_fft))
    return full_conv[start: end]


full_conv = np.convolve(signal, opora)
valid_conv = full_conv[len(opora)-1: len(signal)]
full_fft = ifft(fft(signal, n=len(signal)+len(opora)-1) * fft(opora, n=len(signal)+len(opora)-1))

s1 = signal[:5]
s2 = signal[5:12]
s3 = signal[12:]

prev = []
t1 = valid_from_full_conv(np.append(prev, s1), opora)
prev = np.append(prev, s1)[-(len(opora)-1):]
t2 = valid_from_full_conv(np.append(prev, s2), opora)
prev = np.append(prev, s2)[-(len(opora)-1):]
t3 = valid_from_full_conv(np.append(prev, s3), opora)


plt.plot(np.convolve(signal, opora, 'valid'), label="one part")
plt.plot(np.concatenate([t1,t2,t3]), label="valid_from_full_conv")
plt.legend()
plt.show()


