import numpy as np
import matplotlib.pyplot as plt
from Source.math_utils import xcorr
from scipy.signal import argrelmax


class Finder:
    def __init__(self, opora):
        self._opora = opora
        self._threshold = 0.5 * max(xcorr(opora, opora))
        self._buffer = np.zeros(len(self._opora)-1)
        self._processed = 0

    def reset(self):
        self._buffer = np.zeros(len(self._opora) - 1)
        self._processed = 0

    def step(self, signal):
        # z = xcorr(np.append(self._buffer, signal), self._opora)
        z = np.correlate(np.append(self._buffer, signal), self._opora, 'same')
        cor_maxs = argrelmax(z, order=len(self._opora))[0]
        cor_maxs = cor_maxs[z[cor_maxs] >= self._threshold]


        opora_starts = cor_maxs-len(self._buffer)-len(self._opora)//2+self._processed
        print(f"opora_starts={opora_starts}")

        plt.plot(z)
        plt.plot(cor_maxs, z[cor_maxs], '*')
        plt.hlines(self._threshold, color='r', xmin=0, xmax=len(z))

        if len(cor_maxs) > 0:
            clean = cor_maxs[-1]-len(self._buffer)-len(self._opora)//2+len(self._opora)
            n_zeros = min(max(clean - (len(signal) - len(self._opora)), 0), len(self._buffer))
            print(f"clean={clean}, n_zeros={n_zeros}")
            self._buffer = signal[-len(self._buffer):]
            self._buffer[:n_zeros] = 0
        else:
            self._buffer = signal[-len(self._buffer):]
        self._processed += len(signal)

        # plt.show()
        return opora_starts


if __name__ == '__main__':
    opora = np.sin(2 * np.pi * 3 * np.linspace(0, 1, 100))
    signal = np.pad(opora, (100, 123))
    signal = np.concatenate(np.repeat([signal], 15, axis=0))
    signal = signal + 0.2 * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))

    plt.plot(signal)
    plt.show()

    f = Finder(opora)
    expected = f.step(signal)

    f.reset()
    slices = []
    i = 0
    size = 530
    while i < len(signal):
        slices.append(signal[i: i+size])
        i += size

    actual = []
    for s in slices:
        actual.extend(f.step(s))

    print(expected)
    print(actual)

    assert np.array_equal(actual, expected)
