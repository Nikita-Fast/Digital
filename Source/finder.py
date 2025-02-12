import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fft import fft, ifft
from Source.math_utils import xcorr, convolve


class Correlator:
    def __init__(self, opora):
        self._opora = opora
        self._buf = []
        self._buf_capacity = len(opora) - 1

    def step(self, s):
        signal = np.append(self._buf, s)

        z = np.array([])
        if len(signal) >= len(self._opora):
            z = xcorr(signal, self._opora)
            # z = np.correlate(signal, self._opora, mode='valid')

        new_buf = np.append(self._buf, s)[-self._buf_capacity:]
        self._buf = new_buf

        return z

    def reset(self):
        self._buf = []


class MatchedFilter:
    def __init__(self, opora):
        self._h = np.conj(opora[::-1])
        self._buf = []
        self._buf_capacity = len(opora) - 1

    def step(self, s):
        signal = np.append(self._buf, s)

        z = np.array([])
        if len(signal) >= len(self._h):
            z = convolve(signal, self._h)

        new_buf = np.append(self._buf, s)[-self._buf_capacity:]
        self._buf = new_buf

        return z

    def reset(self):
        self._buf = []


class CorrelatorFFT:
    def __init__(self, opora, n_fft):
        self._opora = opora
        self._len_opora = len(opora)
        self._opora_spectre = fft(np.conj(opora)[::-1], n=n_fft)
        self._n_fft = n_fft
        self._buf_capacity = len(opora) - 1
        self._buf = np.zeros(self._buf_capacity)

    def valid_from_full_conv(self, s):
        start = self._len_opora - 1
        end = len(s)
        n_fft = len(s) + len(self._opora) - 1
        print(f"n_fft={n_fft}")
        full_conv = ifft(fft(s, n=n_fft) * fft(np.conj(self._opora)[::-1], n=n_fft))
        return full_conv[start: end]

    def step(self, s):
        prev_and_cur = np.append(self._buf, s)
        z = self.valid_from_full_conv(prev_and_cur)
        self._buf = prev_and_cur[-self._buf_capacity:]
        return z

    def reset(self):
        self._buf = []


if __name__ == '__main__':

    # opora = [1,2,-3,0.5]
    # signal = [1,-0.5,1.5,2,5,6,7,8,9]
    #
    # corr_calc = Correlator(opora)
    # expected = corr_calc.step(signal)
    # corr_calc.reset()
    #
    # z_list = []
    # z_list.append(corr_calc.step(signal[:3]))
    # z_list.append(corr_calc.step(signal[3:5]))
    # z_list.append(corr_calc.step([]))
    # z_list.append(corr_calc.step(signal[5:7]))
    # z_list.append(corr_calc.step([]))
    # z_list.append(corr_calc.step(signal[7:9]))
    # actual = np.concatenate(z_list)
    #
    # print(expected)
    # print(actual)
    # print(np.isclose(expected, actual))

    opora = np.sin(2 * np.pi * 3 * np.linspace(0, 1, 100))
    signal = np.pad(opora, (123, 321))
    signal = signal + 0.2 * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))

    correlator = Correlator(opora)
    correlator_fft = CorrelatorFFT(opora, n_fft=128)
    expected = correlator.step(signal)
    correlator.reset()

    x = [[]]
    n_fft_required = 199
    n_fft = n_fft_required - 99
    d_size = n_fft - 100 + 1
    print(d_size)
    for i in range(0, len(signal), d_size):
        x.append(correlator_fft.step(signal[i: i + d_size]))
    actual = np.concatenate(x)

    print(len(actual))

    plt.plot(actual[99:], label="actual")
    plt.plot(expected, label="expected")
    plt.legend()
    plt.show()
