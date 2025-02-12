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
        self._buf = []

    def valid_from_full_conv(self, s):
        start = self._len_opora - 1
        end = len(s)
        n_fft = len(s) + len(self._opora) - 1
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

    x = []
    x.append(correlator_fft.step(signal[:100]))
    x.append(correlator_fft.step(signal[100:300]))
    x.append(correlator_fft.step(signal[300:450]))
    x.append(correlator_fft.step(signal[450:]))
    # x.append(correlator_fft.step(signal))
    actual = np.concatenate(x)

    # s_opora = scipy.fft.fft(np.conj(opora[::-1]), n=len(signal))
    # s_signal = scipy.fft.fft(signal, n=len(signal))
    # actual = scipy.fft.ifft(s_signal * s_opora, n=len(signal))
    #
    # w_size = len(opora) - 1
    # s_opora2 = scipy.fft.fft(np.conj(opora[::-1]), n=272)
    # s1,s2 = signal[:272],signal[272:]
    # s_signal1 = scipy.fft.fft(s1)
    # s_signal2 = scipy.fft.fft(s2)
    # actual1 = scipy.fft.ifft(s_signal1 * s_opora2)
    # actual2 = scipy.fft.ifft(s_signal2 * s_opora2)
    #
    # plt.plot(actual1)
    # plt.plot(np.pad(actual2, (len(actual1),0)))
    # plt.show()

    plt.plot(actual, label="actual")
    plt.plot(expected, label="expected")
    # plt.plot(actual[-N:], label="correlator_fft_cut")
    # plt.plot(np.append(actual1,actual2), label="correlator_fft2")
    plt.legend()
    plt.show()
