import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import ifft, fft


class MatchedFilterFFT:
    # Алгоритм overlap and sum взят из главы 8.7 книги Discrete-Time Signal Processing by Alan V. Oppenheim
    def __init__(self, ref_signal, n_fft):
        self.ref_signal_len = len(ref_signal)
        self.ref_signal_spectre = fft(np.conj(ref_signal)[::-1], n=n_fft)
        self.n_fft = n_fft
        self.buf = np.zeros(len(ref_signal)-1, dtype=complex)

    @property
    def max_input_size(self):
        return self.n_fft - self.ref_signal_len + 1

    def step(self, s):
        L = len(s)
        P = self.ref_signal_len

        if self.n_fft < 2*P:
            raise RuntimeError("Размер FFT слишком маленький")
        if L <= P:
            raise RuntimeError("Входной сигнал слишком короткий")

        z = ifft(fft(s, n=self.n_fft) * self.ref_signal_spectre)

        overlap_prev, curr_data, overlap_next = z[:P-1], z[P-1: L], z[L: L+P-1]
        res = np.zeros(L, dtype=complex)
        res[:P-1] = self.buf + overlap_prev
        res[P-1:] = curr_data
        self.buf = overlap_next

        return res

    def reset(self):
        self.buf = []


if __name__ == '__main__':
    np.random.seed(0)
    opora = np.random.choice([-1, 1], 100) + 1j*np.random.choice([-1, 1], 100)
    s = np.pad(opora, (400, 400))
    s = s + 0.01 * (np.random.randn(len(s)) + 1j*np.random.randn(len(s)))

    mf_fft  = MatchedFilterFFT(opora, n_fft=512)
    res = [[]]
    i = 0
    while i + mf_fft.max_input_size < len(s):
        res.append(mf_fft.step(s[i: i+mf_fft.max_input_size]))
        i += mf_fft.max_input_size
    res = np.concatenate(res)

    np_corr = np.correlate(s, opora, 'full')

    assert np.all(np.isclose(np_corr[:len(res)], res[:len(res)]))

    plt.plot(np.abs(np_corr), label="np.correlate")
    plt.plot(np.abs(res), label="MatchedFilterFFT")
    plt.legend()
    plt.show()
