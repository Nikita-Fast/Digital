import numpy as np
from matplotlib import pyplot as plt


def xcorr(s, opora):
    if len(s) < len(opora):
        raise RuntimeError("xcorr error! len(s) < len(opora)")

    N = max(len(s), len(opora)) - min(len(s), len(opora)) + 1
    z = np.zeros(N, dtype=complex)

    conj_opora = np.conj(opora)
    for i in range(N):
        z[i] = s[i:i+len(opora)] @ conj_opora
        # for j in range(len(opora)):
        #     z[i] += s[i + j] * np.conj(opora[j])

    return z


def convolve(s, opora):
    if len(s) < len(opora):
        raise RuntimeError("xcorr error! len(s) < len(opora)")

    N = max(len(s), len(opora)) - min(len(s), len(opora)) + 1
    z = np.zeros(N, dtype=complex)

    rev_opora = opora[::-1]
    for i in range(N):
        z[i] = s[i:i + len(opora)] @ rev_opora
        # for j in range(len(opora)):
        #     z[i] += s[i + j] * opora[len(opora)-1-j]

    return z


if __name__ == '__main__':
    opora = np.sin(2*np.pi*3*np.linspace(0, 1, 100))
    signal = np.pad(opora, (123, 321))
    signal = signal + 0.5 * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))

    a = xcorr(signal, opora)
    b = np.correlate(signal, opora, 'valid')

    c = xcorr(opora, signal)
    d = np.correlate(opora, signal, 'valid')

    plt.plot(a, label="my")
    plt.plot(b, label="numpy")
    plt.legend()
    plt.show()

    plt.plot(np.imag(c), label="my")
    plt.plot(np.imag(d), label="numpy")
    plt.legend()
    plt.show()
