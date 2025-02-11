import numpy as np
import matplotlib.pyplot as plt


def xcorr(x, y):
    z = np.zeros(len(x) + len(y) - 1, dtype=complex)
    N = max(len(x), len(y))

    for k in range(len(z)):
        z_k = 0
        for i in range(len(x)):
            y_index = i-k+N-1
            y_star = np.conj(y[y_index]) if 0 <= y_index < len(y) else 0
            z_k += x[i] * y_star
        z[k] = z_k
        print(k)
    return z


def xcorr2(x, y):
    N = max(len(x), len(y))
    x = np.pad(x, (0, N-len(x)))
    y = np.pad(y, (0, N-len(y)))
    corr_len = N
    res = []
    for i in range(corr_len):
        res.append(sum([np.conj(x[i]) * y[(m+i) % N] for m in range(N)]))
    return res


if __name__ == '__main__':
    # opora = np.sin(2*np.pi*3*np.linspace(0, 1, 100))
    # s = np.pad(opora, (234, 321))
    # s = s + 0.1 * (np.random.randn(len(s)) + 1j*np.random.randn(len(s)))

    s = [1, 2, 3, 4, 5]
    opora = [0, 1, 0.5]

    z = xcorr(s, opora)
    zz = np.correlate(s, opora, "full")

    tt = xcorr2(s, opora)

    # plt.plot(s)
    # plt.plot(opora)
    # plt.plot(np.abs(z))
    plt.plot(np.real(zz), label='numpy')
    plt.plot(np.real(z), label="my")
    plt.plot(np.real(tt), label="xui2")

    plt.legend()
    plt.show()
