import numpy as np
from matplotlib import pyplot as plt


def xcorr(x, y):
    x = np.pad(x, (len(y)-1, len(y)-1))
    z = np.zeros(len(x)-len(y)+1, dtype=complex)
    for i in range(len(x)-len(y)+1):
        for j in range(len(y)):
            z[i] += x[i+j] * np.conj(y[j])
    return z


if __name__ == '__main__':
    opora = np.sin(2*np.pi*3*np.linspace(0, 1, 100))
    signal = np.pad(opora, (123, 321))
    signal = signal + 0.1 * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))

    a = xcorr(signal, opora)
    b = np.correlate(signal, opora, 'full')
    plt.plot(a)
    plt.plot(b)
    print(np.argmax(a))
    print(np.argmax(b))
    plt.show()



