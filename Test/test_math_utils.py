from Source.math_utils import *


def test_xcorr():
    for i in range(2, 10):
        for j in range(1, i):
            x = np.random.randn(i) + 1j * np.random.randn(i)
            y = np.random.randn(j) + 1j * np.random.randn(j)
            a = xcorr(x, y)
            b = np.correlate(x, y, "valid")
            assert np.all(np.isclose(a, b))


def test_convolve():
    for i in range(2, 10):
        for j in range(1, i):
            x = np.random.randn(i) + 1j * np.random.randn(i)
            y = np.random.randn(j) + 1j * np.random.randn(j)
            a = convolve(x, y)
            b = np.convolve(x, y, "valid")
            assert np.all(np.isclose(a, b))