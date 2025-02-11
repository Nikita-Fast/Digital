from Source.math_utils import *


def test_xcorr():
    for i in range(1, 10):
        for j in range(1, 10):
            x = np.random.randn(i) + 1j * np.random.randn(i)
            y = np.random.randn(j) + 1j * np.random.randn(j)
            a = xcorr(x, y)
            b = np.correlate(x, y, "full")
            assert np.all(np.isclose(a, b))