import io
import pstats
from pstats import SortKey

import numpy as np
import matplotlib.pyplot as plt
import cProfile


def bits_to_decimal_symbols(bits, bits_per_symbol: int):
    tmp = bits.reshape((len(bits)//bits_per_symbol, bits_per_symbol))
    return np.fromiter(map(bits_to_decimal, tmp), dtype=int)


def bits_to_decimal(bits):
    n = len(bits)
    return np.sum(bits * (2 ** np.arange(n-1, -1, -1)))


def decimals_to_bits(decimals, bits_per_symbol):
    return np.concatenate([decimal_to_bits(d, bits_per_symbol) for d in decimals])


def decimal_to_bits(x, bits_number: int):
    irange = np.arange(bits_number - 1, -1, -1)
    return (x & (2 ** irange)) >> irange

def gen_constellation(n_points=4):
    if n_points not in [4, 16, 64, 256, 1024, 4096]:
        raise RuntimeError(f"QAM-{n_points} is not supported!")
    n = int(np.round(n_points ** 0.5))
    d = n - 1
    x = np.linspace(-d, d, d+1, dtype=int)
    y = 1j * x
    c = x.reshape((1,n)) - y.reshape((n,1))
    c = c.reshape((n_points))
    return c


class QAM:
    def __init__(self, M: int):
        self._c = gen_constellation(M)
        self._m = int(np.log2(M))

    def mod(self, bits: np.ndarray):
        if len(bits) % self._m != 0:
            raise RuntimeError("number of bits must be multiple of bits_per_symbol!")
        x = bits_to_decimal_symbols(bits, self._m)
        return self._c[x]

    def demod(self, noised: np.ndarray):
        euclid = np.abs(noised.reshape((len(noised), 1)) - self._c)
        decimals = np.argmin(euclid, axis=1)
        return decimals_to_bits(decimals, self._m)


if __name__ == '__main__':
    cp = cProfile.Profile()
    cp.enable()

    bits = np.random.choice([0, 1], 362880)
    qam4 = QAM(64)
    s = qam4.mod(bits)
    n = s + 0.05 * np.random.randn(len(s)) + 0.05j * np.random.randn(len(s))
    # plt.plot(np.real(n), np.imag(n), '*')
    # plt.show()
    demod_bits = qam4.demod(n)

    cp.disable()
    s = io.StringIO()
    sortby = SortKey.TIME
    ps = pstats.Stats(cp, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    print(np.abs(np.sum(bits - demod_bits)))

