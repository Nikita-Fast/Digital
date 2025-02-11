import numpy as np


class ConvolutionalCoder:
    def __init__(self, octal_polynomials: list[str]):
        pp = [np.array(list(bin(int(p, base=8))[2:]), dtype=int) for p in octal_polynomials]
        _len = np.max([len(p) for p in pp])
        self.p = np.array([np.pad(p, (_len - len(p), 0)) for p in pp])
        self.n = len(octal_polynomials)
        self.K = len(self.p[0])

    def step(self, bits):
        n_iters = len(bits) - self.K + 1
        res = np.zeros(n_iters * self.n, dtype=int)
        for i in range(n_iters):
            out = bits[i: i + self.K] @ self.p.T
            res[i*self.n: i*self.n + self.n] = out
        return res % 2


if __name__ == '__main__':
    cc = ConvolutionalCoder(["171", "133"])
    bits = np.random.randint(0, 2, 10)
    print(bits)
    print(cc.step(bits))
