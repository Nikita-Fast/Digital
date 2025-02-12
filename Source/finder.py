import numpy as np
import matplotlib.pyplot as plt

from Source.math_utils import xcorr


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


if __name__ == '__main__':

    opora = [1,2,-3,0.5]
    signal = [1,-0.5,1.5,2,5,6,7,8,9]

    corr_calc = Correlator(opora)
    expected = corr_calc.step(signal)
    corr_calc.reset()

    z_list = []
    z_list.append(corr_calc.step(signal[:3]))
    z_list.append(corr_calc.step(signal[3:5]))
    z_list.append(corr_calc.step([]))
    z_list.append(corr_calc.step(signal[5:7]))
    z_list.append(corr_calc.step([]))
    z_list.append(corr_calc.step(signal[7:9]))
    actual = np.concatenate(z_list)

    print(expected)
    print(actual)
    print(np.isclose(expected, actual))


