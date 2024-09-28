import numpy as np
import matplotlib.pyplot as plt


def amplitude(signal):
    return np.max(np.abs(signal))


if __name__ == '__main__':
    f0 = 2
    fs = 40
    t = np.arange(0, 1, 1/fs)
    x = np.sin(2*np.pi * f0 * t)

    print(f"x amp = {amplitude(x)}")

    x2 = 0.4 * np.sin(2*np.pi * 2*f0 * t)

    plt.plot(x + x2, marker='.')
    plt.show()
