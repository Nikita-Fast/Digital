import numpy as np
from scipy.special import sinc
import matplotlib.pyplot as plt


if __name__ == "__main__":
    T = 1
    t = np.linspace(-4 * T, 4 * T, 8 * 16)

    for beta in (0, 0.25, 0.5, 1):
        h = sinc(t/T) * (np.cos(np.pi * beta * t / T) / (1 - (2 * beta * t / T)**2))
        H = np.fft.fftshift(np.fft.fft(h))
        plt.plot(t, h, label=f"beta={beta}")

    plt.legend()
    plt.grid()
    plt.show()
