import numpy as np
import matplotlib.pyplot as plt


def hevisade(t):
    return np.where(t > 0, 1, np.where(t == 0, 1/2, 0))


if __name__ == '__main__':
    f0 = 5
    fs = 200
    # t = np.arange(-1, 1, 1/fs)
    t_start = -1
    t_end = 1
    t = np.linspace(t_start, t_end, (t_end - t_start) * fs)
    w0 = 2*np.pi * f0
    U = np.cos(w0/16/2 * t)
    # гармоническое колебание
    s1 = U * np.sin(w0 * t)

    a = 3
    # гауссов импульс
    s2 = U * np.exp(-a**2 * t**2)

    # экспоненциальный импульс
    s3 = np.where(t >= 0, U * np.exp(-a * t), 0)

    T = 0.25
    # финитный прямоугольный видеоимпульс
    s4 = np.where(np.abs(t) <= T, U, 0)

    # финитный треугольный видеоимпульс
    s5 = np.where(np.logical_and(0 <= t, t <= T), U/T * (T - t), 0)

    T = np.max(t) - np.min(t)
    # периодический сигнал
    s6 = np.concatenate([np.sin(w0 * t + 2*np.pi * k * T) for k in range(-2, 2)])

    # функция хевисайда
    s7 = hevisade(t)

    def phase_func(t):
        return np.ones(len(t)) * 2.5

    start_phase = - np.pi / 4
    # радиосигнал
    U = np.where(U >= 0.85, U, 0)
    u = U * np.cos(w0 * t + phase_func(t) + start_phase)

    plt.plot(u, marker='.')
    plt.plot(U)
    plt.show()
