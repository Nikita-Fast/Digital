import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    marker_len = 32
    marker = np.random.randint(0,2,marker_len)
    bits = np.concatenate([np.random.randint(0,2,16), marker, np.random.randint(0,2,16)])
    x = np.correlate(bits + 0.1*np.random.randn(len(bits)), marker, mode='full')
    i = np.argmax(x)
    print(i - marker_len)
    plt.plot(x)
    plt.axvline(x=i, color='r', label='corr peak')
    plt.legend()
    plt.show()