import logging

import numpy as np
from matplotlib import pyplot as plt
import scipy

from Source.Channels.awgn import AwgnChannel

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

i = np.array([3, 1, -1, -3])
constellation_qam16 = np.ravel(np.reshape(i, (len(i), 1)) - 1j * i)

bits_per_symbol = 4

res = {}
for ebn0 in np.arange(0, 15, 0.5):
    bit_stream = scipy.stats.bernoulli.rvs(p=0.5, size=1000000)
    bit_groups = bit_stream.reshape((-1, bits_per_symbol))

    powers_of_two = 2 ** np.arange(bits_per_symbol - 1, -1, -1)

    symbols = bit_groups @ powers_of_two
    qam_symbols = constellation_qam16[symbols]

    channel = AwgnChannel()
    channel_symbols = channel.step(qam_symbols, 1, bits_per_symbol, EbN0_db=ebn0)

    # Демодулируем
    distances = np.abs(channel_symbols.reshape(-1, 1) - constellation_qam16)
    idx = np.argmin(distances, axis=1)
    demodulated_symbols = constellation_qam16[idx]

    shifts = np.arange(bits_per_symbol - 1, -1, -1)
    demodulated_bits = np.ravel((idx.reshape(-1, 1) >> shifts) & 1)

    n_bit_errors = np.count_nonzero(bit_stream - demodulated_bits)
    ber = n_bit_errors / len(bit_stream)
    logging.debug(f"{ber=}")

    res[ebn0] = ber

x = res.keys()
y = list(res.values())
plt.yscale("log")
plt.plot(x, y, "*-")
plt.show()


