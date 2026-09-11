import logging

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class AwgnChannel:
    def __init__(self, seed=42):
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

    def step(
        self,
        clean_signal,
        samples_per_symbol: int,
        bits_per_symbol: int,
        EbN0_db: float
    ):
        S_power = np.mean(np.abs(clean_signal)**2)
        N_power = S_power * (samples_per_symbol / bits_per_symbol) / 10 ** (EbN0_db / 10)

        sigma = np.sqrt(N_power / 2)
        logger.debug(f"{N_power=:.3f}, {sigma=:.3f}")

        n_real = self._rng.normal(loc=0, scale=sigma, size=len(clean_signal))
        n_imag = self._rng.normal(loc=0, scale=sigma, size=len(clean_signal))

        return clean_signal + n_real + 1j * n_imag


if __name__ == '__main__':
    rng = np.random.default_rng()

    target_noise_power = 3.5
    sigma = np.sqrt(target_noise_power / 2)

    # оценка параметров распределения
    n = rng.normal(loc=0, scale=sigma, size=100000) + 1j * rng.normal(loc=0, scale=sigma, size=100000)
    print(f"mu={np.mean(n):.3f}")
    print(f"sigma**2 = {sigma**2:.3f}")

    variance = np.var(n, ddof=0)
    variance_unbiased = np.var(n, ddof=1)
    print(f"Variance = {variance:.3f}")
    print(f"Variance unbiased = {variance_unbiased:.3f}")

    power = np.mean(np.abs(n)**2)
    print(f"Power = {power:.3f}")
