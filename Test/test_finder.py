import numpy as np
import pytest
from matplotlib import pyplot as plt

from Source.finder import Correlator, MatchedFilter


@pytest.mark.parametrize("slice_len", list(range(1, 500, 35)))
def test_correlator(slice_len):
    opora = np.sin(2 * np.pi * 3 * np.linspace(0, 1, 100))
    signal = np.pad(opora, (100, 123))
    signal = np.concatenate(np.repeat([signal], 7, axis=0))
    signal = signal + 0.5 * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))

    corr_calc = Correlator(opora)
    expected = corr_calc.step(signal)

    corr_calc.reset()
    i = 0
    slices = []
    while i < len(signal):
        slices.append(signal[i: i + slice_len])
        i += slice_len

    actual = np.concatenate([corr_calc.step(s) for s in slices])

    assert np.all(np.isclose(expected, actual))


@pytest.mark.parametrize("slice_len", list(range(1, 500, 35)))
def test_matched_filter(slice_len):
    opora = np.sin(2 * np.pi * 3 * np.linspace(0, 1, 100))
    signal = np.pad(opora, (100, 123))
    signal = np.concatenate(np.repeat([signal], 7, axis=0))
    signal = signal + 0.5 * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))

    matched_filter = MatchedFilter(opora)
    expected = matched_filter.step(signal)

    matched_filter.reset()
    i = 0
    slices = []
    while i < len(signal):
        slices.append(signal[i: i + slice_len])
        i += slice_len

    actual = np.concatenate([matched_filter.step(s) for s in slices])

    plt.plot(expected, label="single")
    plt.plot(actual, label="sliced")
    plt.legend()
    plt.show()

    assert np.all(np.isclose(expected, actual))