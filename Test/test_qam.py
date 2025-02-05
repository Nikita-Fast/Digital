from Source.qam import *
import pytest
import numpy as np


@pytest.mark.parametrize("bits,expected", [(np.array(list(bin(i)[2:]), dtype=int), i) for i in range(16)])
def test_bits_to_decimal(bits, expected):
    actual = bits_to_decimal(bits)
    assert actual == expected