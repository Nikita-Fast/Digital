import pytest
import numpy as np
from Source.math_utils import *


@pytest.mark.parametrize("x,y,expected", )
def test_bits_to_decimal(x, y, expected):
    actual = xcorr(bits)
    assert actual == expected