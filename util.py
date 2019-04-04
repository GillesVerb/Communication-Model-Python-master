import time
import numpy as np


def uint8_to_bit(uint8_list):
    return "".join([np.binary_repr(x, width=8) for x in uint8_list])


def chunks(s, n):
    """Produce `n`-character chunks from `s`."""
    for start in range(0, len(s), n):
        yield s[start:start+n]


def bit_to_uint8(bit_list):
    """Converts a bit string to a numpy uint8 array

    Arguments:
        bit_list {str} -- Bit list expecting string, otherwise the list is first converted to a but string

    Returns:
        np.ndarray -- uint8 typed ndarray
    """

    if type(bit_list) is not str:
        bit_list = "".join([x for x in bit_list])

    assert len(bit_list) % 8 == 0, "Provided bits length should be divisable by 8"

    splitted_list = [chunk for chunk in chunks(bit_list, 8)]

    return np.array([int(bits, 2) for bits in splitted_list], dtype=np.uint8)


class Time:
    def __init__(self):
        self.t = None

    def tic(self):
        self.t = time.time()

    def toc(self):
        assert self.t is not None, "Call tic() before toc()"
        diff = time.time()-self.t
        self.t = None
        return diff

    def toc_str(self):
        time_diff = self.toc()

        if time_diff < 1:
            time_diff *= 1000
            unit = "ms"
        elif time_diff < 100:
            unit = "s"
        else:
            time_diff /= 60
            unit = "min"

        return f"{time_diff:.2f} {unit}"

    def toc_print(self):
        print(self.toc_str())
