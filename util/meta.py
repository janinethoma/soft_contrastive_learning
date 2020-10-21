import numpy as np


def get_xy(meta):
    return np.array([[e, n] for e, n in zip(meta['easting'], meta['northing'])], dtype=float)
