import random

import numpy as np
from sklearn.neighbors import KDTree


def greedy(xy, d_max):
    """
    Sample xy until no xy is further than d_max from nearest sample
    """
    n = xy.shape[0]
    selected = [random.randrange(n)]
    remaining = np.setdiff1d(range(n), selected).tolist()
    while len(selected) < n:
        tree = KDTree(np.array([xy[i] for i in selected]))
        nn_dists, _ = tree.query([xy[i] for i in remaining], return_distance=True)

        i_max = np.argmax(nn_dists)
        print(nn_dists[i_max])

        if nn_dists[i_max] < d_max:
            break

        selected = selected + [remaining[i_max]]
        remaining = np.delete(remaining, i_max)

    return selected
