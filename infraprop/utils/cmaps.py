import os

import matplotlib as mpl
import numpy as np


def divergent_cmap():
    cmap_path = os.path.join(os.path.dirname(__file__), "cmaps", "divergent.npy")
    cmap = mpl.colors.ListedColormap(np.load(cmap_path))
    return cmap
