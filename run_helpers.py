import os

import numpy as np


def load_useless_features_file(file_path):
    useless_features_list = np.genfromtxt(
        file_path, delimiter=",", skip_header=1, dtype=str
    )

    return useless_features_list
