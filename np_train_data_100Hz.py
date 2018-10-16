# -*- coding: utf-8 -*-

# third-party package
import numpy as np


# 316 epochs, 50 samples, 28 channels, 500ms per epoch
def read_train_data():
    data_x = np.ndarray(shape=(316, 1400), dtype=np.float32)
    data_y = np.ndarray((316, 1), dtype=np.float16)

    with open("./inputData/sp1s_aa_train.csv", "rt") as f:
        for row, val in enumerate(f):
            data_x[row] = val.split()[1:]
            data_y[row] = val.split()[0]

    data_y = data_y.astype(np.int32)
    assert data_x.shape[0] == data_y.shape[0]
    return data_x, data_y
