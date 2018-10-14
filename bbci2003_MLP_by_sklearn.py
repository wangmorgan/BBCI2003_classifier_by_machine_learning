# -*- coding: utf-8 -*-

import os, sys, pickle
import numpy as np
from sklearn.neural_network import MLPClassifier

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
if FILE_DIR not in sys.path:
    sys.path.append(FILE_DIR)

# Parameters
learning_rate = 1e-3

# 316 epochs, 50 samples, 28 channels, 500ms per epoch
data_x = np.ndarray(shape=(316, 1400), dtype=np.float32)
data_y = np.ndarray((316,), dtype=np.float16)

train_data_csv_dir = os.path.join(FILE_DIR, "inputData", "sp1s_aa_train.csv")
with open(train_data_csv_dir, "rt") as f:
    for row, val in enumerate(f):
        data_x[row] = val.split()[1:]
        data_y[row] = val.split()[0]

# print(np.min(data_x), np.max(data_x))
# data_x = data_x.reshape(316, 50, 28)
data_y = data_y.astype(np.uint8)
assert data_x.shape[0] == data_y.shape[0]

x_train, x_test = data_x[:200], data_x[200:]
y_train, y_test = data_y[:200], data_y[200:]


# Create fitted model
def neural_network(X: np.ndarray, Y, learning_rate):
    nn = MLPClassifier(hidden_layer_sizes=(100,), activation="identity",
                       solver="lbfgs", learning_rate_init=learning_rate,)
    nn.fit(X, Y)
    return nn


def do_output(model_dir, input_dir, output_dir):
    import pandas as pd
    with open(model_dir, "rb") as f:
        nn: MLPClassifier = pickle.load(f)
    test_data_x = np.ndarray(shape=(100, 1400)).astype(np.float64)
    with open(input_dir, "r") as f:
        for i, val in enumerate(f):
            test_data_x[i] = val.split()
    output_data_y = nn.predict(test_data_x)
    print(test_data_x)
    print(output_data_y)

    data_frame = pd.DataFrame(test_data_x)
    print(data_frame)
    data_frame.insert(0, 0, output_data_y, allow_duplicates=True)
    print(data_frame)
    data_frame.to_csv(output_dir, header=False, index=False)


if __name__ == "__main__":
    nn = neural_network(x_train, y_train, learning_rate)
    y_pred = nn.predict(x_test)
    score = nn.score(x_test, y_test)
    print("Score: {0:.4f}".format(score))
    print(y_pred)

    # save nn
    pkl_dir = os.path.join(FILE_DIR, "SKPickle", "sk_model.pkl")
    with open(pkl_dir, "wb") as f:
        pickle.dump(nn, f)

    # use nn
    input_dir = os.path.join(FILE_DIR, "inputData", "sp1s_aa_test.csv")
    output_dir = os.path.join(FILE_DIR, "outputData", "sp1s_aa_test_result.csv")
    do_output(pkl_dir, input_dir, output_dir)
