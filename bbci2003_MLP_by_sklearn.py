# -*- coding: utf-8 -*-

# standard package
import os, sys, pickle
# third-party package
import numpy as np
from sklearn.neural_network import MLPClassifier
# module in this repo
from np_train_data_100Hz import read_train_data

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
if FILE_DIR not in sys.path:
    sys.path.append(FILE_DIR)

# Parameters
learning_rate = 1e-3

# 316 epochs, 50 samples, 28 channels, 500ms per epoch
data_x, data_y = read_train_data()

data_y = data_y.astype(np.uint8)
assert data_x.shape[0] == data_y.shape[0]

x_train, x_test = data_x[:256], data_x[256:]
y_train, y_test = data_y[:256], data_y[256:]


# Create fitted model
def neural_network(X: np.ndarray, Y, learning_rate):
    nn = MLPClassifier(hidden_layer_sizes=(128, 64), activation="logistic",
                       solver="sgd", learning_rate_init=learning_rate,
                       learning_rate="adaptive", verbose=True,
                       tol=1e-6, early_stopping=False, max_iter=15000,
                       shuffle=True)
    nn.fit(X, Y)
    return nn


# Make output to csv file
def do_output(model_dir, input_dir, output_dir):
    import pandas as pd
    with open(model_dir, "rb") as f:
        nn: MLPClassifier = pickle.load(f)
    test_data_x = np.ndarray(shape=(100, 1400), dtype=np.float64)
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

    # save nn model
    pkl_dir = os.path.join(FILE_DIR, "SKPickle", "sk_model.pkl")
    with open(pkl_dir, "wb") as f:
        pickle.dump(nn, f)

    # use nn model
    input_dir = os.path.join(FILE_DIR, "inputData", "sp1s_aa_test.csv")
    output_dir = os.path.join(FILE_DIR, "outputData",
                              "sp1s_aa_test_result_by_sklearn.csv")
    do_output(pkl_dir, input_dir, output_dir)
