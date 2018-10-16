# -*- coding: utf-8 -*-

# standard package
import os, sys
# third-party package
import tensorflow as tf
import numpy as np
# module in this repo
from np_train_data_100Hz import read_train_data

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
if FILE_DIR not in sys.path:
    sys.path.append(FILE_DIR)

tf.logging.set_verbosity(tf.logging.INFO)

# Parameters
learning_rate = 1e-3


global_steps = 2000
# batch_size = 158
# display_step = 100

# Network Parameters
# n_input = 1400
# n_classes = 2
# dropout = 0.25


# Application logic below
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 50, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        input_layer, filters=32, kernel_size=5,
        padding="same", activation=tf.nn.sigmoid)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    # Convolutional Layers #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1, filters=64, kernel_size=5,
        padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

    # Dense Layer
    # pool2_flat = tf.reshape(pool2, [-1, 25 * 14 * 64])
    pool2_flat = tf.layers.flatten(pool2)
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph, It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)  # global parameter
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    # 316 epochs, 50 samples, 28 channels, 500ms per epoch
    data_x, data_y = read_train_data()
    train_data = np.asarray(data_x[:200], dtype=np.float64)
    train_labels = np.asarray(data_y[:200], dtype=np.int32)
    eval_data = np.asarray(data_x[200:], dtype=np.float64)
    eval_labels = np.asarray(data_y[200:], dtype=np.int32)

    # Create the Estimator
    model_dir = os.path.join(FILE_DIR, "TFModels", "tmp", "bbci_convnet_model")
    bbci_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    bbci_classifier.train(
        input_fn=train_input_fn,
        steps=global_steps,  # global parameter
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = bbci_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
