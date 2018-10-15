# -*- coding: utf-8 -*-

import os, sys
import tensorflow as tf
import numpy as np

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
if FILE_DIR not in sys.path:
    sys.path.append(FILE_DIR)

# Parameters
learning_rate = 1e-6
num_steps = 31600
batch_size = 158
display_step = 50

# Network Parameters
n_input = 1400
n_classes = 2
dropout = 0.25

sess = tf.Session()

# 316 epochs, 50 samples, 28 channels, 500ms per epoch
train_x = np.ndarray(shape=(316, 1400), dtype=np.float32)
train_y = np.ndarray((316, 1), dtype=np.float16)

with open("./inputData/sp1s_aa_train.csv", "rt") as f:
    for row, val in enumerate(f):
        train_x[row] = val.split()[1:]
        train_y[row] = val.split()[0]

print(np.min(train_x), np.max(train_x))
train_y = train_y.astype(np.uint8)
assert train_x.shape[0] == train_y.shape[0]

data_set = tf.data.Dataset.from_tensor_slices((train_x, train_y))
data_set = data_set.repeat()
data_set = data_set.batch(batch_size)
data_set = data_set.prefetch(batch_size)

iterator = data_set.make_initializable_iterator()
sess.run(iterator.initializer)

# Neural Net Input (matrix, labels)
X, Y = iterator.get_next()


# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # ndarray input is a 1-D vector of 1400 features (50samples * 28channels)
        # reshape to match format [Samples x Channel]
        # Tensor input become 3-D: [Batch Size, Samples, Channel]
        x = tf.reshape(x, shape=[-1, 50, 28])

        x = tf.fake_quant_with_min_max_args(x, -255, 255, num_bits=16)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv1d(x, 32, 5, padding='SAME',
                                 activation=tf.nn.sigmoid,)

        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling1d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv1d(conv1, 64, 2, padding='SAME',
                                 activation=tf.nn.sigmoid,)

        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling1d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.layers.flatten(conv2)

        # Fully connected layer
        fc1 = tf.layers.dense(fc1, 1000)
        # Apply Dropout (if ts_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(X, n_classes, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights, but has
# different behavior for 'dropout' (not applied).
logits_test = conv_net(X, n_classes, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(
#     learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Saver
saver = tf.train.Saver()
save_dir = os.path.join(FILE_DIR, "TFModels", "TF_CNN_Model.cpkt")

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Run the initializer
sess.run(init)

# Training cycle
for step in range(1, num_steps + 1):

    # Run optimization
    sess.run(train_op)

    if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        # (note that this consume a new batch of data)
        loss, acc = sess.run([loss_op, accuracy])
        print("Step " + str(step) + ", Minibatch Loss= " +
              "{:.8f}".format(loss) + ", Training Accuracy= " +
              "{:.4f}".format(acc))

print("Optimization Finished!")

saver.save(sess, save_dir)

sess.close()
