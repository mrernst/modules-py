import sys
sys.path.append("../src/")
import modules as mod
import get_mnist as gm
import tensorflow as tf
import numpy as np


class Lenet5(mod.ComposedModule):
    def define_inner_modules(self, name, activations, filter_shapes, strides, bias_shapes, ksizes, pool_strides):
        self.layers = {}
        # first convolutional layer
        self.layers["conv0"] = mod.ConvolutionalLayerModule("conv0",
            activations[0], filter_shapes[0], strides[0], bias_shapes[0])
        # first max-pooling layer
        self.layers["pool0"] = mod.MaxPoolingModule("pool0", ksizes[0], pool_strides[0])
        # second convolutional layer
        self.layers["conv1"] = mod.ConvolutionalLayerModule(name + "conv1",
            activations[1], filter_shapes[1], strides[1], bias_shapes[1])
        # second max-pooling layer
        self.layers["pool1"] = mod.MaxPoolingModule("pool1", ksizes[0], pool_strides[0])
        self.layers["flat_pool1"] = mod.FlattenModule("flat_pool1")
        # first fully-connected layer
        self.layers["fc0"] = mod.FullyConnectedLayerModule("fc0", activations[2], int(np.prod(np.array(bias_shapes[1]) / np.array(pool_strides[1]))), np.prod(bias_shapes[2]))
        # second fully-connected layer
        self.layers["fc1"] = mod.FullyConnectedLayerModule("fc1", activations[3], np.prod(bias_shapes[2]), np.prod(bias_shapes[3]))
        # connections
        self.layers["pool0"].add_input(self.layers["conv0"])
        self.layers["conv1"].add_input(self.layers["pool0"])
        self.layers["pool1"].add_input(self.layers["conv1"])
        self.layers["flat_pool1"].add_input(self.layers["pool1"])
        self.layers["fc0"].add_input(self.layers["flat_pool1"])
        self.layers["fc1"].add_input(self.layers["fc0"])
        # set input and output
        self.input_module = self.layers["conv0"]
        self.output_module = self.layers["fc1"]


def cross_entropy(a, b, name):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a, labels=b, name=name))

def to_one_hot(ns):
    bs = ns.shape[0]
    ret = np.zeros(shape=(bs, 10), dtype=np.float)
    ret[range(bs), ns] = 1.0
    return ret

BATCH_SIZE = 50

train_images_name = "train-images-idx3-ubyte.gz"  #  training set images (9912422 bytes)
train_data_filename = gm.maybe_download(train_images_name)
train_mnist = gm.extract_data(train_data_filename, 60000)

train_label_name = "train-labels-idx1-ubyte.gz"  #  training set labels (28881 bytes)
train_label_filename = gm.maybe_download(train_label_name)
train_mnist_label = gm.extract_labels(train_label_filename, 60000)

test_image_name = "t10k-images-idx3-ubyte.gz"  #  test set images (1648877 bytes)
test_data_filename = gm.maybe_download(test_image_name)
test_mnist = gm.extract_data(test_data_filename, 5000)

test_label_name = "t10k-labels-idx1-ubyte.gz"  #  test set labels (4542 bytes)
test_label_filename = gm.maybe_download(test_label_name)
test_mnist_label = gm.extract_labels(test_label_filename, 5000)


inp = mod.ConstantPlaceholderModule("input", shape=(BATCH_SIZE, 28, 28, 1))
labels = mod.ConstantPlaceholderModule("input", shape=(BATCH_SIZE, 10))

activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.identity]
filter_shapes = [[8,8,1,6],[8,8,6,16]]
strides = [[1,1,1,1], [1,1,1,1]]
bias_shapes = [[1,28,28,6],[1,14,14,16], [1,120],[1,10]]
ksizes = [[1,4,4,1],[1,4,4,1]]
pool_strides = [[1,2,2,1], [1,2,2,1]]
network = Lenet5("lenet5", activations, filter_shapes, strides, bias_shapes, ksizes, pool_strides)

error = mod.ErrorModule("cross_entropy", cross_entropy)
accuracy = mod.BatchAccuracyModule("accuracy")
optimizer = mod.OptimizerModule("adam", tf.train.AdamOptimizer())

network.add_input(inp)
error.add_input(network)
error.add_input(labels)
accuracy.add_input(network)
accuracy.add_input(labels)
optimizer.add_input(error)
optimizer.create_output(0)
accuracy.create_output(0)



def train_batch(sess, i):
    batch = train_mnist[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
    batch_labels = train_mnist_label[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
    feed_dict = {}
    feed_dict[inp.placeholder] = batch
    feed_dict[labels.placeholder] = to_one_hot(batch_labels)
    err = sess.run(optimizer.outputs[0], feed_dict=feed_dict)
    print("error:\t\t{:.4f}".format(err), end='\r')


def test_epoch(sess):
    print("")
    acc = 0
    for j in range(5000//BATCH_SIZE - 1):
        batch = test_mnist[j * BATCH_SIZE: (j + 1) * BATCH_SIZE]
        batch_labels = test_mnist_label[j * BATCH_SIZE: (j + 1) * BATCH_SIZE]
        feed_dict = {}
        feed_dict[inp.placeholder] = batch
        feed_dict[labels.placeholder] = to_one_hot(batch_labels)
        acc += sess.run(accuracy.outputs[0], feed_dict=feed_dict)
        print("accuracy:\t{:.2f} %".format(100 * acc / (j+1)), end='\r')
    print("")


N_EPOCH = 2
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_number in range(N_EPOCH):
        for i in range(60000//BATCH_SIZE - 1):
            if i % 100 == 0:
                test_epoch(sess)
            train_batch(sess, i)
