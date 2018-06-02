import os
import tensorflow as tf
import numpy as np

from layers import conv, lrn, max_pool, fc


class FVS_CNN(object):

  def __init__(self, num_classes, num_factor_units, skip_layers=None, finetune_layers=None, weights_path='weights/bvlc_alexnet.npy'):
    # TF placeholder for graph input and output
    self.x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    self.y = tf.placeholder(tf.float32, [None, 2])
    self.factor_weights = tf.placeholder(tf.float32, shape=[4096, num_factor_units])
    self.factor_biases = tf.placeholder(tf.float32, shape=[num_factor_units])

    self.num_classes = num_classes
    self.num_factor_units = num_factor_units
    self.skip_layers = skip_layers
    self.finetune_layers = finetune_layers
    self.weights_path = weights_path

    self.factor_weight_dict = {}
    self.factor_bias_dict = {}

    # Call the create function to build the computational graph of AlexNet
    self.build()

  def build(self):
    """Create the network graph."""
    # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
    conv1 = conv(self.x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
    norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
    pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
    conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
    norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
    pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
    fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')

    fc7_factor = fc(fc6, 4096, self.num_factor_units, 'fc7_factor')
    fc7_shared = fc(fc6, 4096, 4096 - self.num_factor_units, 'fc7_shared')

    with tf.variable_scope('fc7_factor', reuse=True):
      self.assign_factor = tf.group(tf.get_variable('weights').assign(self.factor_weights),
                                    tf.get_variable('biases').assign(self.factor_biases))

    # with tf.control_dependencies([assign_weights, assign_biases]):
    fc7_concat = tf.concat([fc7_factor, fc7_shared], axis=1, name='fc7_concat')

    # 8th Layer: FC and return unscaled activations
    self.fc8 = fc(fc7_concat, 4096, self.num_classes, relu=False, name='fc8')

  def load_factor_weights(self, session, factor, weight_dir=None):
    factor = factor[0].decode('UTF-8')
    if factor in self.factor_weight_dict.keys():
      weights = self.factor_weight_dict[factor]
      biases = self.factor_bias_dict[factor]
    else:
      if weight_dir:
        weights = np.load(os.path.join(weight_dir, factor, 'weights.npy'))
        biases = np.load(os.path.join(weight_dir, factor, 'biases.npy'))
      else:
        for data in self.weights_dict['fc7']:
          if len(data.shape) == 1:  # biases
            biases = data[:self.num_factor_units]
          else:  # weights
            weights = data[:, :self.num_factor_units]
    session.run(self.assign_factor, feed_dict={self.factor_weights: weights,
                                               self.factor_biases: biases})

  def update_factor_weights(self, session, factor):
    factor = factor[0].decode('UTF-8')
    with tf.variable_scope('fc7_factor', reuse=True):
      self.factor_weight_dict[factor], self.factor_bias_dict[factor] = \
        session.run([tf.get_variable('weights'), tf.get_variable('biases')])

  def load_initial_weights(self, session):
    self.weights_dict = dict(np.load(self.weights_path, encoding='bytes').item())

    # Loop over all layer names stored in the weights dict
    for op_name in self.weights_dict.keys():
      # Check if layer should be trained from scratch
      if op_name not in self.skip_layers:
        trainable = False
        if op_name in self.finetune_layers:
          trainable = True
        with tf.variable_scope(op_name, reuse=True):
          # Assign weights/biases to their corresponding tf variable
          for data in self.weights_dict[op_name]:
            # Biases
            if len(data.shape) == 1:
              var = tf.get_variable('biases', trainable=trainable)
              session.run(var.assign(data))
            # Weights
            else:
              var = tf.get_variable('weights', trainable=trainable)
              session.run(var.assign(data))

    # Load weights for fc7_shared
    with tf.variable_scope('fc7_shared', reuse=True):
      # Assign weights/biases to their corresponding tf variable
      for data in self.weights_dict['fc7']:
        if len(data.shape) == 1:  # Biases
          var = tf.get_variable('biases', trainable=True)
          session.run(var.assign(data[self.num_factor_units:]))
        else:  # Weights
          var = tf.get_variable('weights', trainable=True)
          session.run(var.assign(data[:, self.num_factor_units:]))
