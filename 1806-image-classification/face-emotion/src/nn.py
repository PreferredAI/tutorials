import tensorflow as tf


class NN:
  """
    Generic class of Neural Network for other models to inherit from.
    
    Attributes:
        x           : Data input placeholder.
        y           : Ground-truth label placeholder.
        is_training : Keep track of training and testing phrases.
  """
  def __init__(self):
    self.x = tf.placeholder(tf.float32, shape=[None, 48, 48, 1])
    self.y = tf.placeholder(tf.int32, shape=[None])
    self.is_training = tf.placeholder(tf.bool)



class MLP(NN):
  """
    Definition of MLP network with 2 fully-connected layers of 512 dimensions.
  """
  def __init__(self, dropout_rate, num_classes):
    NN.__init__(self)

    # Flatten image into feature vector
    features = tf.layers.flatten(self.x)

    # Fully-Connected Layer #1
    fc1 = tf.layers.dense(inputs=features, units=512, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=fc1, rate=dropout_rate, training=self.is_training)

    # Fully-Connected Layer #2
    fc2 = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu)

    # Output Layer
    self.logits = tf.layers.dense(inputs=fc2, units=num_classes)



class Shallow_CNN(NN):
  """
    Definition of shallow CNN with 1 convolutional layer, 1 pooling layer, and 1 fully-connected layer.
  """
  def __init__(self, dropout_rate, num_classes):
    NN.__init__(self)

    # Convolutional Layer
    conv = tf.layers.conv2d(inputs=self.x, filters=32, kernel_size=[5, 5], strides=1, padding="same", activation=tf.nn.relu)

    # Pooling Layer
    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)

    # Flatten out the convolutional layer
    flatten = tf.layers.flatten(pool)
    dropout = tf.layers.dropout(flatten, rate=dropout_rate, training=self.is_training)

    # Fully-Connected Layer
    fc = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu)

    # Output Layer
    self.logits = tf.layers.dense(inputs=fc, units=num_classes)



class Deep_CNN(NN):
  """
    Definition of deep CNN with 4 convolutional layers, 3 pooling layer, and 1 fully-connected layer.
  """
  def __init__(self, dropout_rate, num_classes):
    NN.__init__(self)

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=self.x, filters=32, kernel_size=[5, 5], strides=1, padding="same", activation=tf.nn.relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    # Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[3, 3], strides=1, padding="same", activation=tf.nn.relu)
    # Pooling Layer #3
    pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # Flatten out the convolutional layer
    flatten = tf.layers.flatten(pool3)
    dropout = tf.layers.dropout(inputs=flatten, rate=dropout_rate, training=self.is_training)

    # Fully-Connected Layer
    fc = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu)

    # Output Layer
    self.logits = tf.layers.dense(inputs=fc, units=num_classes)
