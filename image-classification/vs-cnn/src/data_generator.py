import os
import tensorflow as tf
import numpy as np


IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class DataGenerator(object):

  def __init__(self, data_dir, dataset, train_file=None, test_file=None, batch_size=1, num_threads=1, train_shuffle=True, buffer_size=100000):
    self.data_dir = data_dir
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.buffer_size = buffer_size

    if test_file:
      self._build_test_set(test_file, batch_size)
    else:
      raise ValueError('Test set is always required !')

    # create an reinitializable iterator given the dataset structure
    self.iterator = tf.data.Iterator.from_structure(self.test_set.output_types,
                                                    self.test_set.output_shapes)
    self.test_init_opt = self.iterator.make_initializer(self.test_set)
    self.next = self.iterator.get_next()

    if train_file:
      self._build_train_set(train_file, batch_size, train_shuffle)
      self.train_init_opt = self.iterator.make_initializer(self.train_set)

  def load_train_set(self, sess):
    sess.run(self.train_init_opt)

  def load_test_set(self, sess):
    sess.run(self.test_init_opt)

  def get_next(self, sess):
    return sess.run(self.next)

  def _parse_factor(self, img_path):
    tokens = img_path.split('/')[-1].split('_')
    if self.dataset == 'business':
      return '{}_{}'.format(tokens[0], tokens[2])
    elif self.dataset == 'user':
      return '{}_{}'.format(tokens[0], tokens[3])
    else:
      raise ValueError('Invalid dataset: %s.' % self.dataset)

  def _read_txt_file(self, data_file):
    """Read the content of the text file and store it into lists."""
    print('Loading data file: %s' % data_file)
    img_paths = []
    labels = []
    factors = []
    with open(data_file, 'r') as f:
      lines = f.readlines()
      for line in lines:
        items = line.split(' ')
        img_paths.append(os.path.join(self.data_dir, self.dataset, items[0]))
        labels.append(int(items[1]))
        factors.append(self._parse_factor(items[0]))
    return img_paths, labels, factors

  def _build_data_set(self, img_paths, labels, factors, map_fn, shuffle=False):
    img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    factors = tf.convert_to_tensor(factors, dtype=tf.string)
    data = tf.data.Dataset.from_tensor_slices((img_paths, labels, factors))
    if shuffle:
      data = data.shuffle(buffer_size=self.buffer_size)
    data = data.map(map_fn, num_parallel_calls=self.num_threads)
    data = data.batch(self.batch_size)
    data = data.prefetch(self.num_threads)
    return data

  def _build_train_set(self, train_file, batch_size, train_shuffle):
    self.train_img_paths, self.train_labels, self.train_factors = self._read_txt_file(train_file)
    self.train_batches_per_epoch = int(np.ceil(len(self.train_labels) / batch_size))
    self.train_set = self._build_data_set(self.train_img_paths,
                                          self.train_labels,
                                          self.train_factors,
                                          self._parse_function_train,
                                          shuffle=train_shuffle)

  def _build_test_set(self, test_file, batch_size):
    self.test_img_paths, self.test_labels, self.test_factors = self._read_txt_file(test_file)
    self.test_batches_per_epoch = int(np.ceil(len(self.test_labels) / batch_size))
    self.test_set = self._build_data_set(self.test_img_paths,
                                         self.test_labels,
                                         self.test_factors,
                                         self._parse_function_test)

  def _parse_function_train(self, filename, label, factor):
    """Input parser for samples of the training set."""
    # convert label number into one-hot-encoding
    one_hot = tf.one_hot(label, 2)

    # load and preprocess the image
    img_string = tf.read_file(filename)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    """
    Dataaugmentation comes here.
    """
    img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
    img_flipped = tf.image.random_flip_left_right(img_centered)
    img_flipped = tf.image.random_flip_up_down(img_flipped)

    # RGB -> BGR
    img_bgr = img_flipped[:, :, ::-1]

    return img_bgr, one_hot, factor

  def _parse_function_test(self, filename, label, factor):
    """Input parser for samples of the validation/test set."""
    # convert label number into one-hot-encoding
    one_hot = tf.one_hot(label, 2)

    # load and preprocess the image
    img_string = tf.read_file(filename)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

    # RGB -> BGR
    img_bgr = img_centered[:, :, ::-1]

    return img_bgr, one_hot, factor
