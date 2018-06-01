import tensorflow as tf
import pandas as pd
import numpy as np

class DataGenerator(object):
  def __init__(self, train_file, test_file, batch_size, num_threads, buffer_size=10000, train_shuffle=True):
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.buffer_size = buffer_size
    self.train_shuffle = train_shuffle

    # read datasets from csv files
    self.train_img_paths, self.train_labels = self._read_csv_file(train_file)
    self.test_img_paths, self.test_labels = self._read_csv_file(test_file)

    # number of batches per epoch
    self.train_batches_per_epoch = int(np.ceil(len(self.train_labels) / batch_size))
    self.test_batches_per_epoch = int(np.ceil(len(self.test_labels) / batch_size))

    # build datasets
    self._build_train_set()
    self._build_test_set()

    # create an reinitializable iterator given the dataset structure
    self.iterator = tf.data.Iterator.from_structure(self.train_set.output_types,
                                                    self.train_set.output_shapes)
    self.train_init_opt = self.iterator.make_initializer(self.train_set)
    self.test_init_opt = self.iterator.make_initializer(self.test_set)
    self.next = self.iterator.get_next()

  def load_train_set(self, session):
    session.run(self.train_init_opt)

  def load_test_set(self, session):
    session.run(self.test_init_opt)

  def get_next(self, session):
    return session.run(self.next)

  def _read_csv_file(self, data_file):
    """Read the content of the text file and store it into lists."""
    df = pd.read_csv(data_file, header=None)
    img_paths = df[0].values
    labels = df[1].values
    return img_paths, labels

  def _build_data_set(self, img_paths, labels, map_fn, shuffle=False):
    img_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    data = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    if shuffle:
      data = data.shuffle(buffer_size=self.buffer_size)
    data = data.map(map_fn, num_parallel_calls=self.num_threads)
    data = data.batch(self.batch_size)
    data = data.prefetch(self.num_threads)
    return data

  def _build_train_set(self):
    self.train_set = self._build_data_set(self.train_img_paths,
                                          self.train_labels,
                                          self._parse_function_train,
                                          self.train_shuffle)

  def _build_test_set(self):
    self.test_set = self._build_data_set(self.test_img_paths,
                                         self.test_labels,
                                         self._parse_function_test)

  def _parse_function_train(self, filename, label):
    # load and preprocess the image
    img_string = tf.read_file(filename)
    img_decoded = tf.cast(tf.image.decode_jpeg(img_string), dtype=tf.float32)
    """
    Data augmentation comes here.
    """
    img_flipped = tf.image.random_flip_left_right(img_decoded)
    img_scaled = tf.divide(img_flipped, tf.constant(255.0, dtype=tf.float32))
    img_centered = tf.subtract(img_scaled, tf.constant(0.5, dtype=tf.float32))
    return img_centered, label

  def _parse_function_test(self, filename, label):
    # load and preprocess the image
    img_string = tf.read_file(filename)
    img_decoded = tf.cast(tf.image.decode_jpeg(img_string), dtype=tf.float32)
    img_scaled = tf.divide(img_decoded, tf.constant(255.0, dtype=tf.float32))
    img_centered = tf.subtract(img_scaled, tf.constant(0.5, dtype=tf.float32))
    return img_centered, label
