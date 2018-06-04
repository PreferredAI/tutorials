import os
import tensorflow as tf

from nn import Deep_CNN
import cv2
import numpy as np

# Parameters
# ==================================================
tf.app.flags.DEFINE_string("output_dir", "visualization",
                           """Path to the data directory""")
tf.app.flags.DEFINE_string("checkpoint_dir", 'checkpoints/deep',
                           """Path to checkpoint folder""")

tf.app.flags.DEFINE_boolean("allow_soft_placement", True,
                            """Allow device soft device placement""")

FLAGS = tf.app.flags.FLAGS


def center_crop(img):
  height, width, channels  = img.shape
  new_width = new_height = min(width, height)
  left = int((width - new_width) / 2)
  top = int((height - new_height) / 2)
  right = int((width + new_width) / 2)
  bottom = int((height + new_height) / 2)
  return img[top:bottom, left:right]


def load_image(img_file):
  img = cv2.imread(img_file)
  img = center_crop(img)

  new_img = cv2.resize(img, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
  cv2.imwrite(os.path.join(FLAGS.output_dir, img_file.split('/')[-1]), new_img)

  img = cv2.resize(img, (48, 48))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = np.asarray(img) / 255.0 - 0.5
  return img.reshape(1, 48, 48, 1)


def vis_conv1_filters(sess):
  values = sess.run('conv2d/kernel:0')
  values = np.transpose(values, [3, 0, 1, 2])
  values = values.reshape(32, 5, 5)

  values = values / np.max(values) * 255.0

  A = np.ones([25, 49]) * 50

  for i in range(values.shape[0]):
    m = 1 + int(np.floor(i / 8)) * 6
    n = 1 + (i % 8) * 6
    A[m: m + 5, n: n + 5] = values[i, :]

  A = cv2.resize(A, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_AREA)
  cv2.imwrite(os.path.join(FLAGS.output_dir, "conv1_filters.jpg"), A)


def vis_conv4(sess, model, img_path):
  img = load_image(img_path)
  values = sess.run(model.conv4, feed_dict={model.x: img,
                                            model.is_training: False})[0]
  values = values / np.max(values) * 255.0
  values = np.transpose(values, [2, 0, 1])

  A = np.ones([114, 226]) * 50

  for i in range(values.shape[0]):
    m = 2 + int(np.floor(i / 16)) * 14
    n = 2 + (i % 16) * 14
    A[m: m + 12, n: n + 12] = values[i, :]


  A = cv2.resize(A, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_AREA)
  cv2.imwrite(os.path.join(FLAGS.output_dir, "conv4_{}".format(img_path.split('/')[-1])), A)


def main(_):
  # Build Graph
  model = Deep_CNN()

  # Initialize an saver for store model checkpoints
  saver = tf.train.Saver()
  print('\nLoading model from {}\n'.format(FLAGS.checkpoint_dir))

  # Create a session
  session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=session_conf) as sess:
    # Restore trained model
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    print("Model loaded!")

    vis_conv1_filters(sess)

    vis_conv4(sess, model, 'data/images/0/130.jpg')
    vis_conv4(sess, model, 'data/images/0/607.jpg')

    vis_conv4(sess, model, 'data/images/1/82.jpg')
    vis_conv4(sess, model, 'data/images/1/791.jpg')



if __name__ == '__main__':
  tf.app.run()
