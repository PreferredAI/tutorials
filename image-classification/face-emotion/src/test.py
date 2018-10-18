import os
import tensorflow as tf

from nn import MLP, Shallow_CNN, Deep_CNN
import cv2
import numpy as np

# Parameters
# ==================================================
tf.app.flags.DEFINE_string("data_dir", "test_images",
                           """Path to the data directory""")
tf.app.flags.DEFINE_string("model", "mlp",
                           """"Type of model (mlp or shallow or deep)""")
tf.app.flags.DEFINE_string("checkpoint_dir", 'checkpoints',
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
  img = cv2.resize(img, (48, 48))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = np.asarray(img) / 255.0 - 0.5
  return img.reshape(1, 48, 48, 1)


def init_model():
  # Select the model
  if FLAGS.model == 'mlp':
    model = MLP()
  elif FLAGS.model == 'shallow':
    model = Shallow_CNN()
  elif FLAGS.model == 'deep':
    model = Deep_CNN()
  else:
    raise ValueError('--model should be "mlp", "shallow", or "deep"')

  return model


def to_label(class_idx):
  labels = {0: 'sad', 1: 'happy'}
  return labels[class_idx]


def main(_):
  # Build Graph
  model = init_model()
  prediction = tf.argmax(model.logits, axis=1)

  # Initialize an saver for store model checkpoints
  saver = tf.train.Saver()
  checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.model)
  print('\nLoading model from {}\n'.format(checkpoint_dir))

  # Create a session
  session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=session_conf) as sess:
    # Restore trained model
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    print("Model loaded!")

    print('\n{:20} {:20}'.format('Image', 'Predicted As'))
    print('-' * 40)
    for img_name in os.listdir(FLAGS.data_dir):
      img = load_image(os.path.join(FLAGS.data_dir, img_name))
      label = to_label(sess.run(prediction, feed_dict={model.x: img,
                                                    model.is_training: False})[0])
      print('{:20} {:20}'.format(img_name, label))


if __name__ == '__main__':
  tf.app.run()
