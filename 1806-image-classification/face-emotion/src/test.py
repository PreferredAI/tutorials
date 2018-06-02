import os
import tensorflow as tf

from nn import MLP, Shallow_CNN, Deep_CNN
from PIL import Image
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
  width, height = img.size
  new_width = new_height = min(width, height)
  left = (width - new_width) / 2
  top = (height - new_height) / 2
  right = (width + new_width) / 2
  bottom = (height + new_height) / 2
  return img.crop((left, top, right, bottom))


def load_image(img_file):
  img = Image.open(img_file)
  img = center_crop(img)
  img = img.resize((48, 48), Image.ANTIALIAS)
  img = img.convert('L')  # convert image to grayscale
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
    raise ValueError('--model should be "shallow" or "deep"')

  return model


def to_label(class_idx):
  labels = {0: 'sad', 1: 'happy'}
  return labels[class_idx]


def main(_):
  # Build Graph
  model = init_model()
  softmax = tf.argmax(model.logits, axis=1)

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
      label = to_label(sess.run(softmax, feed_dict={model.x: img,
                                                    model.is_training: False})[0])
      print('{:20} {:20}'.format(img_name, label))


if __name__ == '__main__':
  tf.app.run()
