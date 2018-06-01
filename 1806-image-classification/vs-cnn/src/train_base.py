import os
import tensorflow as tf

from model_base import VS_CNN
from data_generator import DataGenerator
from datetime import datetime
from tqdm import tqdm
from tensorboard_logging import Logger

# Parameters
# ==================================================
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("data_dir", "../data",
                           """Path to data folder""")
tf.app.flags.DEFINE_string("dataset", "business",
                           """Name of dataset (business or user)""")

tf.app.flags.DEFINE_integer("num_checkpoints", 1,
                            """Number of checkpoints to store (default: 1)""")
tf.app.flags.DEFINE_integer("num_epochs", 20,
                            """Number of training epochs (default: 20)""")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            """Batch Size (default: 64)""")
tf.app.flags.DEFINE_integer("num_threads", 48,
                            """Display after number of steps (default: 48)""")
tf.app.flags.DEFINE_integer("display_step", 10,
                            """Display after number of steps (default: 10)""")

tf.app.flags.DEFINE_float("learning_rate", 0.001,
                          """Learning rate (default: 0.001)""")
tf.app.flags.DEFINE_float("lambda_reg", 0.0005,
                          """Regularization lambda factor (default: 0.0005)""")
tf.app.flags.DEFINE_float("max_grad_norm", 5.0,
                          """Maximum value for gradient clipping (default: 5.0)""")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          """Probability of keeping neurons (default: 0.5)""")

tf.app.flags.DEFINE_boolean("allow_soft_placement", True,
                            """Allow device soft device placement""")

# Network params
num_classes = 2
skip_layers = ['fc8'] # no pre-trained weights
train_layers = ['fc8']
finetune_layers = ['fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']


# Path to the textfiles for the trainings and validation set
train_file = os.path.join(FLAGS.data_dir, FLAGS.dataset, 'train.txt')
val_file = os.path.join(FLAGS.data_dir, FLAGS.dataset, 'val_Boston.txt')

writer_dir = "../tensorboard/{}".format(FLAGS.dataset)
checkpoint_dir = "../checkpoints/{}".format(FLAGS.dataset)

if tf.gfile.Exists(writer_dir):
  tf.gfile.DeleteRecursively(writer_dir)
tf.gfile.MakeDirs(writer_dir)

if tf.gfile.Exists(checkpoint_dir):
  tf.gfile.DeleteRecursively(checkpoint_dir)
tf.gfile.MakeDirs(checkpoint_dir)



def loss_fn(model):
  # Op for calculating the loss
  with tf.name_scope("loss"):
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=model.y, logits=model.fc8)
    l2_regularization = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name])
    loss = tf.reduce_mean(cross_entropy + FLAGS.lambda_reg * l2_regularization)
  return loss


def train_fn(loss, generator):
  # List of trainable variables of the layers we want to train
  var_list1 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in finetune_layers]
  var_list2 = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

  # Train op
  with tf.name_scope("train"):
    grads = tf.gradients(loss, var_list1 + var_list2)
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):]

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
      learning_rate=0.0001,
      global_step=global_step,
      decay_steps=generator.train_batches_per_epoch,
      decay_rate=0.8,
      staircase=True,
      name='exponential_decay_learning_rate')
    opt1 = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    opt2 = tf.train.MomentumOptimizer(10 * learning_rate, momentum=0.9)

    train_op1 = opt1.apply_gradients(zip(grads1, var_list1), global_step)
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
    train_op = tf.group(train_op1, train_op2)
  return train_op, learning_rate


def eval_fn(model):
  # Evaluation op: Accuracy of the model
  with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(model.fc8, 1), tf.argmax(model.y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy


def train(sess, model, generator, train_op, learning_rate, loss, accuracy, epoch, logger):
  # Initialize iterator with the training dataset
  generator.load_train_set(sess)

  loop = tqdm(range(generator.train_batches_per_epoch), 'Training')
  for step in loop:
    # get next data
    batch_img, batch_label, batch_factor = generator.get_next(sess)
    # And run the training op
    _, _loss, _acc = sess.run([train_op, loss, accuracy],
                              feed_dict={model.x: batch_img,
                                         model.y: batch_label})
    loop.set_postfix(loss=_loss)

    # Generate summary with the current batch of data and write to file
    if (step > 0) and (step % FLAGS.display_step == 0 or step == generator.train_batches_per_epoch - 1):
      # Log info into tensorboard
      _step = epoch * generator.train_batches_per_epoch + step
      logger.log_scalar('cross_entropy', _loss, _step)
      logger.log_scalar('accuracy', _acc, _step)
      logger.log_scalar('learning_rate', sess.run(learning_rate), _step)


def test(sess, model, generator, accuracy):
  # Validate the model on the entire validation set
  print("{} Start testing".format(datetime.now()))
  generator.load_test_set(sess)
  test_acc = 0.
  test_count = 0
  for _ in range(generator.test_batches_per_epoch):
    batch_img, batch_label, batch_factor = generator.get_next(sess)
    acc = sess.run(accuracy, feed_dict={model.x: batch_img,
                                        model.y: batch_label})
    test_acc += acc * len(batch_label)
    test_count += len(batch_label)
  test_acc /= test_count
  print("{} Test Accuracy = {:.4f}".format(datetime.now(), test_acc))


def save_model(sess, saver, epoch):
  print("{} Saving checkpoint of model...".format(datetime.now()))
  checkpoint_name = os.path.join(checkpoint_dir,
                                 'model_epoch' + str(epoch + 1) + '.ckpt')
  save_path = saver.save(sess, checkpoint_name)
  print("{} Model checkpoint saved at {}".format(datetime.now(), save_path))


def main(_):
  # Place data loading and preprocessing on the cpu
  with tf.device('/cpu:0'):
    generator = DataGenerator(FLAGS.data_dir, FLAGS.dataset, train_file, val_file, FLAGS.batch_size, FLAGS.num_threads, train_shuffle=True)

  # Initialize model
  model = VS_CNN(num_classes, skip_layers, finetune_layers)
  loss = loss_fn(model)
  train_op, learning_rate = train_fn(loss, generator)
  accuracy = eval_fn(model)

  # Initialize an saver for store model checkpoints
  saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

  # Start Tensorflow session
  config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=config) as sess:
    # Initialize the FileWriter
    logger = Logger(writer_dir, sess.graph)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(), writer_dir))

    # Loop over number of epochs
    for epoch in range(FLAGS.num_epochs):
      print("\n{} Epoch number: {}".format(datetime.now(), epoch + 1))

      train(sess, model, generator, train_op, learning_rate, loss, accuracy, epoch, logger)

      test(sess, model, generator, accuracy)

      save_model(sess, saver, epoch)


if __name__ == '__main__':
  tf.app.run()
