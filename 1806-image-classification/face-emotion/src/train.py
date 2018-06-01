import os
import tensorflow as tf

from tqdm import tqdm
from nn import MLP, Shallow_CNN, Deep_CNN
from data_generator import DataGenerator


# Parameters
# ==================================================
tf.app.flags.DEFINE_string("data_dir", os.path.join("..", "data"),
                           """Path to the data directory""")
tf.app.flags.DEFINE_string("model", "mlp",
                           """"Type of model (mlp or shallow or deep)""")
tf.app.flags.DEFINE_string("checkpoint_dir", os.path.join('..', 'checkpoints'),
                           """Path to checkpoint folder""")
tf.app.flags.DEFINE_string("log_dir", os.path.join('..', 'log'),
                           """Path to log folder""")


tf.app.flags.DEFINE_integer("num_classes", 2,
                            """Number of label classes""")
tf.app.flags.DEFINE_integer("num_checkpoints", 1,
                            """Number of checkpoints to store (default: 1)""")
tf.app.flags.DEFINE_integer("num_epochs", 10,
                            """Number of training epochs""")
tf.app.flags.DEFINE_integer("batch_size", 32,
                            """Batch Size (default: 32)""")
tf.app.flags.DEFINE_integer("display_step", 10,
                            """Display after number of steps""")

tf.app.flags.DEFINE_float("learning_rate", 0.01,
                          """Learning rate""")
tf.app.flags.DEFINE_float("dropout_rate", 0.5,
                          """Probability of dropping neurons""")

tf.app.flags.DEFINE_boolean("allow_soft_placement", True,
                            """Allow device soft device placement""")

FLAGS = tf.app.flags.FLAGS



def init_data_generator():
  # Prepare data
  train_file = os.path.join(FLAGS.data_dir, 'train.csv')
  test_file = os.path.join(FLAGS.data_dir, 'test.csv')
  # Place data loading and preprocessing on the cpu
  with tf.device('/cpu:0'):
    generator = DataGenerator(train_file, test_file, FLAGS.batch_size, num_threads=4)
  return generator


def init_model():
  # Select the model
  if FLAGS.model == 'mlp':
    model = MLP(FLAGS.dropout_rate, FLAGS.num_classes)
  elif FLAGS.model == 'shallow':
    model = Shallow_CNN(FLAGS.dropout_rate, FLAGS.num_classes)
  elif FLAGS.model == 'deep':
    model = Deep_CNN(FLAGS.dropout_rate, FLAGS.num_classes)
  else:
    raise ValueError('--model should be "mlp", "shallow", or "deep"')

  return model


def loss_fn(logits, labels):
  labels = tf.one_hot(indices=labels, depth=2)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
  return loss


def train_fn(loss):
  # Define the optimizer and compute the gradients
  optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum=0.9)
  grads = optimizer.compute_gradients(loss)
  return optimizer.apply_gradients(grads)


def test_fn(logits, labels):
  pred = tf.argmax(logits, 1)
  correct_pred = tf.equal(pred, tf.cast(labels, tf.int64))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy



def train(sess, model, generator, epoch, train_op, loss, accuracy, summary, writer):
  # Initialize training data
  generator.load_train_set(sess)

  sum_loss = 0.0
  sum_acc = 0.0
  num_examples = 0

  loop = tqdm(range(generator.train_batches_per_epoch), 'Training')
  for step in loop:
    # Get next batch of data
    img_batch, label_batch = generator.get_next(sess)

    # And run the training op, get loss and accuracy
    _, batch_loss, batch_acc = sess.run([train_op, loss, accuracy],
                                        feed_dict={model.x: img_batch,
                                                   model.y: label_batch,
                                                   model.is_training: True})
    loop.set_postfix(loss=batch_loss)

    num_examples += len(label_batch)
    sum_loss += batch_loss * len(label_batch)
    sum_acc += batch_acc * len(label_batch)

    # Generate summary with the current batch of data and write to file
    if step % FLAGS.display_step == 0:
      s = sess.run(summary, feed_dict={model.x: img_batch,
                                       model.y: label_batch,
                                       model.is_training: False})
      current_step = epoch * generator.train_batches_per_epoch + step
      writer.add_summary(s, current_step)

  mean_loss = sum_loss / num_examples
  mean_acc = sum_acc / num_examples
  return mean_loss, mean_acc


def test(sess, model, generator, epoch, loss, accuracy, summary, writer):
  # Initialize test data
  generator.load_test_set(sess)

  sum_loss = 0.0
  sum_acc = 0.0
  num_examples = 0

  loop = tqdm(range(generator.test_batches_per_epoch), 'Testing')
  for step in loop:
    # Get next batch of data
    img_batch, label_batch = generator.get_next(sess)

    # And run the training op, get loss and accuracy
    batch_loss, batch_acc, s = sess.run([loss, accuracy, summary],
                                        feed_dict={model.x: img_batch,
                                                   model.y: label_batch,
                                                   model.is_training: False})
    loop.set_postfix(loss=batch_loss)

    num_examples += len(label_batch)
    sum_loss += batch_loss * len(label_batch)
    sum_acc += batch_acc * len(label_batch)

    current_step = epoch * generator.test_batches_per_epoch + step
    writer.add_summary(s, current_step)

  mean_loss = sum_loss / num_examples
  mean_acc = sum_acc / num_examples
  return mean_loss, mean_acc


def main(_):
  # Construct data generator
  generator = init_data_generator()

  # Build Graph
  model = init_model()
  loss = loss_fn(model.logits, model.y)
  train_opt = train_fn(loss)
  accuracy = test_fn(model.logits, model.y)

  # Summary for monitoring
  tf.summary.scalar('loss', loss)
  tf.summary.scalar('accuracy', accuracy)
  summary = tf.summary.merge_all()

  # Create writers for logging and visualization
  log_dir = os.path.join(FLAGS.log_dir, FLAGS.model)
  if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
  tf.gfile.MakeDirs(log_dir)

  train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'))
  test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))


  # Initialize an saver for store model checkpoints
  saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)
  checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.model)
  if tf.gfile.Exists(checkpoint_dir):
    tf.gfile.DeleteRecursively(checkpoint_dir)
  tf.gfile.MakeDirs(checkpoint_dir)

  # Create a session
  session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=session_conf) as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    train_writer.add_graph(sess.graph)

    best_acc = float('-inf')

    print("\nStart training...")
    print("Open Tensorboard at --logdir {}".format(FLAGS.log_dir))

    # Loop over number of epochs
    for epoch in range(FLAGS.num_epochs):
      print("\nEpoch number: {}".format(epoch + 1))

      train_loss, train_acc = train(sess, model, generator, epoch, train_opt, loss, accuracy, summary, train_writer)
      print('train_loss = {:.4f}, train_acc = {:.2f} %'.format(train_loss, train_acc * 100))

      test_loss, test_acc = test(sess, model, generator, epoch, loss, accuracy, summary, test_writer)
      print('test_loss = {:.4f}, test_acc = {:.2f} %'.format(test_loss, test_acc * 100))

      if best_acc < test_acc:
        best_acc = test_acc
        checkpoint_prefix = os.path.join(checkpoint_dir, 'epoch_{}'.format(epoch + 1))
        path = saver.save(sess, checkpoint_prefix)
        print("Saved model checkpoint to {}".format(path))

    # Finish training, print the best accuracy on test set
    print('\nBest accuracy = {:.2f} %'.format(best_acc * 100))


if __name__ == '__main__':
  tf.app.run()