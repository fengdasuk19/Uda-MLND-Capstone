word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model.")
flags.DEFINE_string(
    "train_data", None,
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")
flags.DEFINE_string(
    "eval_data", None, "Analogy questions. "
    "See README.md for how to get 'questions-words.txt'.")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500,
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")

FLAGS = flags.FLAGS


#def main(_):
#"""Train a word2vec model."""
if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
  print("--train_data --eval_data and --save_path must be specified.")
  sys.exit(1)
opts = Options()
session = tf.InteractiveSession()
#with tf.Graph().as_default(), tf.Session() as session:
with tf.Graph().as_default():
  with tf.device("/cpu:0"):
    model = Word2Vec(opts, session)
    model.read_analogies() # Read analogy questions
  for _ in xrange(opts.epochs_to_train):
    model.train()  # Process one epoch
    model.eval()  # Eval analogies.
  # Perform a final save.
  model.saver.save(session, os.path.join(opts.save_path, "model.ckpt"),
                   global_step=model.global_step)
  if FLAGS.interactive:
    # E.g.,
    # [0]: model.analogy(b'france', b'paris', b'russia')
    # [1]: model.nearby([b'proton', b'elephant', b'maxwell'])
    _start_shell(locals())


#if __name__ == "__main__":
#  tf.app.run()
