import tensorflow as tf

session = tf.InteractiveSession()
"""Train a word2vec model."""
if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
  print("--train_data --eval_data and --save_path must be specified.")
  sys.exit(1)
opts = Options()
#with tf.Graph().as_default() as session:
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

