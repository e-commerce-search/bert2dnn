import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import csv
from tensorboard import summary as summary_lib
import collections
from tokenizer import Tokenizer
from config import Config
import braceexpand
from natsort import natsorted
import glob


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "/home/search/Data/yue.shang/data/amazon_movie_reviews/shuf/", 
    "the directory of input files")
flags.DEFINE_string("input_files", "part-*[1-9].tfrecord", 
    "The glob pattern fopr input files")
flags.DEFINE_string("eval_files", "part-*0.tfrecord", 
    "The glob pattern for input files")

flags.DEFINE_string("model_path", "./models/dnn_test", "model output path")
flags.DEFINE_bool("train_and_eval", True, "Whether to run training.")
flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("num_examples", 650000, "Total num of train examples")
flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")
flags.DEFINE_bool("gen_data_only", True,
                  "Whether to generate train/eval/test data only.")

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
tf.logging.set_verbosity(tf.logging.INFO)
#########################



def file_based_input_fn_builder(input_files_pattern, is_training):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
        "text": tf.VarLenFeature(dtype=tf.string),
        "label": tf.FixedLenFeature([], dtype=tf.float32)
    }

    def _decode_record(record):
        example = tf.parse_single_example(record, name_to_features)
        features = {}
        features['text'] = example['text']
        label = example['label']
        return features, label

    def input_fn():
        """The actual input function."""
        file_patterns = braceexpand.braceexpand(input_files_pattern)
        filenames = natsorted([f for fp in file_patterns for f in glob.glob(fp)])
        print("input files:")
        print(filenames)
        d = tf.data.TFRecordDataset(filenames)
        d = d.map(_decode_record)
        d = d.batch(FLAGS.train_batch_size)
        return d

    return input_fn

def dnn_model_fn(features, labels, mode, params):
    feature_columns = params['feature_columns']
    # print(feature_columns)
    net = tf.feature_column.input_layer(features, feature_columns)
    regularizer = tf.contrib.layers.l2_regularizer(0.05)
    for units in Config.hidden_units:
        net = tf.layers.dense(
            net, units=units, activation=tf.nn.relu, 
            kernel_regularizer=regularizer
        )
        if Config.dropout > 0.0:
            net = tf.layers.dropout(net, Config.dropout, 
                training=(mode == tf.estimator.ModeKeys.TRAIN))

    logits = tf.layers.dense(net, 1, activation=None,
            kernel_regularizer=regularizer)
    
    probs = tf.nn.sigmoid(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {}
        predictions['logits'] = logits
        predictions['probs'] = probs
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    # compute loss
    logits = logits[:, 0]
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)

    # compute metrics
    binary_labels = tf.to_int32(labels)
    auc = tf.metrics.auc(labels=binary_labels, predictions=probs, name='auc')
    acc = tf.metrics.accuracy(labels=binary_labels, predictions=probs)
    metrics = {'auc': auc, 'acc':acc}

    tf.summary.scalar('loss', loss)

    # evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.05)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : acc[1]}, 
        every_n_iter=50)
    
    return tf.estimator.EstimatorSpec(mode, loss=loss, 
        train_op=train_op, training_hooks=[logging_hook])


if __name__ == "__main__":
    run_config = tf.estimator.RunConfig(
        save_checkpoints_secs=30)
    train_files_pattern = os.path.join(FLAGS.data_dir, FLAGS.input_files)
    eval_files_pattern = os.path.join(FLAGS.data_dir, FLAGS.eval_files)
    column = tf.feature_column.categorical_column_with_vocabulary_file(
        key='text', vocabulary_file=Config.vocab_file, num_oov_buckets=0, dtype=tf.string)
    word_embedding_column = tf.feature_column.embedding_column(
        column, dimension=Config.embedding_size)
    
    predictor = tf.estimator.Estimator(
        model_fn=dnn_model_fn,
        params={
            "feature_columns": word_embedding_column
        },
        model_dir=FLAGS.model_path,
        config=run_config)
    
    # if FLAGS.train_and_eval:
    print("***** Running training *****")
    tf.logging.info("***** Running training *****")
    
    train_input_fn = file_based_input_fn_builder(
        input_files_pattern=train_files_pattern,
        is_training=True)
    
    eval_input_fn = file_based_input_fn_builder(
        input_files_pattern=eval_files_pattern,
        is_training=False)

    hooks = [tf.train.ProfilerHook(output_dir=FLAGS.model_path, save_secs=15)]
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        hooks=[])
    
    hook = tf.train.ProfilerHook(output_dir=FLAGS.model_path, save_secs=20)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(predictor, train_spec, eval_spec)
        

