import tensorflow as tf
import numpy as np
import os
import csv
from tensorboard import summary as summary_lib
import collections
import braceexpand
from natsort import natsorted
import glob

from config import Config
# from tokenizer import Tokenizer

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_bool("train_and_eval", True, "Whether to run training.")
flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES'] = Config.visible_gpus
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
        d = d.repeat(Config.epoches)
        d = d.shuffle(buffer_size=2000)
        d = d.map(_decode_record)
        d = d.batch(Config.train_batch_size)
        return d

    return input_fn

def dnn_model_fn(features, labels, mode, params):
    feature_columns = params['feature_columns']
    # <_EmbeddingColumn, len() = 8>

    net = tf.feature_column.input_layer(features, feature_columns)
    # <tf.Tensor 'input_layer/concat:0' shape=(?, 50) dtype=float32>

    regularizer = tf.contrib.layers.l2_regularizer(Config.regularizer_scale)
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
    # <tf.Tensor 'Sigmoid:0' shape=(?, 1) dtype=float32>
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {}
        predictions['logits'] = logits
        predictions['probs'] = probs
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    # compute loss
    labels = tf.expand_dims(labels, 1) # <tf.Tensor 'ExpandDims:0' shape=(?, 1) dtype=float32>
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)

    # compute metrics
    positive_labels = tf.greater(labels, Config.label_thres)
    binary_labels = tf.to_int32(positive_labels)
    auc = tf.metrics.auc(labels=binary_labels, predictions=probs, name='auc')
    acc = tf.metrics.accuracy(labels=binary_labels, predictions=probs)
    tf.Print(acc, [acc], first_n=10)
    metrics = {'auc': auc, 'acc':acc}
    tf.summary.scalar('loss', loss)

    # evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        summary_hook = tf.train.SummarySaverHook(
            save_steps=200, output_dir=Config.model_path, 
            summary_op=tf.summary.merge_all())
        eval_hooks = [summary_hook]
        return tf.estimator.EstimatorSpec(mode, loss=loss, evaluation_hooks=eval_hooks,
            eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=Config.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : acc[1]}, 
        every_n_iter=50)
    
    return tf.estimator.EstimatorSpec(mode, loss=loss, 
        train_op=train_op, training_hooks=[logging_hook])


if __name__ == "__main__":
    run_config = tf.estimator.RunConfig(
        save_checkpoints_secs=30)
    train_files_pattern = os.path.join(Config.data_dir, Config.input_files)
    eval_files_pattern = os.path.join(Config.data_dir, Config.eval_files)
    column = tf.feature_column.categorical_column_with_vocabulary_file(
        key='text', vocabulary_file=Config.vocab_file, num_oov_buckets=0, dtype=tf.string)
    word_embedding_column = tf.feature_column.embedding_column(
        column, dimension=Config.embedding_size, combiner="sqrtn")
    
    predictor = tf.estimator.Estimator(
        model_fn=dnn_model_fn,
        params={
            "feature_columns": word_embedding_column
        },
        model_dir=Config.model_path,
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

    # hooks = [tf.train.ProfilerHook(output_dir=Config.model_path, save_secs=15)]
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        hooks=[])
    
    # hook = tf.train.ProfilerHook(output_dir=Config.model_path, save_steps=200)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(predictor, train_spec, eval_spec)
        

