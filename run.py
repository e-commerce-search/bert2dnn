import tensorflow as tf
import numpy as np
import os
import csv
from tensorboard import summary as summary_lib
import collections
import braceexpand
from natsort import natsorted
import glob
import random
from config import Config

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_bool("do_train", False, "Whether to run training and eval")
flags.DEFINE_bool("do_eval", True, "Whether to run eval.")
flags.DEFINE_bool("do_predict", False,
    "Whether to run the model in inference mode on the test set.")

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
        if not is_training:
            filenames = sorted(filenames)
            tf.logging.info("eval/test files: %s" % ','.join(filenames))
        else:
            random.shuffle(filenames)
            tf.logging.info("train files: %s" % ','.join(filenames))

        d = tf.data.TFRecordDataset(filenames)
        if is_training:
            d = d.repeat(Config.epoches)
            d = d.shuffle(buffer_size=2000)
        d = d.map(_decode_record)
        d = d.batch(Config.batch_size)
        return d
    return input_fn

def dnn_model_fn(features, labels, mode, params):
    feature_columns = params['feature_columns']
    net = tf.feature_column.input_layer(features, feature_columns)
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
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {}
        predictions['logits'] = logits
        predictions['probabilities'] = probs
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    # compute loss
    labels = tf.expand_dims(labels, 1) 
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)

    # compute metrics
    binary_labels = tf.to_int32(tf.greater(labels, Config.label_thres))
    predicted_class = tf.to_int32(tf.greater(probs, 0.5))

    accuracy = tf.metrics.accuracy(binary_labels, predicted_class)
    auc = tf.metrics.auc(binary_labels, probs, name='auc')
    precision = tf.metrics.precision(binary_labels, predicted_class)
    recall = tf.metrics.recall(binary_labels, predicted_class)
    metrics = {'auc': auc, 
        'acc':accuracy, 
        'precision': precision,
        'recall': recall}
    for k, v in metrics.items():
        tf.summary.scalar(k, v[1])

    # evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        summary_hook = tf.train.SummarySaverHook(
            save_steps=200, output_dir=Config.model_path, 
            summary_op=tf.summary.merge_all())
        eval_hooks = [summary_hook]
        return tf.estimator.EstimatorSpec(mode, loss=loss, 
        evaluation_hooks=eval_hooks,
            eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    if Config.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=Config.learning_rate)
    elif Config.optimizer == "adagrad":
        optimizer = tf.train.AdagradOptimizer(learning_rate=Config.learning_rate)
    else:
        raise Exception('Unsupported optimizer...')
    
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, 
        "accuracy" : accuracy[1]}, every_n_iter=200)
    
    return tf.estimator.EstimatorSpec(mode, loss=loss, 
        train_op=train_op, training_hooks=[logging_hook])

def train_and_eval(predictor, train_files_pattern, eval_files_pattern):
    train_input_fn = file_based_input_fn_builder(
        input_files_pattern=train_files_pattern,
        is_training=True)
    eval_input_fn = file_based_input_fn_builder(
        input_files_pattern=eval_files_pattern,
        is_training=False)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        start_delay_secs=10,
        throttle_secs=Config.throttle_secs,
        input_fn=eval_input_fn)
    tf.estimator.train_and_evaluate(predictor, train_spec, eval_spec)

def build_predictor():
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
        config=tf.estimator.RunConfig(save_checkpoints_secs=30))
    return predictor
    
def main(_):
    # os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    os.environ['CUDA_VISIBLE_DEVICES'] = Config.visible_gpus
    tf.logging.set_verbosity(tf.logging.INFO)

    predictor = build_predictor()
    train_files_pattern = os.path.join(Config.data_dir, Config.input_files)
    eval_files_pattern = os.path.join(Config.data_dir, Config.eval_files)

    if FLAGS.do_train:
        train_and_eval(predictor, train_files_pattern, eval_files_pattern)
    
    if FLAGS.do_eval:
        eval_input_fn = file_based_input_fn_builder(
            input_files_pattern=eval_files_pattern,
            is_training=False)
        # change to None when use SST to evaluate all data
        result = predictor.evaluate(input_fn=eval_input_fn, steps=2000) 
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
    
    if FLAGS.do_predict:     
        test_files_pattern = os.path.join(Config.data_dir, Config.test_files)
        predict_input_fn = file_based_input_fn_builder(
            input_files_pattern=test_files_pattern,
            is_training=False
        )
        result = predictor.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(Config.model_path, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)

if __name__ == "__main__":
    tf.app.run()