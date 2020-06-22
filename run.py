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
import model

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_bool("do_train", True, "Whether to run training and eval")
flags.DEFINE_bool("do_eval", False, "Whether to run eval.")
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
            random.Random(14).shuffle(filenames)
            tf.logging.info("train files: %s" % ','.join(filenames))

        d = tf.data.TFRecordDataset(filenames)
        if is_training:
            d = d.repeat(Config.epoches)
            d = d.shuffle(buffer_size=2000)
        d = d.map(_decode_record)
        d = d.batch(Config.batch_size)
        return d
    return input_fn

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
        key='text', vocabulary_file=Config.vocab_file, vocabulary_size=Config.vocab_size,
        num_oov_buckets=0, dtype=tf.string)
    emb_initializer = tf.variance_scaling_initializer(scale=1.0, seed=1, mode='fan_in')
    word_embedding_column = tf.feature_column.embedding_column(
        column, dimension=Config.embedding_size, combiner="sqrtn", initializer=emb_initializer)
    
    predictor = tf.estimator.Estimator(
        model_fn=model.dnn_model_fn,
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
        result = predictor.evaluate(input_fn=eval_input_fn, steps=None) 
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