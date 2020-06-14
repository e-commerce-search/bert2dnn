import tensorflow as tf
import argparse
from tokenizer import Tokenizer
import csv
from config import Config
import os

class InputExample(object):
    def __init__(self, text, label=None):
        self.text = text
        self.label = label


def read_tsv(input_file):
    with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

def get_train_examples(input_train_file):
    lines = read_tsv(input_train_file)
    examples = []
    for (i, line) in enumerate(lines):
        text = line[Config.idx_text]
        label = float(line[Config.idx_label])
        examples.append(InputExample(text, label))
    return examples

def create_tf_example(example, tokenizer):
    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f
    
    def create_float_feature(values):
        f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return f
    
    tokens = tokenizer.tokenize(example.text)
    tf_example = tf.train.Example()
    tf_example.features.feature['text'].bytes_list.value.extend(
        [t.encode() for t in tokens])
    tf_example.features.feature['label'].float_list.value.extend([float(example.label)])

    return tf_example


def file_based_convert_examples_to_tfrecord(
    examples, tokenizer, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    prev_text_a = None
    query_id = -1  # assuming continguous examples for same text_a.
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info(
                "Writing example %d of %d" % (ex_index, len(examples)))
        tf_example = create_tf_example(example, tokenizer)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print("Done write tfrecords to %s" %output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    args, _ = parser.parse_known_args()
    input_file = args.input_file
    output_file = args.output_file

    train_examples = get_train_examples(input_file)
    print("Number of train examples is %d" %len(train_examples))
    tokenizer = Tokenizer(Config.vocab_file)
    if not os.path.exists(output_file):
        file_based_convert_examples_to_tfrecord(train_examples, 
            tokenizer, output_file)