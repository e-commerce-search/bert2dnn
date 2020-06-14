import collections
import tensorflow as tf
import six

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    print("Done... vocab_size is %d" % len(vocab))
    return vocab

def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if item in vocab:
            output.append(vocab[item])
        else:
            output.append(-1)
    return output

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class Tokenizer(object):
    def __init__(self, vocab_file):
        print("init")
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, text):
        # unigram and bigram tokenizer
        tokens = text.lower().split(' ')
        bigrams = [' '.join(pair) for pair in zip(['^'] + tokens, tokens + ['$s'])]
        # to keep the sequence of word by zip bigram and unigram together
        return [t for (b, u) in zip(bigrams, tokens + ['$'])
                for t in [b, u]][:-1]  
    
    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)
    
    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

