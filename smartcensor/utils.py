"""Utilities for reading files."""

import collections.abc

def progress(iterable):
    """Iterate over `iterable`, showing progress if appropriate."""
    try:
        import tqdm
        return tqdm.tqdm(iterable, disable=None)
    except ImportError:
        return iterable

class Vocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""
    def __init__(self):
        words = {'<BOS>', '<EOS>', '<UNK>'}
        self.num_to_word = list(words)    
        self.word_to_num = {word:num for num, word in enumerate(self.num_to_word)}
    def add(self, word):
        if word in self: return
        num = len(self.num_to_word)
        self.num_to_word.append(word)
        self.word_to_num[word] = num
    def discard(self, word):
        raise NotImplementedError()
    def update(self, words):
        self |= words
    def __contains__(self, word):
        return word in self.word_to_num
    def __len__(self):
        return len(self.num_to_word)
    def __iter__(self):
        return iter(self.num_to_word)

    def numberize(self, word):
        """Convert a word into a number."""
        if word in self.word_to_num:
            return self.word_to_num[word]
        else: 
            return self.word_to_num['<UNK>']

    def denumberize(self, num):
        """Convert a number into a word."""
        return self.num_to_word[num]

def read_parallel(ffilename, efilename):
    """Read data from the files named by `ffilename` and `efilename`.

    The files should have the same number of lines.

    Arguments:
      - ffilename: str
      - efilename: str
    Returns: list of pairs of lists of strings. <BOS> and <EOS> are added to all sentences.
    """
    data = []
    for (fline, eline) in zip(open(ffilename), open(efilename)):
        fwords = ['<BOS>'] + fline.split() + ['<EOS>']
        ewords = ['<BOS>'] + eline.split() + ['<EOS>']
        data.append((fwords, ewords))
    return data

def read_mono(filename):
    """Read sentences from the file named by `filename`.

    Argument: filename
    Returns: list of lists of strings. <BOS> and <EOS> are added to each sentence.
    """
    data = []
    for line in open(filename):
        words = ['<BOS>'] + line.split() + ['<EOS>']
        data.append(words)
    return data

def write_mono(data, filename):
    """Write sentences to the file named by `filename`.

    Arguments:
    - data: list of lists of strings. <BOS> and <EOS> are stripped off.
    - filename: str
    """
    with open(filename, 'w') as outfile:
        for words in data:
            if len(words) > 0 and words[0] == '<BOS>': words.pop(0)
            if len(words) > 0 and words[-1] == '<EOS>': words.pop(-1)
            print(' '.join(words), file=outfile)
