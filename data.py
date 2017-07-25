import os
import torch
from joblib import Parallel, delayed

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)



def read_words(line):
    words = line.split() + ['<eos>']
    return words

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize_v2(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize_v2(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize_v2(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path), path
        # Add words to the dictionary
        with open(path, 'r') as f:
            # TODO: joblib
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

    def tokenize_v2(self, path):
        assert os.path.exists(path), path
        print (path)
        result = None
        # f = open(path, 'r').readlines()
        with open(path, 'r') as f:
            result = Parallel(n_jobs=6, verbose=1, backend='threading')(delayed(read_words)(line) for line in f)
        # print('breakpoint')
        tokens = 0
        for words in result:
            tokens += len(words)
            for word in words:
                self.dictionary.add_word(word)
        ids = torch.LongTensor(tokens)
        token = 0
        for words in result:
            for word in words:
                ids[token] = self.dictionary.word2idx[word]
                token += 1
        return ids
