import os
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
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
        self.nvocab = 0
        self.dictionary = Dictionary()
        self.loadVocabFile(os.path.join(path, 'vocab-cased.txt'))
        #self.loadVocabFile(os.path.join(path, 'vocab-cased.txt'))
        self.train = self.tokenize_v3(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize_v3(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize_v3(os.path.join(path, 'test.txt'))


    # Load entries from a file.
    def loadVocabFile(self, filename):
        idx = 0
        self.dictionary.add_word('<unk>') # unknown word
        self.dictionary.add_word('<eos>')
        idx +=2
        for line in open(filename):
            token = line.rstrip('\n')
            self.dictionary.add_word(token)
            idx += 1
        self.nvocab = idx
        print ('nvocab %s' %(self.nvocab))

    def tokenize_v3(self, path):
        '''
        Tokenize only known word.
        :param path:
        :return:
        '''
        assert os.path.exists(path), path
        # Add words to the dictionary
        with open(path, 'r') as f:
            # TODO: joblib
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in tqdm(f):
                words = line.split() + ['<eos>']
                for word in words:
                    # do no use word2idx.keys() because it is much slower
                    if word not in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx['<unk>']
                    else:
                        ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids


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
            for line in tqdm(f):
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
