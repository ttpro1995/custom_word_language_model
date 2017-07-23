# import pytest
import argparse
import data
import torch.nn as nn
import os
from util import save_word_vector
def save_text_word_vector():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--save', type=str, default='saved/test',
                        help='path to save the final model')
    parser.add_argument('--data', type=str, default='../data/penn',
                        help='location of the data corpus')
    args = parser.parse_args()
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    args.cuda = True
    corpus = data.Corpus(args.data)
    emb_model = nn.Embedding(len(corpus.dictionary.idx2word), 300)
    emb_model = emb_model.cuda()
    save_word_vector(args, 'testemb.txt', corpus, emb_model)

save_text_word_vector()