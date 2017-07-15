import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from util import load_word_vectors
import data
from model import ModelWrapper
import sys
from meowlogtool import log_util



parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--embedding_one', type=str, default='../treelstm.pytorch/data/glove/glove.840B.300d',
                    help='location of the data corpus')
parser.add_argument('--embedding_two', type=str, default='/media/vdvinh/25A1FEDE380BDADA/ff/glove_sorted/glove.840B.300d',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--noglove', action='store_true',
                    help='NOT use glove pre-train')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='tmp_save',
                    help='path to save the final model')
parser.add_argument('--channel', type = int, default=1)
args = parser.parse_args()

if not os.path.exists(args.save):
    os.makedirs(args.save)

# log to console and file
logger1 = log_util.create_logger(os.path.join(args.save, 'word_language_model'), print_console=True)
logger1.info("LOG_FILE")  # log using logger1

# attach log to stdout (print function)
s1 = log_util.StreamToLogger(logger1)
sys.stdout = s1

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
# model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
model = ModelWrapper(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied, channel = args.channel)

if args.cuda:
    model.cuda()

if not args.noglove:
    emb_torch = 'sst_embed1.pth'
    emb_torch2 = 'sst_embed2.pth'
    emb_vector_path = args.embedding_one
    emb_vector_path2 = args.embedding_two
    assert os.path.isfile(emb_vector_path + '.txt')
    # assert os.path.isfile(emb_vector_path2 + '.txt')
    ##########################################
    is_preprocessing_data = False
    emb_file = os.path.join(args.data, emb_torch)
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
        print('load %s' % (emb_file))
    else:
    # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(emb_vector_path, ' ')
        print('==> Embedding vocabulary size: %d ' % glove_vocab.size())
        emb = torch.zeros(ntokens, glove_emb.size(1))
        for word in corpus.dictionary.word2idx.keys():
            if glove_vocab.getIndex(word):
                emb[corpus.dictionary.word2idx[word]] = glove_emb[glove_vocab.getIndex(word)]
            else:
                emb[corpus.dictionary.word2idx[word]] = torch.Tensor(emb[corpus.dictionary.word2idx[word]].size()).normal_(-0.05, 0.05)
        torch.save(emb, emb_file)
        glove_emb = None
        glove_vocab = None
        is_preprocessing_data = True  # flag to quit
        print('done creating emb, quit')
    #####################
    if args.channel ==2:
        emb_file2 = os.path.join(args.data, emb_torch2)
        if os.path.isfile(emb_file2):
            emb2 = torch.load(emb_file2)
            print('load %s' % (emb_file2))
        else:
        # load glove embeddings and vocab
            glove_vocab, glove_emb = load_word_vectors(emb_vector_path2, ' ')
            print('==> Embedding vocabulary size: %d ' % glove_vocab.size())
            emb2 = torch.zeros(ntokens, glove_emb.size(1))
            for word in corpus.dictionary.word2idx.keys():
                if glove_vocab.getIndex(word):
                    emb2[corpus.dictionary.word2idx[word]] = glove_emb[glove_vocab.getIndex(word)]
                else:
                    emb2[corpus.dictionary.word2idx[word]] = torch.Tensor(emb2[corpus.dictionary.word2idx[word]].size()).normal_(-0.05, 0.05)
            torch.save(emb2, emb_file2)
            glove_emb = None
            glove_vocab = None
            is_preprocessing_data = True  # flag to quit
            print('done creating emb, quit')



    #######################################################
    if is_preprocessing_data:
        quit()

    model.encoder.state_dict()['weight'].copy_(emb)
    if args.channel == 2:
        model.encoder2.state_dict()['weight'].copy_(emb2)
else:
    print('not use pretrained glove')


criterion = nn.CrossEntropyLoss()
print ('--model info --')
print(model)
###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    d = source[i:i+seq_len]
    t = source[i+1:i+1+seq_len]
    # per = torch.randperm(d.size(1))
    # if args.cuda:
    #     per = per.cuda()
    # t = torch.transpose(t, 0, 1)
    # d = torch.transpose(d, 0, 1)
    # t = t[per]
    # d = d[per]
    # t = torch.transpose(t, 0, 1).contiguous()
    # d = torch.transpose(d, 0, 1).contiguous()
    data = Variable(d, volatile=evaluation)
    target = Variable(t.view(-1))
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # data (seq_len, batch_size) indices
        # target (seq_len * batch_size)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-lr, p.grad.data)

        for p in model.conv_module.parameters():
            p.data.add_(-lr, p.grad.data)

        for p in model.rnn.parameters():
            p.data.add_(-lr, p.grad.data)

        for p in model.decoder.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(args.save, 'model.pt'), 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')



# Load the best saved model.
with open(os.path.join(args.save, 'model.pt'), 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

# save state_dict()
model.save_state_files(args.save)

# python main.py --data ./data/movie5000 --cuda --emsize 300 --nhid 168 --dropout 0.5 --epochs 10
html_log = log_util.up_gist(os.path.join(args.save, 'word_language_model.log'), 'test_doggy', 'test_doggy')
print('link on gist %s' % (html_log))

# python main.py --cuda --emsize 300 --nhid 150 --dropout 0.5 --epochs 40 --save saved_penn3