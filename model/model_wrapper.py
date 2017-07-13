import torch.nn as nn
from torch.autograd import Variable
from conv_model import MultiConvModule
import torch
import os
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        '''

        :param input: (35, 20) (seq_lenm, batch_size)
        :param hidden:
        :return: (35, 20, 10000)
        '''
        emb = self.drop(self.encoder(input)) # (seq, batch, emb_dim)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output) # (seq, batch, emb_dim)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

class ModelWrapper(nn.Module):
    """Container module with an encoder, a custom module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(ModelWrapper, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # your model goes here
        self.conv_module = MultiConvModule(1, ninp, 1, [100, 100], [5, 3])
        in_dim = 200
        self.bidirectional = False
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=nhid, bidirectional=self.bidirectional)
        # and end here
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        # save state file for transfer learning
        self.conv_state_file = 'convolution_state_dict.pth'
        self.lstm_state_file = 'lstm_state_dict.pth'

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def save_state_files(self, dir):
        '''
        Save state dict to file
        :param dir: where to save file
        :return:
        '''
        torch.save(self.conv_module.state_dict(), os.path.join(dir, self.conv_state_file))
        torch.save(self.rnn.state_dict(), os.path.join(dir, self.lstm_state_file))
        print ('save state file to %s'%(dir))

    def forward(self, input, hidden):
        '''
        Model Wrapper forward
        :param input: (35, 20) (seq_lenm, batch_size)
        :param hidden:
        :return: (35, 20, 10000)
        '''
        emb = self.drop(self.encoder(input)) # (seq, batch, emb_dim)
        emb = emb.unsqueeze(1) # (seq, 1, batch, emb_dim)
        c_out = self.conv_module(emb, batch = True)
        output, hidden = self.rnn(c_out, hidden)
        output = self.drop(output) # (seq, batch, emb_dim)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded_output = decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
        return decoded_output # (35, 20, 10000) |

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())