import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

class ConvModule(nn.Module):
    def __init__(self, cuda, emb_dim, n_filter, kernel_size):
        super(ConvModule, self).__init__()
        self.cudaFlag = cuda
        self.padding_size = kernel_size/2
        self.conv = nn.Conv2d(1, n_filter, (kernel_size, emb_dim), padding=(self.padding_size,0))
        self.in_dropout = nn.Dropout(p=0.5)
        self.out_dropout = nn.Dropout(p=0.1)
        if self.cudaFlag:
            self.conv = self.conv.cuda()
            self.in_dropout = self.in_dropout.cuda()
            self.out_dropout = self.out_dropout.cuda()

    def forward(self, sentence):
        """
        Forward function
        :param sentence: sentence embedding matrix (seq_length, 1, emb_dim)
        :return: (seq_length, 1, n_filter)
        """
        sentence = self.in_dropout(sentence)
        sentence = sentence.unsqueeze(2) # (seq_len, 1, 1, emb_dim)
        sentence = torch.transpose(sentence, 0, 2) # (1, 1. seq_len, emb_dim)
        output = self.conv(sentence) # (1, n_filter, seq_len, 1)
        output = output.squeeze(3) #
        output = torch.transpose(output, 0, 2) # (1, n_filter, seq_len)
        output = torch.transpose(output, 1, 2) # (seq_len. n_filter, 1)
        output = self.out_dropout(output) # (seq_len, 1, n_filter)
        return output


class MultiConvModule(nn.Module):
    def __init__(self, cuda, emb_dim, in_channel, n_filters, kernel_sizes, pooling = False):
        super(MultiConvModule, self).__init__()
        self.cudaFlag = cuda
        self.n_conv = len(n_filters)
        self.paddingList = []
        self.pooling = pooling
        self.convList = nn.ModuleList()
        for i in range(self.n_conv):
            kernel_size = kernel_sizes[i]
            n_filter = n_filters[i]
            padding = (kernel_size-1)/2
            conv = nn.Conv2d(in_channel, n_filter, (kernel_size, emb_dim), padding=(padding,0))
            self.paddingList.append(padding)
            self.convList.append(conv)

        self.in_dropout = nn.Dropout(p=0.5)
        self.out_dropout = nn.Dropout(p=0.2)
        if pooling:
            self.max_pooling = nn.MaxPool2d(kernel_size=2, padding=1)
        if self.cudaFlag:
            self.convList = self.convList.cuda()
            self.in_dropout = self.in_dropout.cuda()
            self.out_dropout = self.out_dropout.cuda()
            if pooling:
                self.max_pooling = self.max_pooling.cuda()

    def forward(self, sentence):
        """
        Forward function
        :param sentence: sentence embedding matrix (seq_length, 1, emb_dim)
        :return: (seq_length, 1, n_filter)
        """
        sentence = self.in_dropout(sentence) # (seq_len, in_channel, emb_dim)
        sentence = sentence.unsqueeze(2) # (seq_len, in_channel, 1, emb_dim)
        sentence = torch.transpose(sentence, 0, 2) # (1, in_channel, seq_len, emb_dim)
        #output = self.conv(sentence) # (1, n_filter, seq_len, 1)
        outputList = []
        for i in range(self.n_conv):
            o1 = self.convList[i](sentence)
            if self.pooling:
                o2 = self.max_pooling(o1)
            else:
                o2 = o1
            output = F.relu(o2)
            outputList.append(output)
        output = torch.cat(outputList, 1)

        output = output.squeeze(3) # (1, emb_size, seq)
        output = torch.transpose(output, 0, 2) # (seq, emb_size, 1)
        output = torch.transpose(output, 1, 2) # (seq, emb_size, 1)
        output = self.out_dropout(output) # (seq, 1, emb)
        return output

if __name__ == "__main__":
    emb_size = 300
    sentence = Var(torch.rand(24, 1, emb_size))
    conv = MultiConvModule(0, 300, 1, [100,200,300], [3,5,7])
    output = conv(sentence)
    print (output.size())
