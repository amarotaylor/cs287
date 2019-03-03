import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from torchtext import data, datasets
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
import numpy as np

class SequenceModel(nn.Module):
    def __init__(self, src_vocab_size, context_size, num_layers=2, weight_init=0.08, dropout=0.4):
        super(SequenceModel, self).__init__()
        self.context_size = context_size
        # embedding
        self.embedding = nn.Embedding(src_vocab_size, context_size)
        # langauge summarization
        self.lstm = nn.LSTM(input_size=context_size, hidden_size=context_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        #for p in self.lstm.parameters():
        #    torch.nn.init.uniform_(p, a=weight_init, b=weight_init)
    def forward(self, inputs, h0=None):
        # embed the words 
        embedded = self.embedding(inputs)
        # summarize context
        context, hidden = self.lstm(embedded,h0)
        return context, hidden
    
class LanguageModel(nn.Module):
    def __init__(self, target_vocab_size, hidden_size, context_size, weight_init = 0.08,dropout=0.4):
        super(LanguageModel, self).__init__()
        # context is batch_size x seq_len x context_size
        # context to hidden
        self.embedding = nn.Embedding(target_vocab_size, hidden_size)
        # hidden to hidden 
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # decode hidden state for y_t
        #for p in self.lstm.parameters():
        #    torch.nn.init.uniform_(p, a=weight_init, b=weight_init)
            
        self.translate = nn.Linear(hidden_size, target_vocab_size)

    def forward(self, inputs, h0=None):
        # embed the trg words
        embedded = self.embedding(inputs)
        # setting hidden state to context at t=0
        # otherwise context = prev hidden state
        output, hidden = self.lstm(embedded, h0)
        output = self.translate(output)
        return output,hidden
    
    
    
# LSTM RNN
class RNNet(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, weight_tie=False, weight_init=0.05):
        super(RNNet, self).__init__()
        self.emb = torch.nn.Sequential(torch.nn.Embedding(input_size, hidden_size), torch.nn.Dropout(dropout))
        self.rnn = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, batch_first=True, dropout=dropout)
        self.lnr = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(hidden_size, input_size))
    
        for f in self.parameters():
            torch.nn.init.uniform_(f, a=-weight_init, b=weight_init)
      
        if weight_tie == True:
            self.lnr[1].weight.data=self.emb[0].weight.data
      
    def forward(self, inputs, h0=None):
        x = self.emb(inputs) # batch x seqlen x hidden
        x, hidden = self.rnn(x, h0) # batch x seqlen x hidden
        y = self.lnr(x) # batch x seqlen x vocab
        return y, hidden    
    
    

def repackage_hidden(h):
    return tuple(v.detach() for v in h)
def reverse_sequence(src):
    length = list(src.shape)[1]
    idx = torch.linspace(length-1, 0, steps=length).long()
    rev_src = src[:,idx]
    return rev_src

lsm = torch.nn.LogSoftmax(dim=2)
criterion = nn.CrossEntropyLoss(reduction='none')

def training_loop(e,train_iter,seq2context,context2trg,seq2context_optimizer,context2trg_optimizer,BATCH_SIZE):
    seq2context.train()
    context2trg.train()
    context_size = seq2context.context_size
    h0 = None
    for ix,batch in enumerate(train_iter):
        seq2context_optimizer.zero_grad()
        context2trg_optimizer.zero_grad()
        
        src = batch.src.values.transpose(0,1)
        src = reverse_sequence(src)
        trg = batch.trg.values.transpose(0,1)
        if src.shape[0]!=BATCH_SIZE:
            break
        else:
            # generate hidden state for decoder
            context, hidden_s2c = seq2context(src)
            output, hidden_lm = context2trg(trg[:,:-1],hidden_s2c)
            loss = criterion(output.transpose(2,1),trg[:,1:])
            mask = trg[:,1:]!=1
            loss = loss[mask].sum()
            clip_grad_norm_(seq2context.parameters(), max_norm=5)
            clip_grad_norm_(context2trg.parameters(), max_norm=5)
            loss.backward()
            seq2context_optimizer.step()
            context2trg_optimizer.step()
            var = torch.var(torch.argmax(lsm(output).cpu().detach(),2).float())
        if np.mod(ix,100) == 0:
            print('Epoch: {}, Batch: {}, Loss: {}, Variance: {}'.format(e, ix, loss.cpu().detach()/BATCH_SIZE, var))
            
def validation_loop(e,val_iter,seq2context,context2trg,seq2context_sch,context2trg_sch,BATCH_SIZE):
    seq2context.eval()
    context2trg.eval()
    context_size = seq2context.context_size
    h0 = None
    total_loss = torch.tensor(0.0)
    track_mean = torch.tensor(0.0)
    total_words = torch.tensor(0.0)
    for ix,batch in enumerate(val_iter):        
        src = batch.src.values.transpose(0,1)
        src = reverse_sequence(src)
        trg = batch.trg.values.transpose(0,1)
        if src.shape[0]!=BATCH_SIZE:
            x = 'blah'
        else:
            # generate hidden state for decoder
            context, hidden_s2c = seq2context(src,h0)
            #hidden = repackage_layer(hidden_s2c,context_size)
            output, hidden_lm = context2trg(trg[:,:-1],hidden_s2c)
            loss = criterion(output.transpose(2,1),trg[:,1:])
            mask = trg[:,1:]!=1
            loss = loss[mask].detach().sum()
            #mean_loss = loss/mask.sum().float()
            #clip_grad_norm_(seq2context.parameters(), max_norm=5)
            #clip_grad_norm_(context2trg.parameters(), max_norm=5)
            total_loss += loss
            #track_mean += mean_loss
            total_words += mask.sum().float()
    ppl = torch.exp(total_loss/total_words)
    seq2context_sch.step()
    context2trg_sch.step()
    print('Epoch: {}, Validation loss: {}, Validation ppl: {}'.format(e, total_loss/(BATCH_SIZE*len(val_iter)), ppl))
              