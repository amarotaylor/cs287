
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch
from torchtext import data, datasets
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
import numpy as np



class SequenceModel(nn.Module):
    def __init__(self, src_vocab_size, context_size,weight_init = 0.08):
        super(SequenceModel, self).__init__()
        self.context_size = context_size
        # embedding
        self.embedding = nn.Embedding(src_vocab_size, context_size)
        # langauge summarization
        self.lstm = nn.LSTM(input_size=context_size, hidden_size=context_size, num_layers=4, batch_first=True)
        for p in self.lstm.parameters():
            torch.nn.init.uniform_(p, a=weight_init, b=weight_init)

    def forward(self, inputs, h0=None):
        # embed the words 
        embedded = self.embedding(inputs)
        # summarize context
        context, hidden = self.lstm(embedded,h0)
        return context, hidden
    
class LanguageModel(nn.Module):
    def __init__(self, target_vocab_size, hidden_size, context_size, weight_init = 0.08 ):
        super(LanguageModel, self).__init__()
        # context is batch_size x seq_len x context_size
        # context to hidden
        self.embedding = nn.Embedding(target_vocab_size, hidden_size)
        # hidden to hidden 
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # decode hidden state for y_t
        for p in self.lstm.parameters():
            torch.nn.init.uniform_(p, a=weight_init, b=weight_init)
            
        self.translate = nn.Linear(hidden_size, target_vocab_size)

    def forward(self, inputs, h0=None):
        # embed the trg words
        embedded = self.embedding(inputs)
        embedded = torch.nn.functional.relu(embedded)
        # setting hidden state to context at t=0
        # otherwise context = prev hidden state
        output, hidden = self.lstm(embedded, h0)
        output = self.translate(output)
        return output,hidden

def repackage_hidden(h):
    return tuple(v.detach() for v in h)
def repackage_layer(hidden_s2c,hidden=100,BATCH_SIZE=32):
    return tuple([hidden_s2c[0][-1].detach().view(1,BATCH_SIZE,hidden),hidden_s2c[1][-1].detach().view(1,BATCH_SIZE,hidden)])
    
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
        trg = batch.trg.values.transpose(0,1)
        if src.shape[0]!=BATCH_SIZE:
            break
        else:
            # generate hidden state for decoder
            context, hidden_s2c = seq2context(src,h0)
            hidden = repackage_layer(hidden_s2c,context_size,BATCH_SIZE)
            output, hidden_lm = context2trg(trg[:,:-1],hidden)
            loss = criterion(output.transpose(2,1),trg[:,1:])
            mask = trg[:,1:]!=1
            loss = loss[mask].sum()
            #clip_grad_norm_(seq2context.parameters(), max_norm=5)
            #clip_grad_norm_(context2trg.parameters(), max_norm=5)
            loss.backward()
            seq2context_optimizer.step()
            context2trg_optimizer.step()
            var = torch.var(torch.argmax(lsm(output).cpu().detach(),2).float())
        if np.mod(ix,100) == 0:
            print('Epoch: {}, Batch: {}, loss: {}, Variance: {}'.format(e, ix, loss.cpu().detach(),var))
            
def validation_loop(e,val_iter,seq2context,context2trg,BATCH_SIZE):
    seq2context.eval()
    context2trg.eval()
    context_size = seq2context.context_size
    h0 = None
    total_loss = torch.tensor(0.0)
    track_mean = torch.tensor(0.0)
    for ix,batch in enumerate(val_iter):        
        src = batch.src.values.transpose(0,1)
        trg = batch.trg.values.transpose(0,1)
        if src.shape[0]!=BATCH_SIZE:
            x = 'blah'
        else:
            # generate hidden state for decoder
            context, hidden_s2c = seq2context(src,h0)
            hidden = repackage_layer(hidden_s2c,context_size)
            output, hidden_lm = context2trg(trg[:,:-1],hidden)
            loss = criterion(output.transpose(2,1),trg[:,1:])
            mask = trg[:,1:]!=1
            loss = loss[mask].detach().sum()
            mean_loss = loss/mask.sum().float()
            #clip_grad_norm_(seq2context.parameters(), max_norm=5)
            #clip_grad_norm_(context2trg.parameters(), max_norm=5)
            total_loss += loss
            track_mean += mean_loss
    ppl = torch.exp(track_mean/torch.tensor(ix).float())
    print('Epoch: {}, Validation total loss: {}, Validation ppl: {}'.format(e, total_loss, loss.cpu().detach(),ppl))
              