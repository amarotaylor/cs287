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
    seq2context_sch.step(ppl)
    context2trg_sch.step(ppl)
    
    print('Epoch: {}, Validation loss: {}, Validation ppl: {}'.format(e, total_loss/(BATCH_SIZE*len(val_iter)), ppl))
    return ppl



criterion_train = nn.CrossEntropyLoss(reduction='sum')
lsm2 = nn.LogSoftmax(dim=1)
def attn_training_loop(e,train_iter,seq2context,attn_context2trg,seq2context_optimizer,attn_context2trg_optimizer,BATCH_SIZE=32,context_size=500):
    seq2context.train()
    attn_context2trg.train()
    for ix,batch in enumerate(train_iter):
        src = batch.src.values.transpose(0,1)
        src = reverse_sequence(src)
        trg = batch.trg.values.transpose(0,1)
        if trg.shape[0] == BATCH_SIZE:
        
            seq2context_optimizer.zero_grad()
            attn_context2trg_optimizer.zero_grad()
        
            encoder_outputs, encoder_hidden = seq2context(src)
            loss = 0
            decoder_context = torch.zeros(BATCH_SIZE, context_size, device='cuda') # 32 x 500
            decoder_hidden = encoder_hidden
            sentence = []
            for j in range(trg.shape[1] - 1):
                word_input = trg[:,j]
                decoder_output, decoder_context, decoder_hidden, decoder_attention = attn_context2trg(word_input, decoder_context, decoder_hidden, encoder_outputs)
                #print(decoder_output.shape, trg[i,j+1].view(-1).shape)
                loss += criterion_train(decoder_output, trg[:,j+1])
                
                if np.mod(ix,100) == 0:
                    sentence.extend([torch.argmax(decoder_output[0,:],dim=0)])
                
            loss.backward()
            seq2context_optimizer.step()
            attn_context2trg_optimizer.step()
        
            if np.mod(ix,500) == 0:
                print('Epoch: {}, Batch: {}, Loss: {}'.format(e, ix, loss.cpu().detach()/BATCH_SIZE))
                #print([EN.vocab.itos[i] for i in sentence])
                #print([EN.vocab.itos[i] for i in trg[0,:]])
                
                
def attn_validation_loop(e,val_iter,seq2context,attn_context2trg,scheduler_c2t,scheduler_s2c,BATCH_SIZE=32,context_size=500):
    seq2context.eval()
    attn_context2trg.eval()
    total_loss = 0
    total_words = 0
    for ix,batch in enumerate(val_iter):
        src = batch.src.values.transpose(0,1)
        src = reverse_sequence(src)
        trg = batch.trg.values.transpose(0,1)
        if trg.shape[0] == BATCH_SIZE:
        
            encoder_outputs, encoder_hidden = seq2context(src)
            loss = 0
            decoder_context = torch.zeros(BATCH_SIZE, context_size, device='cuda') # 32 x 500
            decoder_hidden = encoder_hidden
            for j in range(trg.shape[1] - 1):
                word_input = trg[:,j]
                decoder_output, decoder_context, decoder_hidden, decoder_attention = attn_context2trg(word_input, decoder_context, decoder_hidden, encoder_outputs)
                #print(decoder_output.shape, trg[i,j+1].view(-1).shape)
                loss += criterion_train(decoder_output, trg[:,j+1])
                
                  
            total_loss += loss.detach()
            total_words += trg.shape[1]*BATCH_SIZE
    ppl = torch.exp(total_loss/total_words)
    scheduler_c2t.step(ppl)
    scheduler_s2c.step(ppl)
    print('Epoch: {}, Validation PPL: {}'.format(e,ppl))
    return ppl
        
                
class attn_RNNet_batched(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, weight_init=0.05):
        super(attn_RNNet_batched, self).__init__()
        self.emb = torch.nn.Sequential(torch.nn.Embedding(input_size, hidden_size), torch.nn.Dropout(dropout))
        self.rnn = torch.nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, batch_first=True, dropout=dropout)
        self.lnr = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(2*hidden_size, input_size))
    
        for f in self.parameters():
            torch.nn.init.uniform_(f, a=-weight_init, b=weight_init)
      
    def attn_dot(self,rnn_output,encoder_outputs):
        return F.softmax(torch.matmul(rnn_output.squeeze(0),encoder_outputs.transpose(0,1)).squeeze(),dim=0).unsqueeze(0).unsqueeze(0)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        word_embedded = self.emb(word_input)
        rnn_input = torch.cat([word_embedded, last_context], 1).unsqueeze(1) # batch x 1 x hiddenx2
        rnn_output, hidden = self.rnn(rnn_input, last_hidden)
        attn_weights = rnn_output.bmm(encoder_outputs.transpose(1,2))# batch x src_seqlen x 1
        context = attn_weights.bmm(encoder_outputs)
        rnn_output = rnn_output.squeeze(1)
        context = context.squeeze(1)
        output = self.lnr(torch.cat((rnn_output, context), 1))
        # prediction, last_context, last_hidden, weights for vis
        return output, context, hidden, attn_weights 
    
    
def load_model(model,state_dict_path):
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)

    
              