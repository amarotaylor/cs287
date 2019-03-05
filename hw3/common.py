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
                loss = criterion(decoder_output, trg[:,j+1])
                mask = trg[:,j+1]!=1
                total_words += mask.sum()
                track_loss = torch.sum(loss[mask.squeeze()])
            total_loss += track_loss.detach()
            
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

# numpy softmax
def softmax(X, theta = 1.0, axis = None):

    y = np.atleast_2d(X)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(theta)

    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    y = np.exp(y)

    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    p = y / ax_sum

    if len(X.shape) == 1: p = p.flatten()

    return p    
              
def beamsearch(model, seq2context, context2trg, context_size, src, beam_width, max_len, output_width=1, alpha=1, BATCH_SIZE=32, padding=False, EN=None):
    '''
    run beam search and return top predictions
        - model: {'s2s', 's2s_attn'}
        - seq2context: encoder model
        - context2trg: decoder model
        - context_size: hidden size
        - src: tensor of source sentences
        - beam_width: beam search width
        - max_len: maximum length for predictions
        - output_width: number of predictions to return per sentence
        - alpha: string length discount rate; e.g., normalizing factor = 1/(T^alpha)
        - BATCH_SIZE: src batch size
        - padding: pad predictions to max_len
        - EN: English dictionary for string-index conversions
    '''
    # set up
    START_TKN = EN.vocab.stoi["<s>"]
    END_TKN = EN.vocab.stoi["</s>"]
    BEAM_WIDTH = beam_width
    lsm = nn.LogSoftmax(dim=1)
    
    # run forward pass of encoder once
    encoder_outputs, encoder_hidden = seq2context(src)
    decoder_context = torch.zeros(BATCH_SIZE, context_size, device='cuda') # 32 x 500
    decoder_hidden = encoder_hidden
    
    # prepare for beam search
    b_string = torch.zeros((BATCH_SIZE, max_len, BEAM_WIDTH), device='cuda') # stores the top BEAM_WIDTH strings
    b_string[:,0,:] = START_TKN
    b_probs = {} # stores the top BEAM_WIDTH probs
    '''
    b_probs key = tuple(batch idx, beam idx)
    b_probs val = [cum log prob, length]
    '''
    done = {} # stores the finished strings
    '''
    done key = batch idx
    done val = [str, cum log prob, length]
    '''
    predictions = {} # stores the top output_width predictions
    for b in range(BATCH_SIZE):
        done[b] = []
        predictions[b] = []
        for c in range(BEAM_WIDTH):
            b_probs[(b, c)] = [0, 1]

    # loop through target sequence max len
    for i in range(1,max_len):
        print('position:',i)
        if i == 1: # if predicting the word following <s>, take top BEAM_WIDTH preds
            if model == 's2s':
                word_input = b_string[:,0:i,0].long()
                decoder_output, decoder_hidden = context2trg(word_input, encoder_hidden)
                decoder_output = decoder_output.detach()[:,-1,:]
            elif model == 's2s_attn':
                word_input = b_string[:,i-1,0].long()
                decoder_output, decoder_context, decoder_hidden, decoder_attention = context2trg(word_input, 
                                                                                             decoder_context, 
                                                                                             decoder_hidden, 
                                                                                             encoder_outputs)
            logprobs = lsm(decoder_output.detach()) # BATCH_SIZE x VOCAB_SIZE
            toppreds = torch.argsort(logprobs, dim=1, descending=True)[:,0:BEAM_WIDTH] # BATCH_SIZE x BEAM_WIDTH
            b_string[:,i,:] = toppreds
            for b in range(BATCH_SIZE):
                for c in range(BEAM_WIDTH):
                    b_probs[tuple((b,c))][0] += logprobs[b, toppreds[b,c]]
                    b_probs[tuple((b,c))][1] += 1
        elif i == max_len - 1:
            stored_logprobs = torch.zeros((BATCH_SIZE, BEAM_WIDTH, len(EN.vocab)))
            for j in range(BEAM_WIDTH):
                if model == 's2s':
                    word_input = b_string[:,0:i,j].long()
                    decoder_output, decoder_hidden = context2trg(word_input, encoder_hidden)
                    decoder_output = decoder_output.detach()[:,-1,:]
                elif model == 's2s_attn':
                    word_input = b_string[:,i-1,j].long()
                    decoder_output, decoder_context, decoder_hidden, decoder_attention = context2trg(word_input, 
                                                                                                     decoder_context, 
                                                                                                     decoder_hidden, 
                                                                                                     encoder_outputs)
                logprobs = lsm(decoder_output.detach()) # unsorted log probs
                stored_logprobs[:,j,:] = logprobs
        else: # if predicting the word for positions 2+, compare top BEAM_WIDTH preds for each of BEAM_WIDTH strings
            curr_probs = {} # temporary storage
            curr_string = torch.zeros(BATCH_SIZE, i+1, BEAM_WIDTH) # temporary storage

            for j in range(BEAM_WIDTH):
                if j % 1 == 0:
                    print('beam:',j)
                if model == 's2s':
                    word_input = b_string[:,0:i,j].long()
                    decoder_output, decoder_hidden = context2trg(word_input, encoder_hidden)
                    decoder_output = decoder_output.detach()[:,-1,:]
                elif model == 's2s_attn':
                    word_input = b_string[:,i-1,j].long()
                    decoder_output, decoder_context, decoder_hidden, decoder_attention = context2trg(word_input, 
                                                                                                     decoder_context, 
                                                                                                     decoder_hidden, 
                                                                                                     encoder_outputs)
                logprobs = lsm(decoder_output.detach()) # unsorted log probs
                sortedpreds = torch.argsort(logprobs, dim=1, descending=True) # sorted words
                toppreds = sortedpreds[:,0:BEAM_WIDTH] # top words

                # check if any top preds are </s>
                for b in range(BATCH_SIZE):
                    if END_TKN in toppreds[b,:]: # if </s> in top preds
                        # track finished strings
                        done_string = torch.cat((b_string[b,0:i,j],torch.tensor([END_TKN], device='cuda').float()))
                        done_prob = b_probs[tuple((b,j))][0] + logprobs[b,END_TKN]
                        done[b].append([done_string, done_prob, done_string.shape[0]])
                        # replace </s> with 4th best pred
                        done_idx = (toppreds[b,:] == END_TKN).nonzero()
                        toppreds[b,done_idx] = sortedpreds[b,BEAM_WIDTH]

                if j == 0: # if preds are from first beam, take top BEAM_WIDTH preds (temporarily)
                    for b in range(BATCH_SIZE):
                        for c in range(BEAM_WIDTH):
                            new_b_prob = b_probs[tuple((b,j))][0] + logprobs[b,toppreds[b,c]]
                            curr_probs[tuple((b,c))] = new_b_prob # set top prob
                            curr_string[b,0:i,c] = b_string[b,0:i,j] # set sentence
                            curr_string[b,i,c] = toppreds[b,c] # set top word
                else: # if preds are from subsequent beams, compare to existing
                    for b in range(BATCH_SIZE):
                        for c in range(BEAM_WIDTH): # proposed strings
                            replaced = False
                            for d in range(BEAM_WIDTH): # existing strings
                                new_b_prob = b_probs[tuple((b,j))][0] + logprobs[b,toppreds[b,c]]
                                if new_b_prob > curr_probs[tuple((b,d))] and not replaced:
                                    curr_probs[tuple((b,d))] = new_b_prob # update top prob
                                    curr_string[b,0:i,d] = b_string[b,0:i,j] # update sentence
                                    curr_string[b,i,d] = toppreds[b,c] # update top word
                                    replaced = True  
            b_string[:,0:i+1,:] = curr_string
            # update top strings, probs
            for b in range(BATCH_SIZE):
                for c in range(BEAM_WIDTH):
                    b_probs[tuple((b,c))][0] = curr_probs[tuple((b,c))]
                    b_probs[tuple((b,c))][1] += 1

    K = output_width
    for b in range(BATCH_SIZE):
        if len(done[b]) < K:
            gap = K - len(done[b])
            # probs = torch.tensor([b_probs[tuple((b,j))][0] for j in range(BEAM_WIDTH)], device='cuda')
            probs = torch.argsort(stored_logprobs[b].view(-1), descending=True)
            sorted_logprobs = torch.tensor(divmod(probs.numpy(), len(EN.vocab)), device='cuda')
            sorted_logprobs = sorted_logprobs.transpose(0,1) # BEAM_WIDTHxVOCAB_SIZE x 2
            for c in sorted_logprobs[0:gap]:  
                #d = c.item()
                d = c[0].item()
                idx = c[1].item()
                b_string[b,-1,d] = idx
                done_string = b_string[b,:,d].long()
                done_prob = b_probs[tuple((b,d))][0] + stored_logprobs[b,d,idx]
                done_len = max_len
                done[b].append([done_string, done_prob, done_len])
                
    for b in range(BATCH_SIZE):
        print('batch:',b)
        normalized_probs = torch.tensor([], device='cuda')
        for sentence in range(len(done[b])):
            normalized = torch.tensor([done[b][sentence][1]/done[b][sentence][2]**alpha], device='cuda')
            normalized_probs = torch.cat((normalized_probs,normalized),0)
        top = torch.argsort(normalized_probs, descending=True)[0:K]
        for k in range(K):
            best = done[b][top[k]]
            if padding:
                m = nn.ConstantPad1d((0, max_len - best[2]), EN.vocab.stoi['<pad>'])
                predictions[b].append(m(best[0].long()))
            else:
                predictions[b].append(best[0].long())
            if k % 10 == 0:
                print([EN.vocab.itos[i] for i in best[0].long()])
    
    return predictions


def attn_training_split_loop(e,train_iter,seq2context,attn_context2trg,seq2context_optimizer,attn_context2trg_optimizer,BATCH_SIZE=32,context_size=500,EN=None):
    seq2context.train()
    attn_context2trg.train()
    for ix,batch in enumerate(train_iter):
        src = batch.src.values.transpose(0,1)
        src = reverse_sequence(src)
        trg = batch.trg.values.transpose(0,1)
        if trg.shape[0] == BATCH_SIZE:
        
            seq2context_optimizer.zero_grad()
            attn_context2trg_optimizer.zero_grad()
            loss = 0
            decoder_context = torch.zeros(BATCH_SIZE, context_size, device='cuda') # 32 x 500
            encoder_outputs, encoder_hidden = seq2context(src)    
            decoder_hidden = encoder_hidden
            sentence = []
            p = np.random.rand()
            if p > 0.5:
                
            
                for j in range(trg.shape[1] - 1):
                    word_input = trg[:,j]
                    decoder_output, decoder_context, decoder_hidden, decoder_attention = attn_context2trg(word_input, decoder_context, decoder_hidden, encoder_outputs)
                    #print(decoder_output.shape, trg[i,j+1].view(-1).shape)
                    loss += criterion_train(decoder_output, trg[:,j+1])
                
                    if np.mod(ix,100) == 0:
                        sentence.extend([torch.argmax(decoder_output[0,:],dim=0)])
            else:
                
                predictions = beamsearch(seq2context, attn_context2trg, context_size, src,EN=EN, beam_width=1, max_len=trg.shape[1], output_width=1, alpha=1, padding=True)
                predictions = torch.stack([t[0] for t in predictions.values()])
                for j in range(predictions.shape[1] - 1):
                    word_input = predictions[:,j]
                    decoder_output, decoder_context, decoder_hidden, decoder_attention = attn_context2trg(word_input, decoder_context, decoder_hidden, encoder_outputs)
                    #print(decoder_output.shape, trg[i,j+1].view(-1).shape)
                    loss += criterion_train(decoder_output, trg[:,j+1])
            loss.backward()
            seq2context_optimizer.step()
            attn_context2trg_optimizer.step()
        
            if np.mod(ix,500) == 0:
                print('Epoch: {}, Batch: {}, Loss: {}'.format(e, ix, loss.cpu().detach()/BATCH_SIZE))
                #print([EN.vocab.itos[i] for i in sentence])
                #print([EN.vocab.itos[i] for i in trg[0,:]])