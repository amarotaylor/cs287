import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from torchtext import data, datasets
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
import numpy as np

# set up
lsm2 = nn.LogSoftmax(dim=1)
lsm = torch.nn.LogSoftmax(dim=2)
criterion = nn.CrossEntropyLoss(reduction='none')
criterion_train = nn.CrossEntropyLoss(reduction='sum')

def repackage_hidden(h):
    return tuple(v.detach() for v in h)

def reverse_sequence(src):
    length = list(src.shape)[1]
    idx = torch.linspace(length-1, 0, steps=length).long()
    rev_src = src[:,idx]
    return rev_src

def load_model(model,state_dict_path):
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)

def ints_to_sentences(list_of_phrases,EN):
    sentences = []
    for phrase in list_of_phrases:
        sentences.append(" ".join(["|".join([EN.vocab.itos[w] for w in phrase])]))
    return sentences

def escape(l):
    return l.replace("\"", "<quote>").replace(",", "<comma>")

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

# encoder
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
    
# decoder
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
    
# encoder-decoder, training loop
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

# encoder-decoder, validation loop
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
            x = 'reached end'
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

# encoder-decoder with attention
class attn_RNNet_batched(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, weight_init=0.05,weight_tying=False,german_weights = None):
        super(attn_RNNet_batched, self).__init__()
        self.emb = torch.nn.Sequential(torch.nn.Embedding(input_size, hidden_size), torch.nn.Dropout(dropout))
        self.rnn = torch.nn.LSTM(input_size=2*hidden_size, hidden_size=hidden_size, num_layers=num_layers, bias=True, batch_first=True, dropout=dropout)
        self.lnr = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(2*hidden_size, input_size))
    
        for f in self.parameters():
            torch.nn.init.uniform_(f, a=-weight_init, b=weight_init)
        if weight_tying == True:
            attn_context2trg.lnr[1].weight.data[:,:hidden_size] = self.emb[0].weight.data
            
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
    
# encoder-decoder with attention, training loop
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
                l = criterion(decoder_output, trg[:,j+1])
                mask = trg[:,j+1]>=2
                loss += torch.sum(l[mask.squeeze()])
                if np.mod(ix,100) == 0:
                    sentence.extend([torch.argmax(decoder_output[0,:],dim=0)])
                
            loss.backward()
            seq2context_optimizer.step()
            attn_context2trg_optimizer.step()
        
            if np.mod(ix,500) == 0:
                print('Epoch: {}, Batch: {}, Loss: {}'.format(e, ix, loss.cpu().detach()/BATCH_SIZE))
                #print([EN.vocab.itos[i] for i in sentence])
                #print([EN.vocab.itos[i] for i in trg[0,:]])
                
# encoder-decoder with attention, validation loop                
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
                mask = trg[:,j+1]>=2
                total_words += mask.sum()
                track_loss = torch.sum(loss[mask.squeeze()])
                total_loss += track_loss.detach()
            
    ppl = torch.exp(total_loss/total_words)
    scheduler_c2t.step(ppl)
    scheduler_s2c.step(ppl)
    print('Epoch: {}, Validation PPL: {}, Validation Loss: {}'.format(e,ppl,total_loss))
    return total_loss
        
# beam search implementation     
def beam_search(src, attn_seq2context, attn_context2trg, BEAM_WIDTH = 2, BATCH_SIZE=32, max_len=3,context_size=500,EN=None):
    top_p = {}
    top_s = {}
    items = []
    for i in range(BATCH_SIZE):
        top_p[i] = []
        top_s[i] = []
        items.append(i)

    encoder_outputs, encoder_hidden = attn_seq2context(src)
    decoder_context = torch.zeros(BATCH_SIZE, context_size, device='cuda') # 32 x 500
    decoder_hidden = encoder_hidden
    word_input = (torch.zeros(BATCH_SIZE, device='cuda') + EN.vocab.stoi['<s>']).long()
    decoder_output, decoder_context, decoder_hidden, decoder_attention = attn_context2trg(word_input, decoder_context, decoder_hidden, encoder_outputs)
    
    next_words = torch.argsort(lsm2(decoder_output), dim=1, descending=True)[:,0:BEAM_WIDTH].detach()
    p_words_init = torch.stack([torch.index_select(decoder_output[i,:],-1,next_words[i,:]) for i in range(BATCH_SIZE)]).detach()
    p_words_running = torch.stack([p_words_init[:,b].repeat(1,BEAM_WIDTH) for b in range(BEAM_WIDTH)]).view(BEAM_WIDTH**2,BATCH_SIZE).transpose(0,1)

    update = []
    for ix,p in enumerate(next_words):
        update.append([torch.stack(([torch.tensor(EN.vocab.stoi['<s>'], device='cuda')])+([next_words[ix,b]])) for b in range(BEAM_WIDTH)])

    next_words = torch.argsort(lsm2(decoder_output),dim=1, descending=True)[:,0:BEAM_WIDTH].detach()
    p_words_init = torch.stack([torch.index_select(decoder_output[i,:],-1,next_words[i,:]) for i in range(BATCH_SIZE)]).detach()
    p_words_running = torch.stack([p_words_init[:,b].repeat(1,BEAM_WIDTH) for b in range(BEAM_WIDTH)]).view(BEAM_WIDTH**2,BATCH_SIZE).transpose(0,1)
    update = []
    for ix,p in enumerate(next_words):
        update.append([torch.stack(([torch.tensor(EN.vocab.stoi['<s>'], device='cuda')])+([next_words[ix,b]])) for b in range(BEAM_WIDTH)])
    top_s.update(dict(zip(items, update)))
    top_p.update(dict(zip(items, p_words_init)))

    next_words = next_words.transpose(0,1).flatten().long()
    encoder_outputs = encoder_outputs.repeat(BEAM_WIDTH,1,1)
    decoder_hidden = tuple([h.repeat(1,BEAM_WIDTH,1) for h in decoder_hidden])
    decoder_context = decoder_context.repeat(BEAM_WIDTH,1)
    mask = torch.zeros(BATCH_SIZE,BEAM_WIDTH,dtype=torch.uint8).cuda()
    for j in range(max_len-1):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = attn_context2trg(next_words, decoder_context, decoder_hidden, encoder_outputs)
            
            args = torch.argsort(lsm2(decoder_output),dim=1, descending=True)[:,0:BEAM_WIDTH].detach()
            next_words = torch.cat([args[BATCH_SIZE*(b):BATCH_SIZE*(b+1),:] for b in range(BEAM_WIDTH)],dim=1).detach()
            p_words = torch.stack([torch.index_select(decoder_output[i,:],-1,next_words[i,:]) for i in range(BATCH_SIZE)])
            p_words_running += p_words.detach()
            p_words_norm = p_words_running/(j+1)

            word_selector = torch.argsort(p_words_norm,dim=1,descending=True)[:,:BEAM_WIDTH]        
            beam_indicator = (word_selector/BEAM_WIDTH).float().long() #word_selector>=2
            
            prev_words = list(top_s.values())
            words = [torch.stack([prev_words[s][i] for i in beam_indicator[s,:]]) for s in range(BATCH_SIZE)]
            update = []
            for ix,p in enumerate(words):
                update.append([torch.cat((p[b],next_words[ix,b].unsqueeze(0))) for b in range(BEAM_WIDTH)])
            top_s.update(dict(zip(items, update)))
            mask += torch.stack([next_words[s,:].index_select(0,word_selector[s,:]) for s in range(BATCH_SIZE)]) == 3
            update_p = torch.stack([torch.index_select(p_words_running[b], 0, word_selector[b]) for b in range(BATCH_SIZE)])
            top_p.update(dict(zip(items, update_p)))
            p_words_running = torch.stack([update_p[:,b].repeat(1,BEAM_WIDTH) for b in range(BEAM_WIDTH)]).view(BEAM_WIDTH**2,BATCH_SIZE).transpose(0,1)

            indexs = torch.zeros(BATCH_SIZE,BEAM_WIDTH,device='cuda')
            for i in range(BATCH_SIZE):
                indexs[i,:] += i+(BATCH_SIZE*beam_indicator[i,:].float())
            indexs = indexs.long()
            indexs = indexs.transpose(0,1).flatten()
            decoder_hidden = tuple([torch.index_select(h,1,indexs) for h in decoder_hidden])
            decoder_context = torch.index_select(decoder_context,0,indexs)
            next_words = torch.stack([torch.index_select(next_words[s,:],0,word_selector[s,:]) for s in range(BATCH_SIZE)]).transpose(0,1).flatten().long()
    return top_s

# alternate beam search implementation
def beam_search_first3(encoder_outputs, encoder_hidden, attn_seq2context, attn_context2trg, BEAM_WIDTH = 2, BATCH_SIZE=32, max_len=3,context_size=500,EN=None):
    top_p = {}
    top_s = {}
    stopped = torch.zeros((BATCH_SIZE,BEAM_WIDTH), device='cuda') #==1
    items = []
    for i in range(BATCH_SIZE):
        top_p[i] = []
        top_s[i] = []
        items.append(i)

    decoder_context = torch.zeros(BATCH_SIZE, context_size, device='cuda') # 32 x 500
    decoder_hidden = encoder_hidden
    outputs = []
    word_input = (torch.zeros(BATCH_SIZE, device='cuda') + EN.vocab.stoi['<s>']).long()
    decoder_output, decoder_context, decoder_hidden, decoder_attention = attn_context2trg(word_input, decoder_context, decoder_hidden, encoder_outputs)
    outputs.append(decoder_output)
    next_words = torch.argsort(lsm2(decoder_output),dim=1, descending=True)[:,0:BEAM_WIDTH].detach()
    p_words_init = torch.stack([torch.index_select(decoder_output[i,:],-1,next_words[i,:]) for i in range(BATCH_SIZE)]).detach()
    p_words_running = torch.stack([p_words_init[:,b].repeat(1,BEAM_WIDTH) for b in range(BEAM_WIDTH)]).view(BEAM_WIDTH**2,BATCH_SIZE).transpose(0,1)

    update = []
    for ix,p in enumerate(next_words):
        update.append([torch.stack(([torch.tensor(EN.vocab.stoi['<s>'], device='cuda')])+([next_words[ix,b]])) for b in range(BEAM_WIDTH)])

    next_words = torch.argsort(lsm2(decoder_output),dim=1, descending=True)[:,0:BEAM_WIDTH].detach()
    p_words_init = torch.stack([torch.index_select(decoder_output[i,:],-1,next_words[i,:]) for i in range(BATCH_SIZE)]).detach()
    p_words_running = torch.stack([p_words_init[:,b].repeat(1,BEAM_WIDTH) for b in range(BEAM_WIDTH)]).view(BEAM_WIDTH**2,BATCH_SIZE).transpose(0,1)
    update = []
    for ix,p in enumerate(next_words):
        update.append([torch.stack(([torch.tensor(EN.vocab.stoi['<s>'], device='cuda')])+([next_words[ix,b]])) for b in range(BEAM_WIDTH)])
    top_s.update(dict(zip(items, update)))
    top_p.update(dict(zip(items, p_words_init)))

    next_words = next_words.transpose(0,1).flatten().long()
    encoder_outputs = encoder_outputs.repeat(BEAM_WIDTH,1,1)
    decoder_hidden = tuple([h.repeat(1,BEAM_WIDTH,1) for h in decoder_hidden])
    decoder_context = decoder_context.repeat(BEAM_WIDTH,1)
    
    for j in range(max_len-2):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = attn_context2trg(next_words, decoder_context, decoder_hidden, encoder_outputs)
            args = torch.argsort(lsm2(decoder_output),dim=1, descending=True)[:,0:BEAM_WIDTH].detach()
            next_words = torch.cat([args[BATCH_SIZE*(b):BATCH_SIZE*(b+1),:] for b in range(BEAM_WIDTH)],dim=1).detach()
            p_words = torch.stack([torch.index_select(decoder_output[i,:],-1,next_words[i,:]) for i in range(BATCH_SIZE)])
            p_words_running += p_words.detach()
            p_words_norm = p_words_running/(j+1)

            word_selector = torch.argsort(p_words_norm,dim=1,descending=True)[:,:BEAM_WIDTH]        
            beam_indicator = (word_selector/BEAM_WIDTH).float().long() #word_selector>=2

            prev_words = list(top_s.values())
            words = [torch.stack([prev_words[s][i] for i in beam_indicator[s,:]]) for s in range(BATCH_SIZE)]
            #update = []
            #for ix,p in enumerate(words):
            #    update.append([torch.cat((p[b],next_words[ix,b].unsqueeze(0))) for b in range(BEAM_WIDTH)])
            #top_s.update(dict(zip(items, update)))

            #update_p = torch.stack([torch.index_select(p_words_running[b], 0, word_selector[b]) for b in range(BATCH_SIZE)])
            #top_p.update(dict(zip(items, update_p)))

            indexs = torch.zeros(BATCH_SIZE,BEAM_WIDTH,device='cuda')
            for i in range(BATCH_SIZE):
                indexs[i,:] += i+(BATCH_SIZE*beam_indicator[i,:].float())
            indexs = indexs.long()
            indexs = indexs.transpose(0,1).flatten()
            decoder_hidden = tuple([torch.index_select(h,1,indexs) for h in decoder_hidden])
            decoder_context = torch.index_select(decoder_context,0,indexs)
            decoder_output = torch.index_select(decoder_output,0,indexs)
            outputs.append(decoder_output)
            next_words = torch.stack([torch.index_select(next_words[s,:],0,word_selector[s,:]) for s in range(BATCH_SIZE)]).transpose(0,1).flatten().long()
            
    return outputs

# encoder-decoder with attention and split-train
def attn_training_split_loop(e,train_iter,seq2context,attn_context2trg,seq2context_optimizer,attn_context2trg_optimizer,BATCH_SIZE=32,context_size=500,EN=None):
    seq2context.train()
    attn_context2trg.train()
    for ix,batch in enumerate(train_iter):
        src = batch.src.values.transpose(0,1)
        src = reverse_sequence(src)
        trg = batch.trg.values.transpose(0,1)
        decoder_context = torch.zeros(BATCH_SIZE, context_size, device='cuda') # 32 x 500
        encoder_outputs, encoder_hidden = seq2context(src)    
        decoder_hidden = encoder_hidden
        if trg.shape[0] == BATCH_SIZE:
        
            seq2context_optimizer.zero_grad()
            attn_context2trg_optimizer.zero_grad()
            loss = 0
           
            sentence = []
            p = np.random.rand()
            if p > 0.5:
                for j in range(0,trg.shape[1]-2):
                    word_input = trg[:,j]
                    decoder_output, decoder_context, decoder_hidden, decoder_attention = attn_context2trg(word_input, decoder_context, decoder_hidden, encoder_outputs)
                    #print(decoder_output.shape, trg[i,j+1].view(-1).shape)
                    l = criterion(decoder_output, trg[:,j+1])
                    mask = trg[:,j+1]!=1
                    loss += l[mask.squeeze()].sum()
                
                    if np.mod(ix,100) == 0:
                        sentence.extend([torch.argmax(decoder_output[0,:],dim=0)])
            else:
                top_s =  beam_search(src, seq2context, attn_context2trg, BEAM_WIDTH = 5, BATCH_SIZE=BATCH_SIZE, max_len=trg.shape[1],context_size=500,EN=EN)
                predictions = torch.stack([top_s[i][0] for i in range(BATCH_SIZE)])
                for j in range(predictions.shape[1] - 2):
                    word_input = predictions[:,j]
                    decoder_output, decoder_context, decoder_hidden, decoder_attention = attn_context2trg(word_input, decoder_context, decoder_hidden, encoder_outputs)
                    #print(decoder_output.shape, trg[i,j+1].view(-1).shape)
                    l = criterion(decoder_output, trg[:,j+1])
                    mask = trg[:,j+1]!=1
                    loss += l[mask.squeeze()].sum()
            loss.backward()
            seq2context_optimizer.step()
            attn_context2trg_optimizer.step()
        
            if np.mod(ix,500) == 0:
                print('Epoch: {}, Batch: {}, Loss: {}'.format(e, ix, loss.cpu().detach()/BATCH_SIZE))
                #print([EN.vocab.itos[i] for i in sentence])
                #print([EN.vocab.itos[i] for i in trg[0,:]])
                
# encoder-decoder with attention and split-train, top 3
def attn_training_split_loop_top_3(e,train_iter,seq2context,attn_context2trg,seq2context_optimizer,attn_context2trg_optimizer,BATCH_SIZE=32,context_size=500,EN=None):
    seq2context.train()
    attn_context2trg.train()
    for ix,batch in enumerate(train_iter):
        src = batch.src.values.transpose(0,1)
        src = reverse_sequence(src)
        trg = batch.trg.values.transpose(0,1)
        decoder_context = torch.zeros(BATCH_SIZE, context_size, device='cuda') # 32 x 500
        encoder_outputs, encoder_hidden = seq2context(src)    
        decoder_hidden = encoder_hidden
        if trg.shape[0] == BATCH_SIZE:
        
            seq2context_optimizer.zero_grad()
            attn_context2trg_optimizer.zero_grad()
            loss = 0
           
            sentence = []
            p = np.random.rand()
            if p > 1.0:
                for j in range(0,trg.shape[1]-2):
                    word_input = trg[:,j]
                    decoder_output, decoder_context, decoder_hidden, decoder_attention = attn_context2trg(word_input, decoder_context, decoder_hidden, encoder_outputs)
                    #print(decoder_output.shape, trg[i,j+1].view(-1).shape)
                    l = criterion(decoder_output, trg[:,j+1])
                    mask = trg[:,j+1]!=1
                    loss += l[mask.squeeze()].sum()
                
                    if np.mod(ix,100) == 0:
                        sentence.extend([torch.argmax(decoder_output[0,:],dim=0)])
            else:
                outputs = beam_search_first3(encoder_outputs, encoder_hidden, seq2context, attn_context2trg, BEAM_WIDTH = 1, BATCH_SIZE=BATCH_SIZE, max_len=trg.shape[1],EN=EN,context_size=context_size)
                preds = torch.stack([i[:BATCH_SIZE,:] for i in outputs]).transpose(0,1).transpose(1,2)
                l = criterion(preds, trg[:,1:])
                mask = trg[:,1:]!=1
                loss += l[mask.squeeze()].sum()
            loss.backward()
            seq2context_optimizer.step()
            attn_context2trg_optimizer.step()
        
            if np.mod(ix,100) == 0:
                print('Epoch: {}, Batch: {}, Loss: {}'.format(e, ix, loss.cpu().detach()/BATCH_SIZE))
                #print([EN.vocab.itos[i] for i in sentence])
                #print([EN.vocab.itos[i] for i in trg[0,:]])