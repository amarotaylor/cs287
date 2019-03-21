import random
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchtext
from torchtext.vocab import Vectors, GloVe
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

def prepend_null(sent,null_tkn = torch.tensor(56690,dtype=torch.long,device='cuda')):
    null_tkns = null_tkn.repeat(sent.shape[0], 1)
    return torch.cat((null_tkns, sent), 1)

class FeedForward_layer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(FeedForward_layer, self).__init__()
        self.d = nn.Dropout(dropout)
        self.m = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        for param in self.parameters():
            torch.nn.init.normal_(param, mean=0, std=0.01)
    def forward(self, inputs):
        hidden = self.m(self.linear1(self.d(inputs)))
        output = self.m(self.linear2(self.d(hidden)))
        return output
    
class EmbedProject(torch.nn.Module):
    def __init__(self, weights, embed_size, project_size):
        super(EmbedProject, self).__init__()
        self.embed = nn.Embedding.from_pretrained(weights, freeze=True) # weights: input_size x embed_size
        self.linear = nn.Linear(embed_size, project_size)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.01)
    def forward(self, inputs):
        embedding = self.embed(inputs)
        output = self.linear(embedding)
        return output    
    
class Decomposable_Attn_Network(torch.nn.Module):
    def __init__(self,input_size,embed_size,hidden_size,output_size,weights,dropout=0.2,n_layers = 3):
        super(Decomposable_Attn_Network, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.Embedding_layer = EmbedProject(weights,self.embed_size,self.hidden_size)
        self.F = FeedForward_layer(self.hidden_size,self.hidden_size,self.hidden_size)
        self.G = FeedForward_layer(self.hidden_size*2,self.hidden_size,self.hidden_size)
        self.H = FeedForward_layer(self.hidden_size*2,self.hidden_size,self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        torch.nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        
    def attention_and_mask(self,hook1,hook2,sent1,sent2,pad_tkn = 1):
        score1 = torch.bmm(hook1,hook2.transpose(1,2))
        score2 = torch.bmm(hook2,hook1.transpose(1,2))
        mask1 = (sent1 == pad_tkn)
        mask2 = (sent2 == pad_tkn)
        mask1a = mask1.unsqueeze(1).expand(-1, sent2.shape[1], -1).float()
        mask2a = mask2.unsqueeze(1).expand(-1, sent1.shape[1], -1).float()
        score1 = score1 * (1 - mask2a) + (mask2a * - 1e32)
        score2 = score2 * (1 - mask1a) + (mask1a * - 1e32)
        prob1 = F.softmax(score1, dim=2)
        prob2 = F.softmax(score2, dim=2)
        return prob1,prob2,mask1,mask2
    
    def forward(self,sent1, sent2,pad_tkn=1):
        # embed words
        proj1 = self.Embedding_layer(sent1)
        proj2 = self.Embedding_layer(sent2)
        # feedforward layer 1
        f1 = self.F(proj1)
        f2 = self.F(proj2)
        prob1,prob2,mask1,mask2 = self.attention_and_mask(f1,f2,sent1,sent2)
        # apply attention
        proj1_soft = torch.bmm(prob2, proj1)
        proj2_soft = torch.bmm(prob1, proj2)
        # combine feature vectors
        proj1_combined = torch.cat((proj1, proj2_soft), dim=2)
        proj2_combined = torch.cat((proj2, proj1_soft), dim=2)
        # next feedforward layer
        g1 = self.G(proj1_combined)
        g2 = self.G(proj2_combined)
        # mask <pad> tokens
        mask1b = mask1.unsqueeze(2).expand(-1, -1, self.hidden_size).float()
        mask2b = mask2.unsqueeze(2).expand(-1, -1, self.hidden_size).float()
        g1 = g1 * (1 - mask1b)
        g2 = g2 * (1 - mask2b)
        # sum pool
        g1_sum = g1.sum(dim=1)
        g2_sum = g2.sum(dim=1)
        # concatenate and final feed forward
        g_all = torch.cat((g1_sum, g2_sum), dim=1)
        h_all = self.H(g_all)
        output = self.linear(h_all)
        return output
        