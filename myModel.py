import torch
import torch.nn as nn
import torch.nn.functional as F
from myUtils import cp
import logging
import numpy as np
from torch.optim import Adam
from itertools import chain

class Encoder(nn.Module):
    
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.device = opt.device
        self.batch_size = opt.batch_size
        self.vocab_size = opt.vocab_size
        self.embed_size = opt.emb_size
        self.hidden_size = opt.hidden_size        
        self.num_layers = opt.num_layers    
        self.drop_rate = opt.drop_rate
        self.bidirectional = opt.bidirectional  
        self.num_directions = 2 if opt.bidirectional else 1
        self.embeddings = torch.nn.Embedding(self.vocab_size,self.embed_size)
        self.rnn = torch.nn.GRU(input_size=self.embed_size, hidden_size=self.hidden_size,
                                num_layers=self.num_layers, batch_first=True,
                                bidirectional=self.bidirectional, dropout=self.drop_rate)  
        
    def init_hidden(self):
        return torch.randn(self.num_directions,self.batch_size,self.hidden_size).to(self.device)
        
    def forward(self, x):#topic_represent [batch_size, topic_num]
        embeddings = self.embeddings(x)  #[batch, src_len, embed_size]
        hidden_state = self.init_hidden()
        #cp(embeddings.device, 'embeddings.device')
        #cp(hidden_state.device, 'hidden_state.device')
        memory_bank, encoder_final_state = self.rnn(embeddings, hidden_state)
        return memory_bank


class TAMTAttention(nn.Module):

    def __init__(self, opt):
        super(TAMTAttention, self).__init__()
        self.tag_num = opt.tag_num
        self.hidden_size = opt.hidden_size 
        self.bidirectional = opt.bidirectional  
        self.num_directions = 2 if opt.bidirectional else 1
        self.topic_num = opt.topic_num  
        self.batch_size = opt.batch_size
        
        self.W1 = torch.nn.Linear(self.hidden_size*self.num_directions, self.hidden_size*self.num_directions)
        self.W2= torch.nn.Linear(self.topic_num, self.hidden_size*self.num_directions)
        self.attention = nn.Linear(self.hidden_size*self.num_directions, self.tag_num, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, memory_bank, topic_represent):
        seq_len = memory_bank.shape[1]
        topic_represent_expand = topic_represent.unsqueeze(1).expand(self.batch_size, seq_len, self.topic_num) 
        u = torch.tanh(self.W1(memory_bank)+ self.W2(topic_represent_expand))
        attention = self.attention(u)# N, seq_len, labels_num
        weight = F.softmax(attention, -2)# # N, seq_len, labels_num
        tag_embedding = weight.transpose(1, 2) @ memory_bank # N, labels_num, hidden_size
        return tag_embedding    

class Ranker(nn.Module):

    def __init__(self, opt):
        super(Ranker, self).__init__()
        self.hidden_size = opt.hidden_size 
        self.bidirectional = opt.bidirectional  
        self.num_directions = 2 if opt.bidirectional else 1
        self.linear_size = [opt.hidden_size*self.num_directions] + opt.linear_size 
        self.output_size = 1
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s) for in_s, out_s in zip(self.linear_size[:-1], self.linear_size[1:]))    
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(self.linear_size[-1], self.output_size)
        nn.init.xavier_uniform_(self.output.weight)          

    def forward(self, tag_embedding):
        linear_out = tag_embedding
        for linear in self.linear:# N, labels_num, linear1
            linear_out = F.relu(linear(linear_out))# N, labels_num, linear2
        logit = torch.squeeze(self.output(linear_out), -1) # N, labels_num, 1
        return logit   
    
class myModel(nn.Module):

    def __init__(self, opt):
        super(myModel, self).__init__()       
        self.encoder = Encoder(opt)
        self.TAMTAttention = TAMTAttention(opt)
        self.ranker = Ranker(opt) 
        self.attn_mode = opt.attn_mode 
        self.hidden_size = opt.hidden_size 
        self.bidirectional = opt.bidirectional  
        self.num_directions = 2 if opt.bidirectional else 1
        self.output_layer = torch.nn.Linear(self.hidden_size*self.num_directions, opt.tag_num)
              
    def forward(self, x, topic_represent):#topic_represent [batch_size, topic_num]
        memory_bank = self.encoder(x)  

        if self.attn_mode=='TAMTA':
            tag_embedding = self.TAMTAttention(memory_bank, topic_represent)    
            logit = self.ranker(tag_embedding)    
            return logit  
        
        if self.attn_mode=='SE':  
            logit = self.output_layer(memory_bank[:,-1,:])    
            return logit    
               
        
class NTG(nn.Module):
    def __init__(self, opt, hidden_dim=500, l1_strength=0.001):
        super(NTG, self).__init__()
        
        self.input_dim = opt.bow_vocab
        self.topic_num = opt.topic_num
        topic_num = opt.topic_num        
        self.fc11 = nn.Linear(self.input_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, topic_num)
        self.fc22 = nn.Linear(hidden_dim, topic_num)
        self.fcs = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.fcg1 = nn.Linear(topic_num, topic_num)
        self.fcg2 = nn.Linear(topic_num, topic_num)
        self.fcg3 = nn.Linear(topic_num, topic_num)
        self.fcg4 = nn.Linear(topic_num, topic_num)
        self.fcd1 = nn.Linear(topic_num, self.input_dim)
        self.l1_strength = torch.FloatTensor([l1_strength]).to(opt.device)

    def encode(self, x):
        e1 = F.relu(self.fc11(x))
        e1 = F.relu(self.fc12(e1))
        e1 = e1.add(self.fcs(x))
        return self.fc21(e1), self.fc22(e1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
            
    def generate(self, h):
        g1 = torch.tanh(self.fcg1(h))
        g1 = torch.tanh(self.fcg2(g1))
        g1 = torch.tanh(self.fcg3(g1))
        g1 = torch.tanh(self.fcg4(g1))
        g1 = g1.add(h)
        return g1
        
    def decode(self, z):
        d1 = F.softmax(self.fcd1(z), dim=1)
        return d1

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        g = self.generate(z)
        return z, g, self.decode(g), mu, logvar

    def print_topic_words(self, vocab_dic, fn, ifPrint=False,n_top_words=10):
        beta_exp = self.fcd1.weight.data.cpu().numpy().T
        if ifPrint:
            print("Writing to %s" % fn)
        fw = open(fn, 'w')
        for k, beta_k in enumerate(beta_exp):
            topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
            if ifPrint:
                print('Topic {}: {}'.format(k, ' '.join(topic_words)))
            fw.write('{}\n'.format(' '.join(topic_words)))
        fw.close()

def init_optimizers(model, ntg_model, opt):
    optimizer_encoder = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    optimizer_ntg = Adam(params=filter(lambda p: p.requires_grad, ntg_model.parameters()), lr=opt.ntg_learning_rate)
    whole_params = chain(model.parameters(), ntg_model.parameters())
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.learning_rate)
    return optimizer_encoder, optimizer_ntg, optimizer_whole 

def myLoss(y_pred, y_true, opt):
    criteria = nn.BCEWithLogitsLoss() 
    loss = criteria(y_pred, y_true)
    return loss
       
        
def myModelStat(model):
    print('===========================Model Para==================================')
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())     
    print('===========================Model Para==================================')

