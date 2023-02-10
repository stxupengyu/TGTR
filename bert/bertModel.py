import torch
import torch.nn as nn
import torch.nn.functional as F
from myUtils import cp
import logging
import numpy as np
from torch.optim import Adam
from itertools import chain
from transformers import BertModel
from transformers import AdamW

class BERTEncoder(nn.Module):
    
    def __init__(self, opt):
        super(BERTEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(opt.bert_path)
        self.dropout = nn.Dropout(opt.bert_dropout)  
        
    def forward(self, input_id, mask):
        memory_bank, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        return memory_bank


class TAMTAttention(nn.Module):

    def __init__(self, opt):
        super(TAMTAttention, self).__init__()
        self.tag_num = opt.tag_num
        self.hidden_size = opt.hidden_size 
        self.bidirectional = opt.bidirectional  
        self.num_directions = 2 if opt.bidirectional else 1
        self.topic_num = opt.topic_num  
        
        self.W1 = torch.nn.Linear(768, 768)
        self.W2= torch.nn.Linear(self.topic_num, 768)
        self.attention = nn.Linear(768, self.tag_num, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, memory_bank, topic_represent):
        seq_len = memory_bank.shape[1]
        batch_size = memory_bank.shape[0]
        topic_represent_expand = topic_represent.unsqueeze(1).expand(batch_size, seq_len, self.topic_num) 
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
        self.output_size = 1
        
        self.linear_size = [768] + opt.linear_size 
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
    
class BERTModel(nn.Module):

    def __init__(self, opt):
        super(BERTModel, self).__init__()       
        self.BERTEncoder = BERTEncoder(opt) 
        self.TAMTAttention = TAMTAttention(opt)
        self.ranker = Ranker(opt) 
        self.hidden_size = opt.hidden_size 
        self.bidirectional = opt.bidirectional  
        self.num_directions = 2 if opt.bidirectional else 1
        self.output_layer = torch.nn.Linear(self.hidden_size*self.num_directions, opt.tag_num)
            
    def forward(self, input_id, mask, topic_represent):#topic_represent [batch_size, topic_num]
        memory_bank = self.BERTEncoder(input_id, mask) 
        tag_embedding = self.TAMTAttention(memory_bank, topic_represent)    
        logit = self.ranker(tag_embedding)    
        return logit          

def init_optimizers(model, ntg_model, opt):
    optimizer_encoder = AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.bert_learning_rate, eps=opt.bert_adam_epsilon)
    optimizer_ntg = Adam(params=filter(lambda p: p.requires_grad, ntg_model.parameters()), lr=opt.ntg_learning_rate)
    whole_params = chain(model.parameters(), ntg_model.parameters())
    optimizer_whole = AdamW(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.bert_learning_rate, eps=opt.bert_adam_epsilon)   
    return optimizer_encoder, optimizer_ntg, optimizer_whole 

def myLoss(y_pred, y_true, opt):
    criteria = nn.BCEWithLogitsLoss() 
    loss = criteria(y_pred, y_true)
    return loss
