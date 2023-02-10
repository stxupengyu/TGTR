import numpy as np
import torch.utils.data as data_utils
import torch
from transformers import BertTokenizer
import random
from myUtils import fp, cp, time_since
import pickle
import time
import logging
import gensim
import re
from nltk.tokenize import word_tokenize
from myDataset import txt2list, dataSplit, build_vocab, BowFeature, encode_one_hot, padding, make_bow_dictionary
from torch import nn

def dataProcess(opt):
    tokenized_pairs = txt2list(opt.txt_path, opt.split_token, opt.dataset_size)#cost time       
    train, valid, test = dataSplit(tokenized_pairs, opt.seed)
    word2idx, idx2word, token_freq_counter, tag2idx, idx2tag = build_vocab(train, opt.vocab_size)
    opt.tag_num = len(tag2idx)
    print("Building bow dictionary from training data")
    bow_dictionary = make_bow_dictionary(train, opt.bow_vocab)
    print("Bow dict_size: %d after filtered" % len(bow_dictionary))

    tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
    X_train, y_train, X_bow_train = build_bert_dataset(tokenizer, train, word2idx, tag2idx, bow_dictionary, opt.vocab_size, opt.max_src_len)
    X_valid, y_valid, X_bow_valid = build_bert_dataset(tokenizer, valid, word2idx, tag2idx, bow_dictionary, opt.vocab_size, opt.max_src_len)
    X_test, y_test, X_bow_test = build_bert_dataset(tokenizer, test, word2idx, tag2idx, bow_dictionary, opt.vocab_size, opt.max_src_len)         

    #BoW feature
    train_bow_data = data_utils.TensorDataset(torch.from_numpy(X_bow_train).type(torch.float32))
    val_bow_data = data_utils.TensorDataset(torch.from_numpy(X_bow_valid).type(torch.float32))
    test_bow_data = data_utils.TensorDataset(torch.from_numpy(X_bow_test).type(torch.float32))
    train_bow_loader = data_utils.DataLoader(train_bow_data, opt.batch_size, shuffle=True, drop_last=True)
    valid_bow_loader = data_utils.DataLoader(val_bow_data, opt.batch_size, shuffle=True, drop_last=True)
    test_bow_loader = data_utils.DataLoader(test_bow_data,opt.batch_size, drop_last=True)
    
    train_data = Dataset(X_bow_train,X_train,y_train)
    val_data = Dataset(X_bow_valid,X_valid,y_valid)                                          
    test_data = Dataset(X_bow_test, X_test,y_test)        
            
    train_loader = data_utils.DataLoader(train_data, opt.bert_batch_size, shuffle=True, drop_last=True)
    val_loader = data_utils.DataLoader(val_data, opt.bert_batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, opt.bert_batch_size, drop_last=True)
  
    print("======================Data Load Done======================")
    
    return train_loader, val_loader, test_loader, train_bow_loader, \
    valid_bow_loader, test_bow_loader, bow_dictionary, opt
  

def build_bert_dataset(tokenizer, src_trgs_pairs, word2idx, tag2idx, bow_dictionary, vocab_size, max_seq_len):
    '''
    build train/valid/test dataset
    '''
    text = []
    label = []
    bow = [] 
    for idx, (source, targets) in enumerate(src_trgs_pairs):
        src = ' '.join(source)
        src = tokenizer(src, padding='max_length', max_length = 512, truncation=True,
                            return_tensors="pt")
        trg = [tag2idx[w] for w in targets if w in tag2idx]
        src_bow = bow_dictionary.doc2bow(source)
        text.append(src)
        label.append(trg)
        bow.append(src_bow)    
    bow = BowFeature(bow, bow_dictionary)
    label =  [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in label] 
    return text, np.array(label), np.array(bow)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, bow, texts, labels):
        self.labels = labels
        self.texts = texts
        self.bow = torch.from_numpy(bow).type(torch.float32)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        labels = self.labels[idx]     
        return labels

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    
    def get_batch_bow(self, idx):
        # Fetch a batch of inputs
        return self.bow[idx]

    def __getitem__(self, idx):
        batch_bow = self.get_batch_bow(idx)
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_bow, batch_texts, batch_y
        