import numpy as np
import torch.utils.data as data_utils
import torch
import random
from myUtils import fp, cp, time_since
import pickle
import time
import logging
import gensim
import re
from nltk.tokenize import word_tokenize
import random
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import re
from nltk.tokenize import word_tokenize
import os
import gensim

def dataProcess(opt):
    tokenized_pairs = txt2list(opt.txt_path, opt.split_token, opt.dataset_size)#cost time      
    train, valid, test = dataSplit(tokenized_pairs, opt.seed)
    word2idx, idx2word, token_freq_counter, tag2idx, idx2tag = build_vocab(train, opt.vocab_size)
    print("Building bow dictionary from training data")
    bow_dictionary = make_bow_dictionary(train, opt.bow_vocab)
    print("Bow dict_size: %d after filtered" % len(bow_dictionary))
    X_train, y_train, X_bow_train = build_dataset(train, word2idx, tag2idx, bow_dictionary, opt.vocab_size, opt.max_src_len)
    X_valid, y_valid, X_bow_valid = build_dataset(valid, word2idx, tag2idx, bow_dictionary, opt.vocab_size, opt.max_src_len)
    X_test, y_test, X_bow_test = build_dataset(test, word2idx, tag2idx, bow_dictionary, opt.vocab_size, opt.max_src_len)
    opt.tag_num = len(tag2idx)  
    
    #BoW feature
    train_bow_data = data_utils.TensorDataset(torch.from_numpy(X_bow_train).type(torch.float32))
    val_bow_data = data_utils.TensorDataset(torch.from_numpy(X_bow_valid).type(torch.float32))
    test_bow_data = data_utils.TensorDataset(torch.from_numpy(X_bow_test).type(torch.float32))
    
    train_bow_loader = data_utils.DataLoader(train_bow_data, opt.batch_size, shuffle=True, drop_last=True)
    valid_bow_loader = data_utils.DataLoader(val_bow_data, opt.batch_size, shuffle=True, drop_last=True)
    test_bow_loader = data_utils.DataLoader(test_bow_data,opt.batch_size, drop_last=True)
    
    #Nomral feature and label
    train_data = data_utils.TensorDataset(torch.from_numpy(X_bow_train).type(torch.float32),
                                          torch.from_numpy(X_train).type(torch.LongTensor),
                                          torch.from_numpy(y_train).type(torch.LongTensor))
    val_data = data_utils.TensorDataset(torch.from_numpy(X_bow_valid).type(torch.float32),
                                        torch.from_numpy(X_valid).type(torch.LongTensor),
                                          torch.from_numpy(y_valid).type(torch.LongTensor))                                          
    test_data = data_utils.TensorDataset(torch.from_numpy(X_bow_test).type(torch.float32),
                                         torch.from_numpy(X_test).type(torch.LongTensor),
                                         torch.from_numpy(y_test).type(torch.LongTensor))
    
    train_loader = data_utils.DataLoader(train_data, opt.batch_size, shuffle=True, drop_last=True)
    val_loader = data_utils.DataLoader(val_data, opt.batch_size, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, opt.batch_size, drop_last=True)
  
    print("======================Data Load Done======================")
    
    return train_loader, val_loader, test_loader, train_bow_loader, \
    valid_bow_loader, test_bow_loader, bow_dictionary, opt
    


def txt2list(txt_path, split_token, dataset_size):
    tokenized_src = []
    tokenized_trg = []
    max_src_len = 0
    max_trg_len = 0
    f=open(txt_path, 'r')
    for line in f.readlines():
        # process src and trg line 
        lineVec = line.strip().split(split_token)#split by 
        src_line = lineVec[0]
        trg_line = lineVec[1]        
        src_word_list = word_tokenize(src_line)
        trg_word_list = trg_line.strip().split(';') 

        # Truncate the sequence if it is too long
#         src_word_list = src_word_list[:max_src_len]
#         trg_word_list = trg_list[:max_trg_len]
        if len(src_word_list)>max_src_len:
            max_src_len = len(src_word_list)
        if len(trg_word_list)>max_trg_len:
            max_trg_len = len(trg_word_list)

        tokenized_src.append(src_word_list)
        tokenized_trg.append(trg_word_list)

    assert len(tokenized_src) == len(tokenized_trg), \
        'the number of records in source and target are not the same'
    
    print('origainal max_src_len', max_src_len)
    print('origainal max_trg_len', max_trg_len)
    tokenized_pairs = list(zip(tokenized_src, tokenized_trg))
    
    if dataset_size !=1:
        cut = int(len(tokenized_pairs)*dataset_size)
        tokenized_pairs = tokenized_pairs[:cut]
        
    print("Finish reading %d lines" % len(tokenized_src))
    return tokenized_pairs


def dataSplit(tokenized_pairs, random_seed):
    random.seed(random_seed)
    random.shuffle(tokenized_pairs)
    data_length = len(tokenized_pairs)
    train_length = int(data_length*.8)
    valid_length = int(data_length*.9)
    train, valid, test = tokenized_pairs[:train_length], tokenized_pairs[train_length:valid_length],\
                                     tokenized_pairs[valid_length:]  
    return train, valid, test


def build_vocab(tokenized_src_trg_pairs, vocab_size):
    '''
    Build the vocabulary from the training (src, trg) pairs
    :param tokenized_src_trg_pairs: list of (src, trg) pairs
    :return: word2idx, idx2word, token_freq_counter
    '''
    # Build vocabulary from training src and trg
    print("Building vocabulary from training data")
    token_freq_counter = Counter()
    token_freq_counter_tag = Counter()
    for src_word_list, trg_word_lists in tokenized_src_trg_pairs:
        token_freq_counter.update(src_word_list)
        token_freq_counter_tag.update(trg_word_lists)

    # Discard special tokens if already present
    special_tokens = ['<pad>', '<unk>']
    num_special_tokens = len(special_tokens)

    for s_t in special_tokens:
        if s_t in token_freq_counter:
            del token_freq_counter[s_t]

    word2idx = dict()
    idx2word = dict()
    for idx, word in enumerate(special_tokens):
        word2idx[word] = idx
        idx2word[idx] = word

    sorted_word2idx = sorted(token_freq_counter.items(), key=lambda x: x[1], reverse=True)

    sorted_words = [x[0] for x in sorted_word2idx]

    for idx, word in enumerate(sorted_words):
        word2idx[word] = idx + num_special_tokens

    for idx, word in enumerate(sorted_words):
        idx2word[idx + num_special_tokens] = word

    tag2idx = dict()
    idx2tag = dict()

    sorted_tag2idx = sorted(token_freq_counter_tag.items(), key=lambda x: x[1], reverse=True)

    sorted_tags = [x[0] for x in sorted_tag2idx]

    for idx, tag in enumerate(sorted_tags):
        tag2idx[tag] = idx

    for idx, tag in enumerate(sorted_tags):
        idx2tag[idx] = tag       
        
    print("Total vocab_size: %d, predefined vocab_size: %d" % (len(word2idx), vocab_size))
    print("Total tag_size: %d" %len(tag2idx))   
    
    return word2idx, idx2word, token_freq_counter, tag2idx, idx2tag

def make_bow_dictionary(tokenized_src_trg_pairs, bow_vocab):
    '''
    Build bag-of-word dictionary from tokenized_src_trg_pairs
    :param tokenized_src_trg_pairs: a list of (src, trg) pairs
    :param data_dir: data address, for distinguishing Weibo/Twitter/StackExchange
    :param bow_vocab: the size the bow vocabulary
    :return: bow_dictionary, a gensim.corpora.Dictionary object
    '''
    doc_bow = []
    tgt_set = set()
    
    def bowProcess(cur_bow):
        temp = [inst.lower() for inst in cur_bow]
        return temp
    
    for src, tgt in tokenized_src_trg_pairs:
        cur_bow = []
        cur_bow.extend(src)
        cur_bow.extend(tgt)
        cur_bow = bowProcess(cur_bow)
        doc_bow.append(cur_bow)
        
    bow_dictionary = gensim.corpora.Dictionary(doc_bow)
    # Remove single letter or character tokens
    len_1_words = list(filter(lambda w: len(w) <= 2, bow_dictionary.values()))
    bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, len_1_words)))

    def read_stopwords(fn):
        return set([line.strip() for line in open(fn) if len(line.strip()) != 0])

    # Read stopwords from file (bow vocabulary should not contain stopwords)
    STOPWORDS = gensim.parsing.preprocessing.STOPWORDS
    stopwords1 = read_stopwords("stopwords/en.txt")
    stopwords2 = read_stopwords("stopwords/physics.txt")
    final_stopwords = set(STOPWORDS).union(stopwords1).union(stopwords2)

    bow_dictionary.filter_tokens(list(map(bow_dictionary.token2id.get, final_stopwords)))

    print("The original bow vocabulary: %d" % len(bow_dictionary))
    bow_dictionary.filter_extremes(no_below=3, keep_n=bow_vocab)
    bow_dictionary.compactify()
    bow_dictionary.id2token = dict([(id, t) for t, id in bow_dictionary.token2id.items()])
    # for debug
    sorted_dfs = sorted(bow_dictionary.dfs.items(), key=lambda x: x[1], reverse=True)
    sorted_dfs_token = [(bow_dictionary.id2token[id], cnt) for id, cnt in sorted_dfs]
    print('The top 100 non-stop-words: ', sorted_dfs_token[:100])
    return bow_dictionary

def padding(input_list, max_seq_len, word2idx):
    padded_batch = word2idx['<pad>'] * np.ones((len(input_list), max_seq_len), dtype=np.int)
    for j in range(len(input_list)):
        current_len = len(input_list[j])
        if current_len <= max_seq_len:
            padded_batch[j][:current_len] = input_list[j]
        else:
            padded_batch[j] = input_list[j][:max_seq_len]
    return padded_batch

def BowFeature(input_list, bow_dictionary):
    '''
    generate Bow Feature for train\val\test src
    '''
    bow_vocab = len(bow_dictionary)
    res_src_bow = np.zeros((len(input_list), bow_vocab), dtype=np.int)
    for idx, bow in enumerate(input_list):
        bow_k = [k for k, v in bow]
        bow_v = [v for k, v in bow]
        res_src_bow[idx, bow_k] = bow_v
    return res_src_bow

def encode_one_hot(inst, vocab_size, label_from):
    '''
    one hot for a value x, int, x>=1
    '''
    one_hots = np.zeros(vocab_size, dtype=np.float32)
    for value in inst:
        one_hots[value-label_from]=1
    return one_hots

def build_dataset(src_trgs_pairs, word2idx, tag2idx, bow_dictionary, vocab_size, max_seq_len):
    '''
    build train/valid/test dataset
    '''
    text = []
    label = []
    bow = [] 
    for idx, (source, targets) in enumerate(src_trgs_pairs):
        src = [word2idx[w] if w in word2idx and word2idx[w] < vocab_size
               else word2idx['<unk>'] for w in source]
        trg = [tag2idx[w] for w in targets if w in tag2idx]
        src_bow = bow_dictionary.doc2bow(source)
        text.append(src)
        label.append(trg)
        bow.append(src_bow)    
    bow = BowFeature(bow, bow_dictionary)
    text = padding(text, max_seq_len, word2idx)
    label =  [encode_one_hot(inst, len(tag2idx), label_from=0) for inst in label] 
    return np.array(text), np.array(label), np.array(bow)
