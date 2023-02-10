from bert.bertConfig import bert_opt
from myUtils import cp, time_since, k_flod_avg
from bert.bertDataset import dataProcess
import torch
import torch.nn.functional as F
from bert.bertModel import BERTModel, init_optimizers
from myModel import NTG
from bert.bertTrain import bertTrain
from bert.bertTest import bertTest
import random
import time

def BTGTR(opt):
    opt = bert_opt(opt)

    #Dataset
    start_time = time.time()
    train_loader, val_loader, test_loader, train_bow_loader, \
    valid_bow_loader, test_bow_loader, bow_dictionary, opt = dataProcess(opt) 
    load_data_time = time_since(start_time)
    print('Time for loading the data: %.1f' %load_data_time)

    #Model
    start_time = time.time()
    model = BERTModel(opt)
    ntg_model = NTG(opt)
    model = model.to(opt.device)
    ntg_model = ntg_model.to(opt.device)
    optimizer_encoder, optimizer_ntg, optimizer_whole = init_optimizers(model, ntg_model, opt)
    
    if opt.use_pretrained != True:
        #Train   
        check_pt_ntg_model_path, check_pt_model_path = bertTrain(model, ntg_model,
                                                               optimizer_encoder, optimizer_ntg, optimizer_whole,
                                                               train_loader, val_loader, test_loader, train_bow_loader,valid_bow_loader, test_bow_loader, bow_dictionary,opt) 
        training_time = time_since(start_time)
        print('Time for training: %.1f' % training_time)
    else:
        check_pt_ntg_model_path, check_pt_model_path = opt.pretrained_ntg_model_path, opt.pretrained_model_path

    #Test
    start_time = time.time()
    result =  bertTest(model, ntg_model, test_loader , check_pt_ntg_model_path, check_pt_model_path, opt) 
    test_time = time_since(start_time)
    print('Time for testing: %.1f' % test_time)
    return result


