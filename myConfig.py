import logging
import sys
import os
import torch
import numpy as np
import random
import json
import time
from config.law import law_opt
from config.academia import academia_opt
from config.physics import physics_opt
from config.AU import AU_opt

def path_opt(opt):
    #data path parameter
    opt.data_path = '/data/pengyu/%s/'%opt.dataset
    opt.txt_path = opt.data_path+'%s.txt'%opt.dataset  
    timemark = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
    opt.timemark = timemark
    opt.model_path = opt.data_path+'model/%s/%s.%s'
    return opt

def common_parameter_opt(opt):  
    #multi task parameter
    opt.train_mode = 'joint' #ntg, joint, use_ntg
    opt.warm_up_epochs = 3 #warm up  
    opt.attn_mode = 'TAMTA'  #TAMTA 
    opt.delta = 0.1
    
    #ntg parameter
    opt.ntg_warm_up_epochs = 100
    opt.ntg_early_stopping = True
    opt.ntg_early_stop_tolerance = 10
    opt.add_two_loss = True
    opt.ntg_learning_rate = 0.001

    #topic parameter
    opt.bow_vocab = 10000
    opt.topic_num = 60
    opt.topic_type = 'z' #z, g
    opt.target_sparsity = 0.7

    #data parameter
    opt.split_token = '<Tags>:'
    opt.vocab_size = 50000
    opt.max_src_len = 200
    opt.max_trg_len = 5

    #model parameter
    opt.hidden_size = 128
    opt.emb_size = 100
    opt.bidirectional = True
    opt.num_layers = 1
    opt.linear_size = [512, 256]

    #training parameter
    opt.gpuid = 3
    opt.drop_rate = 0.1
    opt.learning_rate = 0.001
    opt.epochs = 100
    opt.batch_size = 256
    opt.start_epoch = 1
    opt.early_stop_tolerance = 10
    opt.learning_rate_decay = 0.5
    opt.seed = 56215
    
    #evaluation parameter
    opt.top_K_list = [1,3,5]
    opt.single_metric = False
    opt.k_flod = 1
    opt.round = 1 
    opt.report_one_epoch = True
    opt.dataset_size = 1
    
    #pre-train parameter
    opt.use_pretrained = False
    opt.pretrained_ntg_model_path = "/data/pengyu/physics/model/joint/physics.seed74894.emb100.hid128.20220401-202852/e10.val_loss=0.009.model_ntg-0h-08m"
    opt.pretrained_model_path = "/data/pengyu/physics/model/joint/physics.seed74894.emb100.hid128.20220401-202852/e10.val_loss=0.009.model-0h-08m"
    return opt  

def specific_parameter_opt(opt):
    if opt.dataset=='law':
        opt = law_opt(opt)
    if opt.dataset=='academia':
        opt = academia_opt(opt)
    if opt.dataset=='physics':
        opt = physics_opt(opt)
    if opt.dataset=='AU':
        opt = AU_opt(opt)
    return opt

def random_seed_opt(opt):
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
    return opt

def train_mode_opt(opt):
    opt.joint_train = False
    opt.only_train_ntg = False
    opt.use_pretrained_ntg = False
    if opt.train_mode == 'joint':
        opt.joint_train = True
    if opt.train_mode == 'ntg':
        opt.only_train_ntg = True
    if opt.train_mode == 'use_ntg':
        opt.use_pretrained_ntg = True  
    return opt

def gpu_opt(opt):
    if torch.cuda.is_available():
        opt.device = torch.device("cuda:%d" % opt.gpuid)
        torch.cuda.set_device(opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        print("CUDA is not available, fall back to CPU.")  
    return opt

def process_opt(opt):
    opt = path_opt(opt)
    opt.exp = opt.dataset
    opt = train_mode_opt(opt)
    opt = random_seed_opt(opt)
    opt = gpu_opt(opt)

    # only train ntg
    if opt.only_train_ntg:
        assert opt.ntg_warm_up_epochs > 0 
        opt.exp += '.topic_num{}'.format(opt.topic_num)
        opt.exp += '.ntg_warm_up_%d' % opt.ntg_warm_up_epochs
        opt.model_path = opt.model_path % (opt.train_mode, opt.exp, opt.timemark)
        if not os.path.exists(opt.model_path):
            os.makedirs(opt.model_path)
        print("Only training the ntg for %d epochs and save it to %s" % (opt.ntg_warm_up_epochs, opt.model_path))
        return opt
        
    # joint train settings
    opt.exp += '.seed{}'.format(opt.seed)
    opt.model_path = opt.model_path % (opt.train_mode, opt.exp, opt.timemark)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    print('Model_PATH : ' + opt.model_path)  
    
    return opt        
       
