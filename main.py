import myDataset
from myUtils import cp, time_since, k_flod_avg
import torch
import torch.nn.functional as F
from myModel import myModel, NTG, myModelStat, init_optimizers
from myTrain import myTrain
from myTest import myTest
import random
import time
from myConfig import process_opt, common_parameter_opt, specific_parameter_opt
import argparse
from bert.BTGTR import BTGTR

def main(opt):
    
    if opt.encoder == 'bert':
        result = BTGTR(opt)
        return result
    
    #Dataset
    start_time = time.time()
    train_loader, val_loader, test_loader, train_bow_loader, \
    valid_bow_loader, test_bow_loader, bow_dictionary, opt = myDataset.dataProcess(opt) 
    load_data_time = time_since(start_time)
    print('Time for loading the data: %.1f' %load_data_time)

    #Model
    start_time = time.time()
    model = myModel(opt).to(opt.device)
    ntg_model = NTG(opt).to(opt.device)
    optimizer_encoder, optimizer_ntg, optimizer_whole = init_optimizers(model, ntg_model, opt)
    
    #Train 
    if opt.use_pretrained != True:
        check_pt_ntg_model_path, check_pt_model_path = myTrain(model, ntg_model,
                                                               optimizer_encoder, optimizer_ntg, optimizer_whole,
                                                               train_loader, val_loader, test_loader, bow_dictionary,
                                                               train_bow_loader, valid_bow_loader, opt) 
        training_time = time_since(start_time)
        print('Time for training: %.1f' % training_time)
    else:
        check_pt_ntg_model_path, check_pt_model_path = opt.pretrained_ntg_model_path, opt.pretrained_model_path

    #Test
    start_time = time.time()
    result =  myTest(model, ntg_model, test_loader ,check_pt_ntg_model_path, check_pt_model_path, opt) 
    test_time = time_since(start_time)
    print('Time for testing: %.1f' % test_time)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-dataset', type=str, default='law', required=True)
    parser.add_argument('-encoder', type=str, default='lstm')
    opt = parser.parse_args() 
    opt = common_parameter_opt(opt)
    opt = specific_parameter_opt(opt)
    opt = process_opt(opt)
    main(opt)