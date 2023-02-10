import numpy as np
from tqdm import tqdm
from myUtils import fp, cp, time_since, convert_time2str, txt2print
from bert.bertModel import myLoss
from bert.bertTest import valid_one_epoch, bertTest
from myTrain import train_ntg
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import time
import sys
import logging
import math

def bertTrain(model, ntg_model,optimizer_encoder, optimizer_ntg, optimizer_whole,train_data_loader, valid_data_loader, test_data_loader,train_bow_loader,valid_bow_loader, test_bow_loader, bow_dictionary,opt):      
    print('======================  Start Training  =========================')
    #cp(next(model.parameters()).device, 'next(model.parameters()).device')
    #cp(next(ntg_model.parameters()).device, 'next(ntg_model.parameters()).device')
    
    if opt.only_train_ntg :
        best_ntg_model_path = train_ntg(ntg_model, train_bow_loader, valid_bow_loader, optimizer_ntg, bow_dictionary, opt)
        sys.exit()
    
    if opt.use_pretrained_ntg :
        print("Loading ntg model from %s" % opt.pretrained_ntg_model_path)
        ntg_model.load_state_dict(torch.load(opt.pretrained_ntg_model_path, map_location=opt.device))
        #cp(next(ntg_model.parameters()).device, 'next(ntg_model.parameters()).device')
        check_pt_ntg_model_path, check_pt_model_path  = train_all(model, ntg_model, optimizer_encoder, optimizer_ntg, optimizer_whole, 
            train_data_loader, valid_data_loader, test_data_loader, opt)
        return check_pt_ntg_model_path, check_pt_model_path    
    
    if opt.joint_train :
        best_ntg_model_path = train_ntg(ntg_model, train_bow_loader, valid_bow_loader, optimizer_ntg,bow_dictionary, opt)
        ntg_model.load_state_dict(torch.load(best_ntg_model_path, map_location=opt.device))
        check_pt_ntg_model_path, check_pt_model_path  = train_all(model, ntg_model, optimizer_encoder, optimizer_ntg, optimizer_whole, 
            train_data_loader, valid_data_loader, test_data_loader, opt)    
        print('check_pt_model_path', check_pt_model_path)
        print('check_pt_ntg_model_path', check_pt_ntg_model_path)
        return check_pt_ntg_model_path, check_pt_model_path    

def train_all(model, ntg_model, optimizer_encoder, optimizer_ntg, optimizer_whole, 
            train_data_loader, valid_data_loader, test_data_loader, opt):
    num_stop_dropping = 0
    best_valid_loss = float('inf')
    t0 = time.time()
    print("\nEntering main training for %d epochs" % opt.epochs)
    for epoch in range(opt.start_epoch, opt.epochs + 1):     
        train_loss = train_one_epoch(model, ntg_model, optimizer_encoder, optimizer_whole, opt, epoch, train_data_loader)
        model.eval()
        valid_loss = valid_one_epoch(valid_data_loader, model, ntg_model, opt)
        if valid_loss < best_valid_loss:  # update the best valid loss and save the model parameters
#             print("Valid loss drops")
            sys.stdout.flush()
            best_valid_loss = valid_loss
            num_stop_dropping = 0
            
            check_pt_model_path = os.path.join(opt.model_path, 'e%d.val_loss=%.3f.model-%s' %
                                               (epoch, valid_loss, convert_time2str(time.time() - t0)))
            torch.save(
                model.state_dict(),
                open(check_pt_model_path, 'wb')
            )
#             print('Saving encoder checkpoints to %s' % check_pt_model_path)

            check_pt_ntg_model_path = check_pt_model_path.replace('.model-', '.model_ntg-')
            # save model parameters
            torch.save(
                ntg_model.state_dict(),
                open(check_pt_ntg_model_path, 'wb')
            )
#             print('Saving ntg checkpoints to %s' % check_pt_ntg_model_path)               
        else:
#             print("Valid loss does not drop")
            sys.stdout.flush()
            num_stop_dropping += 1

#         print('Epoch: %d; Time spent: %.2f' % (epoch, time.time() - t0))
        print(
            'training loss: %.3f; validation loss: %.3f; best validation loss: %.3f' % (
                train_loss, valid_loss, best_valid_loss))
        if opt.report_one_epoch:  
            bertTest(model, ntg_model, test_data_loader, check_pt_ntg_model_path, check_pt_model_path, opt)
            
        if num_stop_dropping >= opt.early_stop_tolerance:  
            print('Have not increased for %d check points, early stop training' % num_stop_dropping)
            break                                
    return check_pt_ntg_model_path, check_pt_model_path 

def train_one_epoch(model, ntg_model, optimizer_encoder, optimizer_whole, opt, epoch, train_data_loader):
    if epoch <= opt.warm_up_epochs: #warm up train
        optimizer = optimizer_encoder
        model.train()
        ntg_model.eval()
        print("\nWarmup Training epoch: {}/{}".format(epoch, opt.epochs))        
    else:#train whole
        optimizer = optimizer_whole
        unfix_model(model)
        model.train()
        ntg_model.train()
        print("\nJointly Training epoch: {}/{}".format(epoch, opt.epochs))    
       
    train_batch_num = len(train_data_loader)
    total_loss = 0
    for batch_i, batch in enumerate(tqdm(train_data_loader)):
        batch_loss = train_one_batch(batch, model, ntg_model, optimizer, opt, batch_i)
        total_loss += batch_loss
    current_train_loss = total_loss/train_batch_num
    return current_train_loss
      
def train_one_batch(batch, model, ntg_model, optimizer, opt, batch_i):
    #model.to(opt.device)
    #model.train()
    
    # train for one batch
    src_bow, src, trg = batch
    # move data to GPU if available
    mask = src['attention_mask'].to(opt.device)
    input_id = src['input_ids'].squeeze(1).to(opt.device)
    trg = trg.to(opt.device)

    # model.train()
    optimizer.zero_grad()
     
    src_bow = src_bow.to(opt.device)
    src_bow_norm = F.normalize(src_bow)
    if opt.topic_type == 'z':
        topic_represent, _, recon_batch, mu, logvar = ntg_model(src_bow_norm)
    else:
        _, topic_represent, recon_batch, mu, logvar = ntg_model(src_bow_norm)

    if opt.add_two_loss:
        ntg_loss = loss_function(recon_batch, src_bow, mu, logvar)

    y_pred = model(input_id, mask, topic_represent)
#     print('y_pred', y_pred)
    loss = myLoss(y_pred, trg.float(), opt)

    if math.isnan(loss.item()):
        print("Batch i: %d" % batch_i)
        print("src")
        print(src)
        print("trg")
        print(trg)
        raise ValueError("Loss is NaN")

    if opt.add_two_loss:
        #cp(loss, 'loss')
        #cp(ntg_loss, 'ntg_loss')
        loss = loss+ opt.delta * ntg_loss
    # back propagation on the normalized loss
    loss.backward()
    optimizer.step()
    
    batch_loss = loss.item()           
    return batch_loss

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar): 
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD       

def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfix_model(model):
    for param in model.parameters():
        param.requires_grad = True
