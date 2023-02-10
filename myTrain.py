import numpy as np
from tqdm import tqdm
from myUtils import fp, cp, time_since, convert_time2str, txt2print
from myModel import myLoss
from myTest import valid_one_epoch, myTest
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
import time
import sys
import logging
import math

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
    for batch_i, batch in enumerate(train_data_loader):
        batch_loss = train_one_batch(batch, model, ntg_model, optimizer, opt, batch_i)
        total_loss += batch_loss
    current_train_loss = total_loss/train_batch_num
    return current_train_loss

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
            myTest(model, ntg_model, test_data_loader, check_pt_ntg_model_path, check_pt_model_path, opt)
        if num_stop_dropping >= opt.early_stop_tolerance:  
            print('Have not increased for %d check points, early stop training' % num_stop_dropping)
            break                                
    return check_pt_ntg_model_path, check_pt_model_path     


def train_ntg(ntg_model, train_bow_loader, valid_bow_loader, optimizer_ntg, bow_dictionary, opt): 
    num_stop_dropping = 0
    best_valid_loss = float('inf')
    best_ntg_model_path = os.path.join(opt.model_path, 'best_ntg_model')
    best_ntg_print_path = os.path.join(opt.model_path, 'topwords.txt')
    print("\nWarming up ntg for %d epochs" % opt.ntg_warm_up_epochs)
    
    for epoch in range(1, opt.ntg_warm_up_epochs + 1):
        sparsity = train_ntg_one_epoch(ntg_model, train_bow_loader, optimizer_ntg, opt, epoch)
        valid_loss = test_ntg_one_epoch(ntg_model, valid_bow_loader, opt, epoch)
        if epoch%10==0:
            print('ntg Training %d/%d, validation loss: %.3f'%(epoch,
                                                                               opt.ntg_warm_up_epochs,valid_loss))
        if opt.ntg_early_stopping:
            
            if valid_loss < best_valid_loss:
                num_stop_dropping = 0
                best_valid_loss = valid_loss
                sys.stdout.flush()              
                best_ntg_model_name = os.path.join(opt.model_path, 'e%d.val_loss=%.3f.sparsity=%.3f.ntg_model' %
                                               (epoch, valid_loss, sparsity))
                best_ntg_print_name = os.path.join(opt.model_path, 'topwords_e%d.txt' % epoch)        
                torch.save(
                    ntg_model.state_dict(),
                    open(best_ntg_model_path, 'wb')
                )
                ntg_model.print_topic_words(bow_dictionary, best_ntg_print_path)
            else:
                num_stop_dropping += 1
            if num_stop_dropping >= opt.ntg_early_stop_tolerance:  
                print('Have not increased for %d check points, early stop training' % num_stop_dropping)
                break      
                
        if not opt.ntg_early_stopping: 
            if epoch==opt.ntg_warm_up_epochs: 
                sys.stdout.flush()              
                best_ntg_model_name = os.path.join(opt.model_path, 'e%d.val_loss=%.3f.sparsity=%.3f.ntg_model' %
                                               (epoch, valid_loss, sparsity))
                best_ntg_print_name = os.path.join(opt.model_path, 'topwords_e%d.txt' % epoch)        
                torch.save(
                    ntg_model.state_dict(),
                    open(best_ntg_model_path, 'wb')
                )
                ntg_model.print_topic_words(bow_dictionary, best_ntg_print_path)
                
    os.rename(best_ntg_model_path , best_ntg_model_name)
    os.rename(best_ntg_print_path , best_ntg_print_name)
    txt2print(best_ntg_print_name)    
    print("\nSaving warm up ntg model into %s" % best_ntg_model_name) 
    print('==========================train_ntg completed===========================')           
    return best_ntg_model_name                

def myTrain(model, ntg_model, optimizer_encoder, optimizer_ntg, optimizer_whole, 
        train_data_loader, valid_data_loader, test_data_loader, bow_dictionary, train_bow_loader,
        valid_bow_loader, opt):      
    print('======================  Start Training  =========================')
    #cp(next(model.parameters()).device, 'next(model.parameters()).device')
    #cp(next(ntg_model.parameters()).device, 'next(ntg_model.parameters()).device')
    
    if opt.only_train_ntg :
        best_ntg_model_path = train_ntg(ntg_model, train_bow_loader, valid_bow_loader, optimizer_ntg,bow_dictionary, opt)
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
   
   
def train_one_batch(batch, model, ntg_model, optimizer, opt, batch_i):
    #model.to(opt.device)
    #model.train()
    
    # train for one batch
    src_bow, src, trg = batch
    # move data to GPU if available
    src = src.to(opt.device)
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

    #assert 1==0 
    y_pred = model(src, topic_represent)
    loss = myLoss(y_pred, trg.float(),opt)

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
        
def train_ntg_one_epoch(model, dataloader, optimizer, opt, epoch):
    model.train()
    train_loss = 0
    for batch_idx, data_bow in enumerate(dataloader):
        ##cp(data_bow,'data_bow')
        [data_bow] = data_bow
        data_bow = data_bow.to(opt.device)
        # normalize data
        data_bow_norm = F.normalize(data_bow)
        optimizer.zero_grad()
        
        ##cp(data_bow,'data_bow')
        ##cp(data_bow_norm,'data_bow_norm')
        
        _, _, recon_batch, mu, logvar = model(data_bow_norm)
        
#         cp(recon_batch.shape,'recon_batch.shape')
#         cp(data_bow.shape,'data_bow.shape')
#         cp(data_bow_norm.shape,'data_bow_norm.shape')
#         cp(mu.shape,'mu.shape')
        
        loss = loss_function(recon_batch, data_bow, mu, logvar)
        loss = loss + model.l1_strength * l1_penalty(model.fcd1.weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data_bow), len(dataloader.dataset),
#                        100. * batch_idx / len(dataloader),
#                        loss.item() / len(data_bow)))

#     print('====>Train epoch: {} Average loss: {:.4f}'.format(
#         epoch, train_loss / len(dataloader.dataset)))
    sparsity = check_sparsity(model.fcd1.weight.data)
#     print("Overall sparsity = %.3f, l1 strength = %.5f" % (sparsity, model.l1_strength))
#     print("Target sparsity = %.3f" % opt.target_sparsity)
    update_l1(model.l1_strength, sparsity, opt.target_sparsity)
    return sparsity

def test_ntg_one_epoch(model, dataloader, opt, epoch):#no use 
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            [data_bow] =  data_bow
            data_bow = data_bow.to(opt.device)
            data_bow_norm = F.normalize(data_bow)

            _, _, recon_batch, mu, logvar = model(data_bow_norm)
            test_loss += loss_function(recon_batch, data_bow, mu, logvar).item()

    avg_loss = test_loss / len(dataloader.dataset)
#     print('====> Val epoch: {} Average loss:  {:.4f}'.format(epoch, avg_loss))
    return avg_loss

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
#     cp(x,'x')
#     cp(x.shape,'x.shape') 
#     cp(int(x.max()),'x.max()')
      
#     cp(recon_x,'recon_x')
#     cp(recon_x.shape,'recon_x.shape')
#     cp(int(recon_x.max()),'recon_x.max()')
    
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def l1_penalty(para):
    return nn.L1Loss()(para, torch.zeros_like(para))

def check_sparsity(para, sparsity_threshold=1e-3):
    num_weights = para.shape[0] * para.shape[1]
    num_zero = (para.abs() < sparsity_threshold).sum().float()
    return num_zero / float(num_weights)

def update_l1(cur_l1, cur_sparsity, sparsity_target):
    diff = sparsity_target - cur_sparsity
    cur_l1.mul_(2.0 ** diff)

def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfix_model(model):
    for param in model.parameters():
        param.requires_grad = True
