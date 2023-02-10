import numpy as np
from tqdm import tqdm
from myModel import myLoss, NTG
from myUtils import fp, cp, time_since
import torch
import torch.nn.functional as F
import time

def valid_one_epoch(data_loader, model, ntg_model, opt):

    model.to(opt.device)
    ntg_model.to(opt.device)
    model.eval()
    ntg_model.eval()
    total_loss = 0
    valid_batch_num = len(data_loader)
    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            src_bow, src, trg = batch

            # move data to GPU if available
            src = src.to(opt.device)
            trg = trg.to(opt.device)
           
            src_bow = src_bow.to(opt.device)
            src_bow_norm = F.normalize(src_bow)
            if opt.topic_type == 'z':
                topic_represent, _, _, _, _ = ntg_model(src_bow_norm)
            else:
                _, topic_represent, _, _, _ = ntg_model(src_bow_norm)

            y_pred = model(src, topic_represent)
            loss = myLoss(y_pred, trg.float(),opt)
            total_loss += loss.item()
            
    valid_loss = total_loss/valid_batch_num
    return valid_loss

def myTest(model, ntg_model, test_data_loader , check_pt_ntg_model_path, check_pt_model_path, opt):
   
    print("=======================Result=======================")
    
    #load best model 
    model.load_state_dict(torch.load(check_pt_model_path, map_location=opt.device))
    model.to(opt.device)
    model.eval()
    ntg_model.load_state_dict(torch.load(check_pt_ntg_model_path, map_location=opt.device))
    ntg_model.to(opt.device)
    ntg_model.eval()
  
    #test
    y_test = []
    y_pred = []  
    with torch.no_grad():
        for batch_i, batch in enumerate(test_data_loader):
            src_bow, src, trg = batch
            # move data to GPU if available
            src = src.to(opt.device)
            trg = trg.to(opt.device)
            #ntg
            src_bow = src_bow.to(opt.device)
            src_bow_norm = F.normalize(src_bow)
            if opt.topic_type == 'z':
                topic_represent, _, recon_batch, mu, logvar = ntg_model(src_bow_norm)
            else:
                _, topic_represent, recon_batch, mu, logvar = ntg_model(src_bow_norm)
            #main model
            pred = model(src, topic_represent)

            labels_cpu = trg.data.cpu().float().numpy()
            pred_cpu = pred.data.cpu().numpy()
            pred_cpu = np.exp(pred_cpu)
            
            y_test.append(labels_cpu)
            y_pred.append(pred_cpu)
    result = evaluate(y_test, y_pred, opt)
    return result  
            
def evaluate(y_real, y_pred, opt): 
    y_real = np.array(y_real)
    y_real = np.reshape(y_real,(-1,y_real.shape[-1]))
    y_pred = np.array(y_pred)
    y_pred = np.reshape(y_pred,(-1,y_pred.shape[-1]))    
    for top_K in opt.top_K_list:
        precision, recall, f1 = evaluator(y_real, y_pred, top_K)
        precision, recall, f1 = round(precision,opt.round),round( recall,opt.round),round( f1,opt.round)
        print('pre@%d,re@%d,f1@%d'%(top_K,top_K,top_K))
        print(precision, recall, f1)  
    return [precision, recall, f1]    
           
        
def evaluator(y_true, y_pred, top_K):
    precision_K = []
    recall_K = []
    f1_K = []
    for i in range(y_pred.shape[0]):
        if np.sum(y_true[i, :])==0:
            continue
        top_indices = y_pred[i].argsort()[-top_K:]
        p = np.sum(y_true[i, top_indices]) / top_K
        r = np.sum(y_true[i, top_indices]) / np.sum(y_true[i, :])
        precision_K.append(p)
        recall_K.append(r)
    precision = np.mean(np.array(precision_K))*100
    recall = np.mean(np.array(recall_K))*100
    f1 = precision*recall*2/(precision+recall)
    return round(precision,3),round( recall,3),round( f1,3)   