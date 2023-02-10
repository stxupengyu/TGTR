import numpy as np
from tqdm import tqdm
from bert.bertModel import myLoss
from myTest import evaluator
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
            mask = src['attention_mask'].to(opt.device)
            input_id = src['input_ids'].squeeze(1).to(opt.device)
            trg = trg.to(opt.device)
           
            src_bow = src_bow.to(opt.device)
            src_bow_norm = F.normalize(src_bow)
            if opt.topic_type == 'z':
                topic_represent, _, _, _, _ = ntg_model(src_bow_norm)
            else:
                _, topic_represent, _, _, _ = ntg_model(src_bow_norm)

            y_pred = model(input_id, mask, topic_represent)
            loss = myLoss(y_pred, trg.float(),opt)
            total_loss += loss.item()
            
    valid_loss = total_loss/valid_batch_num
    return valid_loss

def bertTest(model, ntg_model, test_data_loader , check_pt_ntg_model_path, check_pt_model_path, opt):
   
    print("=======================Result=======================")

    #load best model 
    model.load_state_dict(torch.load(check_pt_model_path, map_location=opt.device))
    model.to(opt.device)
    model.eval()
    ntg_model.load_state_dict(torch.load(check_pt_ntg_model_path, map_location=opt.device))
    ntg_model.to(opt.device)
    ntg_model.eval()
  
    #test
    y_test = None
    y_pred = None
    with torch.no_grad():
        for batch_i, batch in enumerate(test_data_loader):
            src_bow, test_input, trg = batch
            # move data to GPU if available
            mask = test_input['attention_mask'].to(opt.device)
            input_id = test_input['input_ids'].squeeze(1).to(opt.device)
            test_label = trg.to(opt.device)
            #NTG
            src_bow = src_bow.to(opt.device)
            src_bow_norm = F.normalize(src_bow)
            if opt.topic_type == 'z':
                topic_represent, _, recon_batch, mu, logvar = ntg_model(src_bow_norm)
            else:
                _, topic_represent, recon_batch, mu, logvar = ntg_model(src_bow_norm)
            #main model
            output = model(input_id, mask, topic_represent)
            if y_test is None:
                y_test = test_label
                y_pred = output
            else:
                y_test = torch.cat((test_label,y_test),0)           
                y_pred = torch.cat((output,y_pred),0)             
    y_test = y_test.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    result = evaluate(y_test, y_pred)
    return result  
            
def evaluate(y_real, y_pred): 
    result = []
    for top_K in [1, 3, 5]:
        precision, recall, f1 = evaluator(y_real, y_pred, top_K)
        print('pre@%d,re@%d,f1@%d'%(top_K,top_K,top_K))
        print(round(precision,1),round( recall,1),round( f1,1)) 
        result.append([precision, recall, f1])  
    return result    
           
