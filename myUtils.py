import os
import sys
import csv
import codecs
import time
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def k_flod_avg(record, opt):
    record = np.array(record)
    print('==================================Average Result================================')
    if len(record.shape)==2:
        avg_f1 = round(record.mean(),opt.round)
        print('avg f1@5', avg_f1)
        plt.hist(record)
        plt.show()
        return avg_f1    
    if len(record.shape)==3:
        avg_all = np.round(record.mean(axis=0),opt.round)
        avg_all_reshape = np.reshape(avg_all, (-1)) 
        print('avg report\n', pd.DataFrame(avg_all_reshape).T)
        print('f1@5 stat')
        plt.hist(record[:,-1,-1])
        plt.show()
        return avg_all_reshape

def time_since(start_time):
    return time.time()-start_time
    
def convert_time2str(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh-%02dm" % (h, m)
    
def cp(var, var_name='begin'):
    '''
    clearly print, in other words, let us clearly find the print
    '''
    print('===================%s===================='%var_name)
    print(var)    
    print('===================%s===================='%var_name)
       
def fp(name):
    '''
    fast print the str and related variable
    '''
    print(name+'\n', sys._getframe().f_back.f_locals[name])


def txt2print(best_ntm_print_name):
    i = 0 
    f=open(best_ntm_print_name, 'r')
    for line in f.readlines():
        print('Topic %d : %s'%(i, line.strip()))    
        i+=1