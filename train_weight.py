# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:17:09 2020

@author: a-kojima
"""
import glob
import random
import torch
import sys
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from vector_unpacker import vector_unpack
from utils import get_id_seq, get_word2id_dict


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(training_data, epoch_):
    acc_utts = 0
    
    while acc_utts < len(training_data):
        f = [] # features
        w = [] # word         
        batch_size = 0
        
        # make batch
        while batch_size < BATCH_SIZE and acc_utts < len(training_data):
            split_line = training_data[acc_utts].replace('\n', '').split('\t')
            if len(split_line) == 4:
                content_words = split_line[2].replace('\'', '').replace('[', '').replace(']', '').replace(' ', '').split(',')
                ids = get_id_seq(word2id_dict, content_words)            
                features = split_line[3]
                featire_path = feaure_direc + '/' + os.path.basename(features)            
                f.append(torch.tensor(np.load(featire_path)).float())
                w.append(torch.tensor(ids))
                batch_size += 1
            acc_utts += 1
            print(acc_utts+1, len(training_data))
                        
        if len(f) != BATCH_SIZE:
            break
        
        # padding
        f_lengths = torch.tensor(np.array([len(x) for x in f], dtype=np.int32), device = DEVICE)        
        w_lengths = torch.tensor(np.array([len(t) for t in w], dtype=np.int32), device = DEVICE)                    
        sorted_f_lengths, perm_index = f_lengths.sort(0, descending = True)
        sorted_w_lengths = w_lengths[perm_index]        
        padded_f = nn.utils.rnn.pad_sequence(f, batch_first = True) 
        padded_w = nn.utils.rnn.pad_sequence(w, batch_first = True)         
        padded_sorted_f = padded_f[perm_index] 
        padded_sorted_w = padded_w[perm_index]     
        
        # forward
        y, y_hat = model(padded_sorted_f, sorted_w_lengths, padded_sorted_w)
        
        # get loss
        loss = model.get_loss(y, y_hat)
        
        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        
        print('loss', loss)


#=============================
# parameters
#=============================
max_iteration = 5000
TAU = 100
BATCH_SIZE = 512
LEARNING_RATE = 0.001
FEATURE_DIM = 300
DECAY = 20

# training file list
paramete_file_d = r'\\ami-storage\CTD_Works\a-kojima\hayashi_work\amed-text-data\parameters\parameter_list_D_train.txt'
paramete_file_h = r'\\ami-storage\CTD_Works\a-kojima\hayashi_work\amed-text-data\parameters\parameter_list_H_train.txt'
vocab = r'\\ami-storage\CTD_Works\a-kojima\hayashi_work\amed-text-data\vocab_list.txt'
feaure_direc = r'\\ami-storage\CTD_Works\a-kojima\hayashi_work\amed-text-data\dir_numpy'


#=============================
# dump file content
#=============================
f = open(paramete_file_d, 'r', encoding='utf-8')
train_list = f.readlines()
f.close()

f = open(paramete_file_h, 'r', encoding='utf-8')
train_list2 = f.readlines()
f.close()
train_list.extend(train_list2)
random.shuffle(train_list)

#=============================
# load model, dict  and optimizer
#=============================
word2id_dict = get_word2id_dict(vocab)
model = vector_unpack(len(word2id_dict))
model.train().to(DEVICE)
# model.load_state_dict(torch.load('./'))
#optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

#=============================
# load model, dict  and optimizer
#=============================
current_lr = LEARNING_RATE
for epoch in range(0, max_iteration):
#    global_count_optimize, learning_rate = train_epoch(model, optimizer, training_data, global_count_optimize, learning_rate)
    if epoch >= DECAY:
        for param_group in optimizer.param_groups:
            current_lr = current_lr * 0.8
            param_group['lr'] = current_lr
    
    train(train_list, epoch)
    #  apply threshold eq (13)   
    model.apply_threshold((epoch+1) / (max_iteration * TAU))
    model.to(DEVICE)
        
    # model save
    torch.save(model.state_dict(), "./vector_unpack_weight_{}".format(str(epoch)))
