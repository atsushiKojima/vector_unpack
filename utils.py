# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:24:05 2020

@author: a-kojima
"""

def get_word2id_dict(dict_path):
    f = open(dict_path, 'r', encoding='utf-8')
    lines = f.readlines()
    f.close()
    d = {}
    for i, line in enumerate(lines):        
        d[line.replace('\n', '')] = i
    return d


def get_id_seq(dict_, word_seq):
    result = []
    for i in word_seq:
        result.append(dict_[i])
    return result
        
        