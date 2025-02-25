# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:22:54 2020

@author: a-kojima
"""

import torch
import numpy as np
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

MIN_VAL = 10 ** (-10)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class vector_unpack(nn.Module):
    def __init__(self, num_of_vocab):
        super(vector_unpack, self).__init__()      
        
        # 重みを 1 / 単語数 で初期化
        rand = (-0.1 - 0.1) * torch.rand(num_of_vocab) + 0.1

        param_dict = {}
        for i in range(num_of_vocab):
            param_dict[str(i)] = torch.nn.Parameter(torch.ones(1) * rand[i] , requires_grad=True)
        self.weights = nn.ParameterDict(param_dict)
        self.num_vocab = num_of_vocab
    
    def apply_threshold(self, threshold):
        for i in range(0, self.num_vocab):
            if self.weights[str(i)] < threshold:
                self.weights[str(i)] = torch.nn.Parameter(torch.zeros(1) , requires_grad=True)
            
    def get_loss(self, y, y_hat):      
        # loss計算 eq(5)
        return torch.sum(0.5 * torch.sum((y - y_hat) ** 2, dim=1)) 
    
    def forward(self, vector_sequence, sentence_length, word_sequence):
        
        B, T, _ = vector_sequence.size()
        # padding した分の後ろの単語をマスキング
        for index, length in enumerate(sentence_length):
            vector_sequence[index, length:, :] = 0
            
        # L1 norm で正規化 eq(3)
        y = torch.sum(vector_sequence, dim=1) / torch.sqrt(torch.sum(torch.abs(torch.sum(vector_sequence, dim=1)), dim=1)).unsqueeze(1)
        y = y.to(DEVICE)

        # ベクトルの期待値を計算 eq(4)
        y_hat = torch.zeros(vector_sequence.size()).to(DEVICE)
        for i in range(B):
            for j in range(sentence_length[i]):
                y_hat[i, j, :] = vector_sequence[i, j, :].to(DEVICE) * self.weights[str(int(word_sequence[i, j]))]
        y_hat = torch.sum(y_hat, dim=1)
        return y, y_hat        

if __name__ == '__main__':

    # parameters
    NUMBER_OF_SENTENCE = 3 # 重みの更新に用いるバッチサイズ
    SENTENCE_LENGTH = [15, 10, 12] # バッチに含まれる各文の単語数
    MAX_NUM_OF_WORDS = max(SENTENCE_LENGTH) 
    VECTOR_DIM = 128 # word2vecの次元
    VOCAB = 100 # word2vecのvocab数
    LEARNG_RATE = 0.01 # eq (12) のデルタ
    
    # 各分の単語系列
    # ただし、行列演算できるように、SENTENCE_LENGTHの最大値にあわせる
    WORD_SEQUENCE = np.random.randint(0, VOCAB, (NUMBER_OF_SENTENCE, MAX_NUM_OF_WORDS)) 
    # 後で、重みが更新されてるか確認したいので、0 番目のvocabだけ必ず含むようにする
    WORD_SEQUENCE[:, 0] = 0
    
    # word2vec
    word2vec_seq = np.random.randn(NUMBER_OF_SENTENCE, MAX_NUM_OF_WORDS, VECTOR_DIM)
    
    # modelとoptimizerの定義
    model = vector_unpack(VOCAB)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNG_RATE, momentum=0.9)
    
    
    # y と y_hat計算
    y, y_hat = model(word2vec_seq, SENTENCE_LENGTH, WORD_SEQUENCE)
    
    # loss 計算
    loss = model.get_loss(y, y_hat)
    print('loss', loss)
    
    # 重みの値をｌ更新
    print('こうしんまえの重み',  model.weights[str(0)])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print('こうしん後の重み',  model.weights[str(0)])
