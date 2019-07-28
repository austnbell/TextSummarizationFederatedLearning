#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:31:23 2019

Script to train model prior to federated implementation

@author: austin.bellibm.com
"""

import numpy as np 
from gensim.models import KeyedVectors

############## DEFINE PARAMETERS
doc_len = 100
sent_len = 50


lr = .001
bs = 32
epochs = 5

wv = KeyedVectors.load_word2vec_format("./Models/GoogleNews-vectors-negative300.txt", binary = False)
vocab_sz = len(wv.vocab)
word_embed_dim = 300
pos_embed_dim = 100 # for both relative and absolute
hidden_sz = 500 # for all LSTM

