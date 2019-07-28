#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:01:58 2019

Create vocab object

@author: austin.bellibm.com
"""
import numpy as np
import pickle
from os import path
import re
from gensim.models import KeyedVectors


class Vocab(object):
    def __init__(self, word_index_path, embed_path):
        self.word_index_path = word_index_path
        self.embed_path = embed_path
        self.pad_token = "<PAD>"
        self.pad_idx = 0
        self.unk_token = "<UNK>"
        self.unk_idx = 1
        
        if path.exists(word_index_path) and path.exists(embed_path+".npy"):
            word_index = open(word_index_path, 'r').readlines()
            self.word_index = [word.replace("\n","") for word in word_index]
            
            self.embedding_matrix = np.load(embed_path+".npy")
        else:
            word_index, self.embedding_matrix = self.gen_vocab()
            self.word_index = [word.replace("\n","") for word in word_index]
            
        self.word_idx_dict = {word:i for i, word in enumerate(self.word_index)}


    # Initialization of embeddings
    def gen_vocab(self):
        wv = KeyedVectors.load_word2vec_format("./Models/GoogleNews-vectors-negative300.bin", binary = True)
        data_paths = ["./Vendor/Train_Vendor","./Vendor/Test_Vendor",
                      "./Buyer1/Train_Buyer1","./Buyer1/Test_Buyer1",
                      "./Buyer2/Train_Buyer2","./Buyer2/Test_Buyer2"]
        
        word_index = []
        word_index.extend([self.pad_token, self.unk_token])
        for in_path in data_paths:
            with(open(in_path, 'rb')) as f:
                corpus = pickle.load(f)
                
            docs, _ = zip(*[(line['doc'], line['labels']) for line in corpus])
            
            # add new words to the word index
            for doc in docs:
                sentences = doc.split("\n")
                for sent in sentences:
                    sent = sent.split(" ") # already tokenized
                    word_index.extend([word for word in sent if word not in word_index])
    
    
            print("Completed: " + in_path)
        # generate embedding matrix
        emb_matrix = np.zeros((len(word_index), 300))
        for i, word in enumerate(word_index):
            try:
                emb_matrix[i] = wv[word]
            except:
                pass
            
        # save 
        with open(self.word_index_path, 'w') as f:
            for word in word_index:
                f.write(word +"\n")
        np.save(self.embed_path, emb_matrix)
        
        return word_index, emb_matrix
    
    
    def word2idx(self, word):
        return self.word_idx_dict[word] if word in self.word_index else self.unk_idx
    
    def idx2word(self, idx):
        return self.word_index[idx] 
    
    def get_vector(self, inp, word = False):
        if word == False:
            return self.embedding_matrix[inp]
        else:
            return self.embedding_matrix[self.word2idx(word)]

    
    
             

    