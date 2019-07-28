#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:55:16 2019

Prep data for use in Keras SummaRunner Model 
Series of functions that convert json data into npz format with word2vec idx

output format is npz files
nested array of news documents, with each document composed of multiple sentences 

@author: austin.bellibm.com
"""
import numpy as np
import pickle
import spacy
import sys, os
sys.path.append(os.getcwd())

from Programs.SumaRuNNer.Vocab import Vocab

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


#import sys
#sys.path.append("/Users/austin.bellibm.com/Documents/FederatedLearning/Part 3 - Applied NLP")


def gen_keras_input(docs, labels, doc_len, sent_len):
    """
    Function that generates encoded numpy arrays for Keras input
    Returns: encoded sequence and label arrays
    """
    doc_set = []
    label_set = []
    
    for j, (doc, label) in enumerate(zip(docs, labels)):
        sentences = doc.split("\n")
        
        sent_set = []
        for i, sent in enumerate(sentences):
            sent = [token.orth_ for token in nlp(sent)]
            
            # convert word to idx
            sent = [vocab.word2idx(word) for word in sent]
            
            # if greater than doc length then stop
            if i+1 > doc_len:
                break
           
            sent_set.append(sent)
            
        # if doc length not reached then pad
        label_list = label.split("\n")[:50]
        if len(set(label_list)) == 1: # error handling faulty data - removes cases without summary
            print(j)
            continue
        
        if i-1 < doc_len:
            padded_sent = [0]*sent_len
            for k in range(doc_len-i-1):
                sent_set.append(padded_sent)
                label_list.append(0)
            
        # pad sentences
        sent_set = pad_sequences(sent_set, maxlen = sent_len, padding = 'pre', truncating = 'post').tolist()
        
        # add to doc and label set
        label_set.append(to_categorical(label_list).tolist())
        doc_set.append(sent_set)
        
        
        if j % 1000 == 0:
            print(j)
     
    return doc_set, label_set 


if __name__ == "__main__":
    ######### LOAD
    nlp = spacy.load('en_core_web_sm', disable = ["tagger", "parser", "ner"])
    vocab = Vocab(word_index_path = "./Models/Word_Index.txt",
                  embed_path= "./Models/Embeddings")
    
    print("~~~Loaded (or Generated) Word Vectors - Begin Encoding~~~")
    
    ########## PARAMETERS
    sent_len = 30
    doc_len = 50

    # dictionary = {"input/path":"output/path"}
    data_paths = {"./Vendor/Train_Vendor": "./Vendor/Enc_Train_Vendor",
                  "./Vendor/Test_Vendor": "./Vendor/Enc_Test_Vendor",
                  "./Buyer1/Train_Buyer1": "./Buyer1/Enc_Train_Buyer1",
                  "./Buyer1/Test_Buyer1": "./Buyer1/Enc_Test_Buyer1",
                  "./Buyer2/Train_Buyer2": "./Buyer2/Enc_Train_Buyer2",
                  "./Buyer2/Test_Buyer2": "./Buyer2/Enc_Test_Buyer2"}
    
    
    for in_path, out_path in data_paths.items():
        # Import
        print(in_path)
        with(open(in_path, 'rb')) as f:
            corpus = pickle.load(f)
        
        docs, labels = zip(*[(line['doc'], line['labels']) for line in corpus])

        # generate encodings and export
        doc_set, label_set = gen_keras_input(docs, labels, doc_len, sent_len)
        
        out_path_labels =  out_path + "_labels"
        out_path_docs = out_path + "_docs"
        
        np.save(out_path_labels, label_set)
        np.save(out_path_docs, doc_set)
        print("~~~ Completed: " + in_path)

