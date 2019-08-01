#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:07:22 2019

Keras implementation of SummaRuNNer

@author: austin.bellibm.com
"""


import numpy as np  
import math
import argparse
import sys, os
sys.path.append(os.getcwd())

import tensorflow as tf
from Programs.SumaRuNNer.Vocab import Vocab
from keras.layers import Input, LSTM, Embedding, Dense, Lambda, \
AveragePooling1D, Bidirectional, TimeDistributed, concatenate, Reshape, Flatten
from keras.models import Model
from keras.activations import sigmoid, tanh
from keras.initializers import Constant


        

##############################
# KERAS MODEL
##############################

def SummaRuNNer():

    # initialize embedding layers
    embed_layer = TimeDistributed(Embedding(vocab_sz, 
                            word_embed_dim,
                            embeddings_initializer = Constant(vocab.embedding_matrix),
                            input_length = args.sent_len,
                            trainable = False))
    
    abs_embed_layer = Embedding(args.doc_len, args.pos_embed_dim, input_length = 1, trainable = True)
    rel_embed_layer = Embedding(rel_segments, args.pos_embed_dim, input_length = 1, trainable = True)
    
    
    # input shape [bs, doc length, sentence length]
    doc_input = Input(shape = (int(args.doc_len), int(args.sent_len)), name = 'doc_input') 
    
    # word embedding 
    word_emb_seq = embed_layer(doc_input)
    
    # LSTM on each each word - return sequence
    word_LSTM = TimeDistributed(Bidirectional(LSTM(args.hidden_sz, return_sequences = True)))
    enc_words = word_LSTM(word_emb_seq)
    
    avg_pooler = TimeDistributed(AveragePooling1D(args.sent_len))
    pooled_words = Reshape((args.doc_len,2*args.hidden_sz), name = 'sent_pooler')(avg_pooler(enc_words))
    
    # run another word LSTM so that each sentece is represented by a single vector
    sent_LSTM = Bidirectional(LSTM(args.hidden_sz, return_sequences = True)) 
    enc_sents = sent_LSTM(pooled_words)
    
    # create single vector for document
    doc_pooler = AveragePooling1D(args.doc_len)
    doc = Flatten(name = 'flatten_doc')(doc_pooler(enc_sents))
    d = Dense(int(2*args.hidden_sz), activation = 'tanh', name = 'dense_doc')(doc)
    
    # novelty tracker
    s = Lambda(lambda x: K.zeros_like(x), name = 's_tensor')(d) # [?, 2*h]
    
    probs = []
    # placeholder
    T = Lambda(lambda x: (K.ones_like(x[:,0:1], name = 'T_tensor')))(s) 
    
    
    split_sentences = Lambda(lambda tensor, doc_len: tf.unstack(tensor, doc_len, 1), 
                             arguments = {'doc_len':args.doc_len})(enc_sents)
    
    # run every sentence through classification layer and store probability
    for pos in range(len(split_sentences)):
    
        sent = Lambda(lambda sentences, pos: sentences[pos], arguments = {'pos':pos})(split_sentences)
        
        # run the absolute embedding
        abs_idx = Lambda(lambda T, pos: T*pos, arguments = {'pos':pos})(T) 
        abs_emb = Reshape((args.pos_embed_dim,), name = 'abs_'+str(pos))(abs_embed_layer(abs_idx))
        
        """
        get relative position and run through relative embedding
        refers to a quantized representation that divides each document into a 
        fixed number of segments and computes the segment ID of a given sentence.
        """
        rel_idx = math.floor(((pos + (rel_segments)/2) / args.doc_len)*10) # only works for rel_segments = 10
        rel_idx = Lambda(lambda T, rel_idx: T*(rel_idx-1), arguments = {'rel_idx':rel_idx})(T)
        rel_emb = Reshape((args.pos_embed_dim,), name = 'rel_'+str(pos))(rel_embed_layer(rel_idx))
        
        
        # classifier layer 
        content = Dense(2, name = 'content_'+str(pos))(sent)
        salience = Dense(2, name = 'salience_'+str(pos))(Lambda(lambda x: x[0]*x[1])([sent,d]))
        novelty = Dense(2, name = 'novelty_'+str(pos))(Lambda(lambda x: x[0]*tanh(x[1]))([sent,s]))  
        abs_pos = Dense(2, name = 'abs_pos_'+str(pos))(abs_emb)
        rel_pos = Dense(2, name = 'rel_pos_'+str(pos))(rel_emb)
        
        p = Lambda(lambda x: sigmoid(x[0]+x[1]+x[2]+x[3]+x[4]))([content, salience, novelty, abs_pos, rel_pos])
        probs.append(p)
        
        
        # extract just the probability of label = 1
        p1 = Lambda(lambda p: p[:,1:])(p)
        
        # weighted summation of all sentence encodings until now
        # weight = probability that sentences was part of summary
        s = Lambda(lambda x: x[0] + (x[1]*x[2]))([s, p1, sent])
        
        
    output = Reshape((args.doc_len,2), name = 'prob_reshape')(concatenate(probs, -1))
       
    
    model = Model(inputs = doc_input, outputs = output)
    model.compile(optimizer = 'sgd',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    return model

if __name__ == "__main__":
    
    # parse command line for model arguments
    parser = argparse.ArgumentParser(description='extractive summary')

    parser.add_argument("-cmd", type = str, default = "TRAIN")

    # model inputs
    parser.add_argument("-doc_len", type = int, default = 50)
    parser.add_argument("-sent_len", type = int, default = 30)
    parser.add_argument("-lr", type = float, default = .001)
    parser.add_argument("-bs", type = int, default = 32)
    parser.add_argument("-epochs", type = int, default = 10)

    # model dimensions
    parser.add_argument("-pos_embed_dim", type = int, default = 50)
    parser.add_argument("-hidden_sz", type = int, default = 250)

    args = parser.parse_args()

    # non-alterable parameters 
    vocab = Vocab(word_index_path = "./Models/Word_Index.txt",
                      embed_path= "./Models/Embeddings")
    vocab_sz = len(vocab.word_index)
    word_embed_dim = 300
    rel_segments = 10 # do not change
    
    from keras import backend as K 
    K.clear_session() 
    
    print("~~~Developing Graph~~~")
    model = SummaRuNNer()
    print("~~~Completed Graph~~~")
    
    if args.cmd == "SAVE":
        model.save('./Models/SummaRuNNer_spec.h5')
        
    elif args.cmd == "TRAIN":
        docs = np.load("./Vendor/Enc_Train_Vendor_docs.npy")
        labels = np.load("./Vendor/Enc_Train_Vendor_labels.npy")

        model.fit(docs[:80000], labels[:80000], batch_size = args.bs, epochs = args.epochs, verbose = 2)
        model.save('./Models/SummaRuNNer_Baseline.h5')

"""
import keras
test_model = keras.models.load_model('./model_1564166525.598958.h5', 
                                     custom_objects={"tf": tf,
                                                     "math":math,
                                                     "sigmoid":sigmoid})
"""