#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:54:36 2019

Full pipeline for summarunner

@author: austin.bellibm.com
"""

import numpy as np
import spacy
from keras.preprocessing.sequence import pad_sequences

nlp = spacy.load('en_core_web_sm', disable = ['tagger', 'ner'])


text = """
This could be an example summary! I need to test a wide variety of nlp tools before
adding to my flask app.  This includes the spacy tokenizer, i think? But also, I need 
to create the entire pipeline, which is what I am doing below. Later, I will need to 
incorporate all of my saved models (included embeddings and keras results). 

That is where it will get tricky
"""

class SumPipeline(object):
    def __init__(self, doc_len = 50, sent_len = 30):
        self.doc_len = doc_len
        self.sent_len = sent_len
        

    def RunPipeline(self, text, vocab):
        # create sentences
        text = nlp(text)
        sentences = [sent.string.strip() for sent in text.sents]
        self.sentences = sentences
        self.num_sentences = len(sentences)
        
        # Tokenize and convert to embedding idx
        sent_set = []
        for i, sent in enumerate(sentences):
            sent = sent.replace("\n", " ")
            sent = [token.orth_ for token in nlp(sent)]
            
            # convert word to idx
            sent = [vocab.word2idx(word) for word in sent]
            
            # if greater than doc length then stop
            if i+1 > self.doc_len:
                break
           
            sent_set.append(sent)
         
        # pad document then sentences
        if i-1 < self.doc_len:
            padded_sent = [0]*self.sent_len
            for k in range(self.doc_len-i-1):
                sent_set.append(padded_sent)
           
        sent_set = pad_sequences(sent_set, maxlen = self.sent_len, padding = 'pre', truncating = 'post').tolist()
        
        return np.array([sent_set])
    
    


    