#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 18:12:53 2019

Topic model class for Part 3

@author: austin.bellibm.com
"""

import pandas as pd
import numpy as np
from copy import copy

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score

import spacy
nlp = spacy.load('en_core_web_sm',disable = ["tagger", "parser", "ner"])



# Save to other py file and load in to this file
class model_20ng(object):
    def __init__(self):
        self.nb_vectors = []
        self.SVMs = []
        self.vec = None
       
    def add_topics(self, topics):
        self.topics = topics.copy()
        self.topics += ['Other'] # add other category
        
    def add_result(self, nb_vector, SVM):
        self.nb_vectors.append(nb_vector)
        self.SVMs.append(SVM)
    
    # Tokenizer for vectorizer
    def spacy_tokenizer(self, doc):
        return [x.orth_ for x in nlp(doc)]
    
    # Naive bayes calculation
    def pr(self, x, y_i, y):
        p = x[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)

    # model pipeline
    def get_mdl(self, x, y):
        y = y.values
        r = np.log(self.pr(x,1,y) / self.pr(x,0,y))
        m = LogisticRegression(C=4, dual=True)
        x_nb = x.multiply(r)
        return m.fit(x_nb, y), r
    
    # select the argmax label (i.e., select category)
    def select_max_labels(self, preds, ret_score_threshold = False, score_threshold = 0):
        max_labels = preds.argmax(axis = 1)
    
        if ret_score_threshold == True:
            if score_threshold < 0 or score_threshold > 1:
                raise ValueError("Select score between 0 and 1")
    
            # identify which argmax values exceed threshold score
            threshold_vec = preds.max(axis = 1)
            threshold_vec[threshold_vec >= score_threshold] = 1
            threshold_vec[threshold_vec < score_threshold] = 0
    
        else:
            threshold_vec = np.ones(len(preds))
    
        return max_labels, threshold_vec

    # Evaluate
    def evaluate(self, y_true, y_test):
        accuracy = accuracy_score(y_true, y_test)
        print("Model Accuracy is.... ", str(round(accuracy*100,2)))

        print(classification_report(y_true, y_test))
        
    # run model on new data
    def run_nbsvm(self, data, score_threshold = 0):
        print("\n...Generating TF-IDF Matrices...")
        Xvec = self.vec.transform(data) # apply vectorizer
    
        # run model on data
        preds = np.zeros((len(data), len(self.topics)-1))
    
        print("\n...Fitting models...")
        for i, col in enumerate(self.topics[:-1]):
            print(col)
            model = self.SVMs[i] 
            r = self.nb_vectors[i]
            preds[:,i] = model.predict_proba(Xvec.multiply(r))[:,1]
    
        return preds
        
