#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 11:08:11 2019

evaluator class of keras model that
- convert text
- convert labels (extract gold standard summary)
- returns predicted summary 
- computes rogue scores 
- 

@author: austin.bellibm.com
"""

import numpy as np
from rouge import Rouge

class Evaluator(object):
    def __init__(self, vocab):
        self.vocab = vocab
    
    def _idx2text(self, doc, return_list = False):
        """
        converts a single document from indices to text
        returns: single document and list of sentences
        """
        
        sent_set = []
        for sentence in doc:
            # if last token is padding then skip 
            if sentence[-1] == 0:
                break
            
            sentence = list(map(lambda word: self.vocab.idx2word(word), sentence))
            sentence = list(filter(lambda word: word != "<PAD>", sentence))
            sentence = ' '.join(word for word in sentence)
            sent_set.append(sentence)
        
        if return_list == False:
            text = '. '.join(sentence for sentence in sent_set)
            return text
        else:
            return [sent_set]
        
    
    def gen_text(self, docs, return_list = False):
        """
        Converts numpy array of documents to text
        requires input of 2d (if one doc) or 3d (if more than one doc) numpy array 
        returns list of converted documents
        """
        
        if docs.ndim == 2:
            docs = np.array([docs])
        
        text_set = []
        for doc in docs:
            text = self._idx2text(doc, return_list)
            text_set.append(text)
            
        return text_set
    
    def _extract_sentences(self, sent_set, label):
        """
        Given a single doc and categorical label set - extracts the sentences and combine into summary
        returns: summary of single doc
        """
    
        # condense labels to single vector and subset sent_set
        label_vector = np.argmax(label, axis = 1)[:len(sent_set)]
        sent_set = np.array(sent_set)
        summary = sent_set[label_vector == 1]
        
        return '.  '.join(sent for sent in summary)
    
    def gold_summary(self, docs, labels):
        """
        Extracts the gold standard summary of a given set of document and labels
        Returns list of golda standard summaries
        """
        if docs.ndim == 2:
            docs = np.array([docs])
            
        if labels.ndim == 2:
            labels = np.array([labels])
        
        # extract text
        text = self.gen_text(docs, return_list = True)

        # extract key sentences
        summaries = []
        for i in range(len(labels)):
            sent_set = text[i]
            label = labels[i]
            #print(label)
            summary = self._extract_sentences(sent_set[0], label)
            summaries.append(summary)
            
        return summaries
    
    def predicted_summary(self, docs, model):
        """
        Makes a summarization prediction using given keras model.
        Returns list of predicted summaries
        """
        
        if docs.ndim == 2:
            docs = np.array([docs])
            
        predictions = model.predict(docs)
        
        # extract text
        text = self.gen_text(docs, return_list = True)

        
        # extract key sentences
        summaries = []
        for i in range(len(predictions)):
            sent_set = text[i]
            label = predictions[i]
            summary = self._extract_sentences(sent_set[0], label)
            summaries.append(summary)
            
        return summaries
    
    def compute_rouge(self, gold_summaries, predicted_summaries):
        """
        Computes Rouge Scores (Rouge-1, Rouge-2, and Rouge-L)
        Returns: list of average rouge scores across summaries
        """
        rouge = Rouge()
        scores = rouge.get_scores(predicted_summaries, gold_summaries, avg=True)
        return scores
        