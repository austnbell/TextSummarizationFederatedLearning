#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:31:28 2019

@author: austin.bellibm.com
"""

from flask import Flask, render_template, request
import numpy as np
import os, sys
sys.path.append(os.getcwd())

from Programs.SumaRuNNer.Vocab import Vocab
from Programs.Application.Pipeline import SumPipeline
import keras
import tensorflow as tf
import math
from keras.activations import sigmoid, tanh




# init object 
global vocab, pipe, model, graph
graph = tf.get_default_graph()
vocab = Vocab(word_index_path = "./Models/Word_Index.txt",
                  embed_path= "./Models/Embeddings")
pipe = SumPipeline()

model = keras.models.load_model('./Models/SummaRuNNer_Federated.h5', 
                                custom_objects={"tf": tf,
                                                "math":math,
                                                "sigmoid":sigmoid,
                                                "tanh":tanh})

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def Summarize():
    
    # get text and standardize
    text = request.form['text']
    
    # error handling
    if text == "":
        template_data = {
          'summary' : "No Text Provided"
          }
        
        return render_template("index.html", **template_data)
       
        
    model_input = pipe.RunPipeline(text, vocab)
    
    # Run through model 
    with graph.as_default():
        preds = model.predict(model_input)[0]
    
        label_vector = np.argmax(preds, axis = 1)[:pipe.num_sentences]
        sent_set = np.array(pipe.sentences)
        summary = sent_set[label_vector == 1]
        summary = ' '.join(sent for sent in summary)
        summary = summary.replace("\n", " ")
    
        template_data = {
          'summary' : summary
          }
        
        return render_template("index.html", **template_data)
    


if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5001))
	app.run(host='0.0.0.0', port=port, debug = True)