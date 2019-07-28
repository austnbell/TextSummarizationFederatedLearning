#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:18:35 2019

@author: austin.bellibm.com
"""

evaluator = Evaluator(vocab)

#evaluator.gen_text(docs[0:2])

predicted = evaluator.predicted_summary(docs[0:2], model)
gold = evaluator.gold_summary(docs[0:2], labels[0:2])

evaluator.compute_rouge(gold, predicted)


"""
Notes:
    - â\x80\x98 character is part of the underlying dataset 
        - I will need to clean this out in the beginning
        
    - I subset the sentences to 30 characters which severely messes up the generated text
        - A better result would be to actually extract from the original text 
        - I do not know if I want to account for this (maybe just fake it in the blog)
        - I would need to generate unique ids and this could take time and require a whole other script
"""