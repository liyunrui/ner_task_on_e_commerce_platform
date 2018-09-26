#! /usr/bin/env python3
"""
Created on Sun July 12 2018

@author: Ray

"""

import os # This module provides a portable way of using operating system functionally. 
import time
s = time.time()
os.chdir('preprocessing')

#=============================
# data preparation
#=============================
os.system('python3 -u preprocessing_amazon.py')
os.system('python3 -u preprocessing.py')

#=============================
# feature engineering
#=============================
print ('starting feature engineering')

os.chdir('../py_feature')

os.system('python3 -u tokens_characteristics.py') 
os.system('python3 -u tokens_context.py')
os.system('python3 -u contextual_token_level.py')
os.system('python3 -u tokens_contextual_similarity.py')
os.system('python3 -u token_wordvec.py')
os.system('python3 -u tf_idf.py')


# semantic_similarity
# grouwn clustering
# groupby --> conditional statics features
# char-level

# cnn feature first

os.system('python3 -u concat.py')

e = time.time()
print ('{} mins'.format( (e-s) / 60.0)) # 33.234624 mins
# #=============================
# # experiment
# #=============================
# print ('starting modeling')

# os.chdir('../py_model')s
# os.system('python3 -u brand_detecting_xgb.py')
# os.system('python3 -u evaluating_bradn_detector.py')
