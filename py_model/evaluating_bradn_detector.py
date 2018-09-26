#! /usr/bin/env python3
"""
Created on July 27 2018

@author: Ray

"""

import logging # recording the result
import os
from xgboost import plot_importance
import pandas as pd
import numpy as np
import pickle  # load model
import gc
import time
import multiprocessing as mp # for speeding up some process
from utils import get_best_threshold
from utils import get_positive_token
from utils import init_logging
from utils import evaluating_system
import logging
from datetime import datetime # for the newest version control

def evaluate(T):
	# log path
	log_dir = '../log/log_system_performance'
	init_logging(log_dir)
	# model path
	output_dir = '../output/model/'
	date_str = datetime.now().strftime('%Y-%m-%d')
	output_dir = os.path.join(output_dir, date_str)	

	if T == 1:
		name = 'tv_and_laptop'
		#--------------------------
		# load model trained on amazon
		#--------------------------
		model = pickle.load(open(os.path.join(output_dir, 'tv_laptop_amazon_xgb.model'), "rb"))
		#--------------------------
		# load shopee data
		#--------------------------
		df = pd.read_hdf('../features/tv_and_laptop/all_features.h5')
		# feat = [f for f in df.columns.tolist() if 'pos' not in f]
		# df = df[feat]

		#--------------------------
		# save model importance plot
		#--------------------------
	elif T == 2:
		name = 'beauty'
		#--------------------------
		# load model trained on amazon
		#--------------------------
		model = pickle.load(open(os.path.join(output_dir, 'beauty_amazon_xgb.model'), "rb"))
		#--------------------------
		# load shopee data
		#--------------------------
		df = pd.read_hdf('../features/personal_care_and_beauty/all_features.h5')

	elif T == 3:
		name = 'beauty_benchmark'
		#--------------------------
		# load model trained on amazon
		#--------------------------
		model = pickle.load(open(os.path.join(output_dir, 'personal_care_and_beauty_xgb.model'), "rb"))
		#--------------------------
		# load amazon data
		#--------------------------
		df = pd.read_hdf('../features/personal_care_and_beauty/all_features.h5')

	elif T == 4:
		name = 'tv_laptop_benchmark'
		#--------------------------
		# load model trained on amazon
		#--------------------------
		model = pickle.load(open(os.path.join(output_dir, 'tv_and_laptop_xgb.model'), "rb"))
		#--------------------------
		# load amazon data
		#--------------------------
		df = pd.read_hdf('../features/tv_and_laptop/all_features.h5')
		# feat = [f for f in df.columns.tolist() if 'pos' not in f]
		# df = df[feat]

	#--------------------------
	# determining the threshold
	#--------------------------

	# take 50% of shopee data as validating data
	val = df[df.is_valid == 0]
	x_val = val.drop(['item_name', 'tokens', 'label', 'is_valid'], axis = 1)
	y_val = val.label
	# prediction
	valid_yhat_prob_is_brand = model.predict_proba(x_val)[:,1] 

	# combination val and val_predict
	df = pd.concat([val[['item_name', 'tokens', 'label']].reset_index(drop = True),
	           pd.DataFrame({'score': valid_yhat_prob_is_brand})
	               ]
	         , axis = 1).rename(columns = {'label':'y_true'})

	gc.collect()
	# get the best threshould
	bestdf, best_th, best_performance = get_best_threshold(df, interval = 0.0001, measurement = 'f1')

	logging.info('best_performance: {} - {}'.format(best_performance, name))
	logging.info('best_th: {} - {}'.format(best_th, name))
	#--------------------------
	# evaluating the system performace
	#--------------------------

	df1 = bestdf.groupby('item_name') \
	.apply(lambda x: get_positive_token(x, flag = 'atual_brand')) \
	.to_frame('acutal_brand').reset_index()

	df2 = bestdf.groupby('item_name') \
	.apply(lambda x: get_positive_token(x, flag = 'predicted_brand')) \
	.to_frame('predicted_brand').reset_index()

	# output
	evaluation = pd.merge(df1, df2, on = 'item_name', how = 'left')[['item_name', 'acutal_brand','predicted_brand']]
	f1, r, p, accuracy = evaluating_system(evaluation)

	logging.info('accuracy: {} - {}'.format(accuracy, name))
	logging.info('f1: {} - {}'.format(f1, name))
	logging.info('precision: {} - {}'.format(p, name))
	logging.info('recall: {} - {}'.format(r, name))

	del df1, df2
	gc.collect()

def multi(T):
	'''
	It's for using multi preprosessing to speed up each model training process.

	parameters:
	---------------------
	T: int. 1 or 2 for telling difference between tv_and_laptop and personal_care_and_beauty.
	'''
	evaluate(T)

if __name__ == '__main__':

	##################################################
	# Main
	##################################################
	s = time.time()

	mp_pool = mp.Pool(4)
	mp_pool.map(multi, [1, 2, 3, 4])

	e = time.time()
	print (e-s, ' time')





