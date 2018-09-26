#! /usr/bin/env python3
'''
It's built for creating models for each brand and predict.

Each brand has a one individual model.

Created on July 20 2018

@author: Ray
'''

import pandas as pd
import numpy as np
import os
import gc
import time
from xgboost import plot_importance
from xgboost import XGBClassifier
from utils import init_logging
import logging
import multiprocessing as mp # for speeding up some process
import pickle
import warnings
from datetime import datetime # for the newest version control
warnings.simplefilter(action='ignore', category=FutureWarning)

def build_model(T):
	'''
	It's for using multi preprosessing to speed up concating feature process.

	parameters:
	---------------------
	T: int. 1, 2, ..
	'''
	##################################################
	# Step1: loading features
	##################################################

	#--------------------
	# path setting
	#--------------------
	seed = 1030
	np.random.seed(seed)
	log_dir = '../log'
	init_logging(log_dir)
	output_dir = '../features/'
	#--------------------
	# laod data including label
	#--------------------	
	if T == 1:
		name = 'tv_laptop_amazon' 
		df = pd.read_hdf(os.path.join(output_dir, '{}/all_features.h5'.format(name)))
		# feat = [f for f in df.columns.tolist() if 'pos' not in f]
		# df = df[feat]
	elif T == 2:
		name = 'beauty_amazon'
		df = pd.read_hdf(os.path.join(output_dir, '{}/all_features.h5'.format(name)))
	elif T == 3:
		name = 'tv_and_laptop'
		df = pd.read_hdf(os.path.join(output_dir, '{}/all_features.h5'.format(name)))
		# feat = [f for f in df.columns.tolist() if 'pos' not in f]
		# df = df[feat]
	elif T == 4:
		name = 'personal_care_and_beauty'
		df = pd.read_hdf(os.path.join(output_dir, '{}/all_features.h5'.format(name)))
	else:
		pass
	#----------------------------------
	# prepare training data
	#----------------------------------

	# hold-out methold
	train = df[df.is_valid == 0]
	val = df[(df.is_valid == 1)]

	logging.info('num_training_sample - {} {}'.format(name, train.shape[0]))
	logging.info('num_validating_sample - {} {}'.format(name, val.shape[0]))

	# hold-out train/val split for base model
	x_train = train.drop(['item_name', 'tokens', 'label', 'is_valid'], axis = 1)
	y_train = train.label
	x_val = val.drop(['item_name', 'tokens', 'label', 'is_valid'], axis = 1)
	y_val = val.label

	##################################################
	# Step2: model training
	##################################################

	#----------------------------------
	# base_model for getting feature importance as thereshold
	#----------------------------------
	s = time.time()

	base_model = XGBClassifier(objective = 'binary:logistic',
	                           seed = seed,
	                           n_estimators = 50000,
	                           learning_rate = 0.001,
	                           n_jobs = int(mp.cpu_count() * CPU_USE_RATE),
	                           tree_method = 'hist',
	                      )
	# print (x_train.columns)
	base_model.fit(x_train, y_train, 
	          eval_metric ='mlogloss' ,
	          eval_set = [(x_val, y_val)],
	          verbose = True,
	          early_stopping_rounds = 100)

	e = time.time()

	logging.info('base_model tooks {} mins / {}'.format((e-s)/ 60.0,  name))
	#-----------------------------------
	# saving model
	#-----------------------------------

	# level 1
	output_dir = '../output'
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	# level 2
	date_str = datetime.now().strftime('%Y-%m-%d')
	model_dir = '../output/model'
	output_dir = os.path.join(model_dir, date_str)
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)

	pickle.dump(base_model, open(os.path.join(output_dir, '{}_xgb.model'.format(name)), "wb"))
	logging.info('best logloss on val : {} / {}'.format(min(base_model.evals_result()['validation_0']['mlogloss']), name))

	# ##################################################
	# # Step3: model prediction and evaluation on amazon dataset
	# ##################################################

	#------------------------
	# validating set prediction
	#------------------------
	valid_yhat_prob_is_brand = pd.Series(np.argmax(model.predict_proba(x_val), axis=1))
	logging.info('prediction distribution : {} / {}'.format(valid_yhat_prob_is_brand.value_counts(), name))

	# logging.info('valid Mean: {} - {}'.format(np.mean(valid_yhat_prob_is_brand), name))

	# for_val_concat = pd.DataFrame({'prob_that_is_brand': valid_yhat_prob_is_brand})

	# #------------------------
	# # combine the prediction result of val and test for output evaluation metric
	# #------------------------
	# # val
	# val.reset_index(drop=True, inplace=True)
	# for_val_concat.reset_index(drop=True, inplace=True)
	# output_column = ['item_name','tokens','label','prob_that_is_brand']
	# output_val = pd.concat([val, for_val_concat], axis = 1)[output_column]

	# #----------------------------------
	# # check train/val/test features distribution
	# #----------------------------------

	# # prepare feature names
	# feature_names = list(train.columns)
	# do_not_use_for_training = ['item_name', 'tokens', 'label', 'is_valid']
	# feature_names = [f for f in train.columns if f not in do_not_use_for_training]

	# logging.info('feature_names : {}'.format(feature_names))
	# # create feature distribution diff
	# feature_stats = pd.DataFrame({'feature': feature_names})
	# feature_stats.loc[:, 'train_mean'] = np.nanmean(train[feature_names].values, axis=0).round(4)
	# feature_stats.loc[:, 'val_mean'] = np.nanmean(val[feature_names].values, axis=0).round(4)
	# feature_stats.loc[:, 'train_std'] = np.nanstd(train[feature_names].values, axis=0).round(4)
	# feature_stats.loc[:, 'val_std'] = np.nanstd(val[feature_names].values, axis=0).round(4)
	# feature_stats.loc[:, 'train_val_mean_diff'] = np.abs(feature_stats['train_mean'] - feature_stats['val_mean']) / np.abs(feature_stats['train_std'] + feature_stats['val_std'])  * 2
	# # check missing value in the features
	# feature_stats.loc[:, 'train_nan'] = np.mean(np.isnan(train[feature_names].values), axis=0).round(3)
	# feature_stats.loc[:, 'val_nan'] = np.mean(np.isnan(val[feature_names].values), axis=0).round(3)
	# feature_stats.loc[:, 'train_val_nan_diff'] = np.abs(feature_stats['train_nan'] - feature_stats['val_nan'])

	# logging.info('train_val_mean_diff on average: {} - {}'.format(feature_stats.train_val_mean_diff.mean(), name))

	#----------------------------------
	# analyzing the model predictive power on amazon dataset
	#----------------------------------

	# num_unseen_brands
	seen_brand_set = set(df[(df.label == 1) & (df.is_valid == 0)].tokens.unique())
	validating_brand_set = set(df[(df.label == 1) & (df.is_valid == 1)].tokens.unique())
	unseen_brand_set = validating_brand_set - seen_brand_set
	logging.info(' how many brands that does not exist in the training set : {} - {}'.format(len(unseen_brand_set), name))
	# # ratio_unseen_brands_predicted
	# prediction_val = output_copy
	# prediction_all = pd.concat([prediction_val], axis = 0)
	# prediction_all = prediction_all.groupby(by = 'item_name').apply(lambda x : x[x.label == 1]).reset_index(drop = True)
	# correct_prediction = prediction_all[(prediction_all.label == 1)& (prediction_all.system_prediction_result == 1)]
	# predicted_suc_set = set(correct_prediction.tokens.unique())
	# del correct_prediction, prediction_all, prediction_val
	# gc.collect()
	# intsection = predicted_suc_set.intersection(unseen_brand_set)
	# brand_cannot_detect = unseen_brand_set - predicted_suc_set
	# ratio_unseen_brands_predicted = len(intsection) / len(unseen_brand_set)
	# logging.info('the ratio that unseen brands were successfully predicted by the model : {} - {}'.format(np.round(ratio_unseen_brands_predicted, 4), name))
	# logging.info('brand_cannot_detect : {} - {}'.format(brand_cannot_detect, name))

def multi(T):
	'''
	It's for using multi preprosessing to speed up each model training process.

	parameters:
	---------------------
	T: int. 1, 2, ... for telling difference between tv_and_laptop and personal_care_and_beauty.
	'''
	build_model(T)

if __name__ == '__main__':
	##################################################
	# Main
	##################################################
	s = time.time()

	CPU_USE_RATE = 0.5
	mp_pool = mp.Pool(int(mp.cpu_count() * CPU_USE_RATE))
	mp_pool.map(multi, [1,2,3,4])

	e = time.time()
	print (e-s, ' time')
