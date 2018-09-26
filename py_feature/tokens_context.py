#! /usr/bin/env python3

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime # for the newest version control
from collections import Counter # for statistical features
from nltk import tag # for pos_tagging
from nltk.corpus import wordnet # for geting pos of wordnet
from nltk.stem import WordNetLemmatizer

def position_of_the_tokens(row):
    '''
    Returning positon of the token in the item_name
    '''
    try:
        return row.item_name.split().index(row.tokens) + 1
    except Exception:
        pass # It will make missing value on this feature but it's fine

def is_first_token_in_item_name(row):
    '''
    Check if the token is the first token in the itemname.
    '''
    list_of_tokens = row.item_name.split()
    try:
        if list_of_tokens.index(row.tokens) == 0:
            return 1
        else:
            return 0
    except Exception:
    	print ('tokens', row.tokens)
    	print ('list_of_tokens', list_of_tokens)

def is_second_token_in_item_name(row):
    '''
    Check if the token is the second token in the itemname.
    '''
    list_of_tokens = row.item_name.split()
    if list_of_tokens.index(row.tokens) == 1:
        return 1
    else:
        return 0

def position_of_the_tokens_from_the_bottom(row):
    '''
    Returning positon of the token in the item_name
    '''
    try:
        item_name_ls = row.item_name.split()
        item_name_ls.reverse()
        return item_name_ls.index(row.tokens) + 1
    except Exception:
        pass # It will make missing value on this feature but it's fine

def position_of_the_tokens_in_ratio(row):
    '''
    Returning positon of the token in the item_name
    '''
    try:
        return 1.0 * row.item_name.split().index(row.tokens) / len(row.item_name.split())
    except Exception:
        pass # It will make missing value on this feature but it's fine

def len_of_item_name(row):
	'''
	return how many tokens we have given a item name. Maybe the itemname is unbranded, 
	the length of them is shorter.
	'''
	try:
		return len(row.item_name.split())
	except Exception:
		pass # It will make missing value on this feature but it's fine

def one_hot_encoder(df, ignore_feature, nan_as_category = True):
    '''
    It's helper function for pos_tagger to do One-hot encoding for categorical columns with get_dummies.

    paras:
    ----------------
    ignore_feature: list of string.
    nan_as_category: boolean. If we think of nan as a value of a certain field.
    '''
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    categorical_columns = [col for col in categorical_columns if col not in ignore_feature]
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def pos_tagger(df):
    '''
    High-level pos-tagging
    '''
    try:
        tagged_sent = tag.pos_tag(df.tokens.tolist())
    except Exception:
        print (df.item_name.iloc[0])
        print (df.tokens.tolist())
    df['pos_tagger'] = [pos for token, pos in tagged_sent]
    df['pos_tagger'] = [pos.replace("''", '$') for pos in df.pos_tagger.tolist()]
    return df

def get_wordnet_pos(treebank_tag):
    '''
    It map the treebank tags to WordNet part of speech names
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN

def make_word_idx(item_names):
    '''
    It's a helper function for encode_text.
    
    Return a dict including count of each token happening in the whole dataset.
    
    parameters:
    ------------
    item_names: list, including all tokens in the complete dataset
    
    
    '''
    words = [word.lower() for word in item_names]
    word_counts = Counter(words)

    max_id = 1
    
    word_idx = {}
    for word, count in word_counts.items():
        if count < 3:
            word_idx[word] = 0
        else:
            word_idx[word] = max_id
            max_id += 1

    return word_idx

def encode_text(text, word_idx):
    '''
    encode token into code, thinkg of this as a term freq
    High frequent words usually are noise word.
    
    parameters:
    --------------
    text: str
    word_idx
    '''
    return int(word_idx[text.lower()])

if __name__ == '__main__':
	LEMMATIZING = False
	#--------------------------
	# loading data
	#--------------------------
	# preprocessed_data_path
	input_base_path = '../data/preprocessed'
	# shopee
	tv_and_laptop = pd.read_csv(os.path.join(input_base_path, 'tv_and_laptop.csv'))
	personal_care_and_beauty = pd.read_csv(os.path.join(input_base_path, 'personal_care_and_beauty.csv'))
	# amazon
	beauty_amazon = pd.read_csv(os.path.join(input_base_path, 'beauty_amazon.csv'))
	tv_laptop_amazon = pd.read_csv(os.path.join(input_base_path, 'tv_laptop_amazon.csv'))

	#-------------------------
	# drop itemname and tokens with nan
	#-------------------------
	tv_and_laptop.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)
	personal_care_and_beauty.dropna(subset = ['item_name','tokens'], axis = 0, inplace = True)
	beauty_amazon.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)
	tv_laptop_amazon.dropna(subset = ['item_name', 'tokens'], axis = 0, inplace = True)

	#--------------------------
	# conver type
	#--------------------------
	tv_and_laptop['tokens'] = tv_and_laptop.tokens.astype(str)
	personal_care_and_beauty['tokens'] = personal_care_and_beauty.tokens.astype(str)
	beauty_amazon['tokens'] = beauty_amazon.tokens.astype(str)
	tv_laptop_amazon['tokens'] = tv_laptop_amazon.tokens.astype(str)

	#--------------------------
	# create word_idx for token_freq features
	#--------------------------
	if LEMMATIZING == True:
		print ('LEMMATIZING : {}'.format(LEMMATIZING))
		lemmatizer = WordNetLemmatizer()
		tv_and_laptop = tv_and_laptop.groupby('item_name').apply(pos_tagger)
		personal_care_and_beauty = personal_care_and_beauty.groupby('item_name').apply(pos_tagger)
		beauty_amazon = beauty_amazon.groupby('item_name').apply(pos_tagger)
		tv_laptop_amazon = tv_laptop_amazon.groupby('item_name').apply(pos_tagger)
		# get_wordnet_pos
		tv_and_laptop['wordnet_pos'] = tv_and_laptop.pos_tagger.apply(get_wordnet_pos)
		personal_care_and_beauty['wordnet_pos'] = personal_care_and_beauty.pos_tagger.apply(get_wordnet_pos)
		beauty_amazon['wordnet_pos'] = beauty_amazon.pos_tagger.apply(get_wordnet_pos)
		tv_laptop_amazon['wordnet_pos'] = tv_laptop_amazon.pos_tagger.apply(get_wordnet_pos)
		# lemmatizing
		tv_and_laptop['lemma'] = [lemmatizer.lemmatize(t.lower(), pos) for t, pos in zip(tv_and_laptop.tokens.tolist(), tv_and_laptop.wordnet_pos.tolist())]
		personal_care_and_beauty['lemma'] = [lemmatizer.lemmatize(t.lower(), pos) for t, pos in zip(personal_care_and_beauty.tokens.tolist(), personal_care_and_beauty.wordnet_pos.tolist())]
		beauty_amazon['lemma'] = [lemmatizer.lemmatize(t.lower(), pos) for t, pos in zip(beauty_amazon.tokens.tolist(), beauty_amazon.wordnet_pos.tolist())]
		tv_laptop_amazon['lemma'] = [lemmatizer.lemmatize(t.lower(), pos) for t, pos in zip(tv_laptop_amazon.tokens.tolist(), tv_laptop_amazon.wordnet_pos.tolist())]
		# drop columns that we don't need it
		tv_and_laptop.drop(['wordnet_pos'], axis = 1, inplace = True)
		personal_care_and_beauty.drop(['wordnet_pos'], axis = 1, inplace = True)
		beauty_amazon.drop(['wordnet_pos'], axis = 1, inplace = True)
		tv_laptop_amazon.drop(['wordnet_pos'], axis = 1, inplace = True)
		# make word_index
		word_idx_for_tv_and_laptop = make_word_idx(pd.concat([tv_and_laptop], axis = 0).lemma.tolist())
		word_idx_for_personal_care_and_beauty = make_word_idx(pd.concat([personal_care_and_beauty], axis = 0).lemma.tolist())
		word_idx_for_beauty_amazon = make_word_idx(pd.concat([beauty_amazon], axis = 0).lemma.tolist())
		word_idx_for_tv_laptop_amazon = make_word_idx(pd.concat([tv_laptop_amazon], axis = 0).lemma.tolist())
	else:
		word_idx_for_tv_and_laptop = make_word_idx(pd.concat([tv_and_laptop], axis = 0).tokens.tolist())
		word_idx_for_personal_care_and_beauty = make_word_idx(pd.concat([personal_care_and_beauty], axis = 0).tokens.tolist())
		word_idx_for_beauty_amazon = make_word_idx(pd.concat([beauty_amazon], axis = 0).tokens.tolist())
		word_idx_for_tv_laptop_amazon = make_word_idx(pd.concat([tv_laptop_amazon], axis = 0).tokens.tolist())

	##################################
	# feature engineering
	##################################

	#-------------------------
	# tv_and_laptop /shopee
	#-------------------------
	# position_of_the_tokens_from_the_bottom
	tv_and_laptop['position_of_the_tokens_from_the_bottom'] = tv_and_laptop.apply(position_of_the_tokens_from_the_bottom, axis = 1)
	# position of the tokens
	tv_and_laptop['position_of_the_tokens'] = tv_and_laptop.apply(position_of_the_tokens, axis = 1)
	if LEMMATIZING == True:
		# token_freq
		tv_and_laptop['token_freq'] = tv_and_laptop['lemma'].map(lambda x: encode_text(x, word_idx_for_tv_and_laptop))
		tv_and_laptop.drop(['lemma'], axis = 1, inplace = True)
	else:	
		# token_freq
		tv_and_laptop['token_freq'] = tv_and_laptop['tokens'].map(lambda x: encode_text(x, word_idx_for_tv_and_laptop))
		# pos tagger
		tv_and_laptop = tv_and_laptop.groupby('item_name').apply(pos_tagger)
	# position_of_the_tokens_in_ratio
	tv_and_laptop['position_of_the_tokens_in_ratio'] = tv_and_laptop.apply(position_of_the_tokens_in_ratio, axis = 1)
	# len_of_item_name
	tv_and_laptop['len_of_item_name'] = tv_and_laptop.apply(len_of_item_name, axis = 1)

	tv_and_laptop, cat_cols_tv_shopee = one_hot_encoder(tv_and_laptop, 
                                          nan_as_category = False, 
                                          ignore_feature = ['item_name', 'tokens', 'is_brand', 'is_valid'])
	tv_and_laptop['is_first_token_in_item_name'] = tv_and_laptop.apply(is_first_token_in_item_name, axis = 1)
	# is_second_token_in_item_name
	tv_and_laptop['is_second_token_in_item_name'] = tv_and_laptop.apply(is_second_token_in_item_name, axis = 1)
	
	#-------------------------
	# personal_care_and_beauty /shopee
	#-------------------------
	# position_of_the_tokens_from_the_bottom
	personal_care_and_beauty['position_of_the_tokens_from_the_bottom'] = personal_care_and_beauty.apply(position_of_the_tokens_from_the_bottom, axis = 1)
	# position of the tokens
	personal_care_and_beauty['position_of_the_tokens'] = personal_care_and_beauty.apply(position_of_the_tokens, axis = 1)
	if LEMMATIZING == True:
		# token_freq
		personal_care_and_beauty['token_freq'] = personal_care_and_beauty['lemma'].map(lambda x: encode_text(x, word_idx_for_personal_care_and_beauty))
		personal_care_and_beauty.drop(['lemma'], axis = 1, inplace = True)
	else:
		# token_freq
		personal_care_and_beauty['token_freq'] = personal_care_and_beauty['tokens'].map(lambda x: encode_text(x, word_idx_for_personal_care_and_beauty))	
		# pos tagger
		personal_care_and_beauty = personal_care_and_beauty.groupby('item_name').apply(pos_tagger)
	
	# position_of_the_tokens_in_ratio
	personal_care_and_beauty['position_of_the_tokens_in_ratio'] = personal_care_and_beauty.apply(position_of_the_tokens_in_ratio, axis = 1)
	#len_of_item_name
	personal_care_and_beauty['len_of_item_name'] = personal_care_and_beauty.apply(len_of_item_name, axis = 1)
	personal_care_and_beauty, cat_cols_beauty_shopee = one_hot_encoder(personal_care_and_beauty, 
                                          nan_as_category = False, 
                                          ignore_feature = ['item_name', 'tokens', 'is_brand', 'is_valid'])
	personal_care_and_beauty['is_first_token_in_item_name'] = personal_care_and_beauty.apply(is_first_token_in_item_name, axis = 1)
	# is_second_token_in_item_name
	personal_care_and_beauty['is_second_token_in_item_name'] = personal_care_and_beauty.apply(is_second_token_in_item_name, axis = 1)

	#-------------------------
	# tv_and_laptop /amazone
	#-------------------------
	# position_of_the_tokens_from_the_bottom
	tv_laptop_amazon['position_of_the_tokens_from_the_bottom'] = tv_laptop_amazon.apply(position_of_the_tokens_from_the_bottom, axis = 1)
	# position of the tokens
	tv_laptop_amazon['position_of_the_tokens'] = tv_laptop_amazon.apply(position_of_the_tokens, axis = 1)
	if LEMMATIZING == True:
		# token_freq
		tv_laptop_amazon['token_freq'] = tv_laptop_amazon['lemma'].map(lambda x: encode_text(x, word_idx_for_tv_laptop_amazon))
		tv_laptop_amazon.drop(['lemma'], axis = 1, inplace = True)
	else:
		# token_freq
		tv_laptop_amazon['token_freq'] = tv_laptop_amazon['tokens'].map(lambda x: encode_text(x, word_idx_for_tv_laptop_amazon))
		# pos tagger
		tv_laptop_amazon = tv_laptop_amazon.groupby('item_name').apply(pos_tagger)

	# position_of_the_tokens_in_ratio
	tv_laptop_amazon['position_of_the_tokens_in_ratio'] = tv_laptop_amazon.apply(position_of_the_tokens_in_ratio, axis = 1)
	# len_of_item_name
	tv_laptop_amazon['len_of_item_name'] = tv_laptop_amazon.apply(len_of_item_name, axis = 1)
	tv_laptop_amazon, cat_cols_tv_amazon = one_hot_encoder(tv_laptop_amazon, 
                                          nan_as_category = False, 
                                          ignore_feature = ['item_name', 'tokens', 'is_brand', 'is_valid'])
	tv_laptop_amazon['is_first_token_in_item_name'] = tv_laptop_amazon.apply(is_first_token_in_item_name, axis = 1)
	# is_second_token_in_item_name
	tv_laptop_amazon['is_second_token_in_item_name'] = tv_laptop_amazon.apply(is_second_token_in_item_name, axis = 1)

	#-------------------------
	# personal_care_and_beauty /amazon
	#-------------------------
	# position_of_the_tokens_from_the_bottom
	beauty_amazon['position_of_the_tokens_from_the_bottom'] = beauty_amazon.apply(position_of_the_tokens_from_the_bottom, axis = 1)
	# position of the tokens
	beauty_amazon['position_of_the_tokens'] = beauty_amazon.apply(position_of_the_tokens, axis = 1)
	if LEMMATIZING == True:
		# token_freq
		beauty_amazon['token_freq'] = beauty_amazon['lemma'].map(lambda x: encode_text(x, word_idx_for_beauty_amazon))
		beauty_amazon.drop(['lemma'], axis = 1, inplace = True)
	else:		
		# token_freq
		beauty_amazon['token_freq'] = beauty_amazon['tokens'].map(lambda x: encode_text(x, word_idx_for_beauty_amazon))
		# pos tagger
		beauty_amazon = beauty_amazon.groupby('item_name').apply(pos_tagger)

	# position_of_the_tokens_in_ratio
	beauty_amazon['position_of_the_tokens_in_ratio'] = beauty_amazon.apply(position_of_the_tokens_in_ratio, axis = 1)
	#len_of_item_name
	beauty_amazon['len_of_item_name'] = beauty_amazon.apply(len_of_item_name, axis = 1)
	beauty_amazon, cat_cols_beauty_amazon = one_hot_encoder(beauty_amazon, 
                                          nan_as_category = False, 
                                          ignore_feature = ['item_name', 'tokens', 'is_brand', 'is_valid'])
	beauty_amazon['is_first_token_in_item_name'] = beauty_amazon.apply(is_first_token_in_item_name, axis = 1)
	# is_second_token_in_item_name
	beauty_amazon['is_second_token_in_item_name'] = beauty_amazon.apply(is_second_token_in_item_name, axis = 1)

	#-------------------------
	# use features intersection of amazon and shoppp, since we wanna employ transfer learning
	#-------------------------
	# Step0 : get intersection of features
	pos_feature_in_tv = list(set(cat_cols_tv_shopee) & set(cat_cols_tv_amazon))
	pos_feature_in_beauty = list(set(cat_cols_beauty_shopee) & set(cat_cols_beauty_amazon))
	impute = tv_and_laptop[['item_name', 'tokens'] + pos_feature_in_tv].copy()

	# step1: get all features including 'pos_', later on we will drop them
	drop_pos_tv_shopee = [f for f in tv_and_laptop.columns.tolist() if 'pos_' in f]
	drop_pos_beauty_shopee = [f for f in personal_care_and_beauty.columns.tolist() if 'pos_' in f]
	drop_pos_tv_amazon = [f for f in tv_laptop_amazon.columns.tolist() if 'pos_' in f]
	drop_pos_beauty_amazon = [f for f in beauty_amazon.columns.tolist() if 'pos_' in f]
	# step2: impute all features with intersection
	impute_tv_shopee = tv_and_laptop[['item_name', 'tokens'] + pos_feature_in_tv].copy()
	impute_beauty_shopee = personal_care_and_beauty[['item_name', 'tokens'] + pos_feature_in_beauty].copy()
	impute_tv_amazon = tv_laptop_amazon[['item_name', 'tokens'] + pos_feature_in_tv].copy()
	impute_beauty_amazon = beauty_amazon[['item_name', 'tokens'] + pos_feature_in_beauty].copy()

	tv_and_laptop.drop(drop_pos_tv_shopee, axis = 1, inplace = True)
	personal_care_and_beauty.drop(drop_pos_beauty_shopee, axis = 1, inplace = True)
	tv_laptop_amazon.drop(drop_pos_tv_amazon, axis = 1, inplace = True)
	beauty_amazon.drop(drop_pos_beauty_amazon, axis = 1, inplace = True)
	# step3: merge back
	tv_and_laptop = pd.merge(tv_and_laptop, impute_tv_shopee, on = ['item_name', 'tokens'], how = 'left')
	personal_care_and_beauty = pd.merge(personal_care_and_beauty, impute_beauty_shopee, on = ['item_name', 'tokens'], how = 'left')
	tv_laptop_amazon = pd.merge(tv_laptop_amazon, impute_tv_amazon, on = ['item_name', 'tokens'], how = 'left')
	beauty_amazon = pd.merge(beauty_amazon, impute_beauty_amazon, on = ['item_name', 'tokens'], how = 'left')
	#-------------------------
	# remove no need columns
	#-------------------------
	tv_and_laptop.drop(['is_brand', 'is_valid'], axis = 1 , inplace = True)
	personal_care_and_beauty.drop(['is_brand', 'is_valid'], axis = 1 , inplace = True)
	tv_laptop_amazon.drop(['is_brand', 'is_valid'], axis = 1 , inplace = True)
	beauty_amazon.drop(['is_brand', 'is_valid'], axis = 1 , inplace = True)


	#-------------------------
	# save
	#-------------------------
	tv_and_laptop.to_csv('../features/{}/tokens_context.csv'.format('tv_and_laptop'), index = False)
	personal_care_and_beauty.to_csv('../features/{}/tokens_context.csv'.format('personal_care_and_beauty'), index = False)
	tv_laptop_amazon.to_csv('../features/{}/tokens_context.csv'.format('tv_laptop_amazon'), index = False)
	beauty_amazon.to_csv('../features/{}/tokens_context.csv'.format('beauty_amazon'), index = False)

