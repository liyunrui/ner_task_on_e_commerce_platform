#! /usr/bin/env python3
'''
Created on July 9 2018

@author: Ray

Reference: Attribute Extraction from Product Titles in eCommerce.
'''
import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime # for the newest version control
import sys
sys.path.append('../py_model/')
from utils import init_logging
import logging
import warnings
warnings.simplefilter(action='ignore')
from clean_helpers import clean_name

def brand_split_validating_strategy(df, train_val_rate = 0.8, dataset_source = 'Amazon'):
    '''
    It's helper function for spliting df into training and validating dataset 
    based on unique brand because we hope our model can detect those unseen brands.
    
    parameters:
    -----------------
    df: DataFrame
    train_val_rate: float, where range is (0,1]
    dataset_source: str, for telling/differentiating different dataset
    '''
    if dataset_source == 'Amazon':
        #--------------
        # create a dataframe for splitting
        #--------------  
        brand_stat = df.what_brand_name.value_counts().to_frame('count') \
        .reset_index().rename(columns = {'index':'what_brand_name'})
        brand_stat['percent'] =  brand_stat['count'] / brand_stat['count'].sum()
        #--------------
        # core
        #--------------
        training_brand = []
        validating_brand = []
        cumulative_percent = 0 
        for ix, row in brand_stat.sample(frac = 1.0).iterrows():
            # shuffle the dataframe, for avoiding our training data all are big brands.
            if row['what_brand_name'] != 'Others':
                cumulative_percent += row['percent']
                if cumulative_percent <= train_val_rate:
                    training_brand.append(row['what_brand_name'])
                else:
                    validating_brand.append(row['what_brand_name']) 

        #--------------
        # unit testing for if there is no same brand name in the train and val set.
        #--------------
        percent_train = brand_stat[brand_stat.what_brand_name.isin(training_brand)].percent.sum()
        unique_brand_train = set(brand_stat[brand_stat.what_brand_name.isin(training_brand)].what_brand_name.tolist())
        percent_val = brand_stat[brand_stat.what_brand_name.isin(validating_brand)].percent.sum()
        unique_brand_val = set(brand_stat[brand_stat.what_brand_name.isin(validating_brand)].what_brand_name.tolist())
        print ('no bugging in unit testing' if unique_brand_val.intersection(unique_brand_train) == set() else 'Opps, there are same brand name in train and val.')
        #--------------
        # output
        #--------------    
        df1 = df[df.what_brand_name.isin(training_brand)]
        df2 = df[df.what_brand_name.isin(validating_brand)]
        df1['is_valid'] = 0
        df2['is_valid'] = 1
        # combination
        df = pd.concat([df1, df2], axis = 0)
        del df1, df2, brand_stat, training_brand, validating_brand
        gc.collect() 
    elif dataset_source == 'Shopee':
        #--------------
        # create a dataframe for splitting
        #--------------  
        brand_stat = df.what_brand_name.value_counts().to_frame('count') \
        .reset_index().rename(columns = {'index':'what_brand_name'})
        brand_stat['percent'] =  brand_stat['count'] / brand_stat['count'].sum()
        #--------------
        # core
        #--------------
        # initialize variables
        testing_brand = []
        training_brand = []
        validating_brand = []
        cumulative_percent = 0 
        # shuffle the dataframe, for avoiding our training data all are big brands.
        for ix, row in brand_stat.sample(frac = 1.0).iterrows():
            cumulative_percent += row['percent']
            if cumulative_percent <= 0.5:
                # take 0.5 as testing_brand
                testing_brand.append(row['what_brand_name'])
            elif 0.5 < cumulative_percent <= 0.85:
                # take 0.35 as training_brand
                training_brand.append(row['what_brand_name'])
            else:
                # take the rest 0.15 as validating_brand
                validating_brand.append(row['what_brand_name'])
        #--------------
        # unit testing
        #--------------
        percent_train = brand_stat[brand_stat.what_brand_name.isin(training_brand)].percent.sum()
        unique_brand_train = set(brand_stat[brand_stat.what_brand_name.isin(training_brand)].what_brand_name.tolist())
        percent_val = brand_stat[brand_stat.what_brand_name.isin(validating_brand)].percent.sum()
        unique_brand_val = set(brand_stat[brand_stat.what_brand_name.isin(validating_brand)].what_brand_name.tolist())
        print ('no bugging in unit testing' if unique_brand_val.intersection(unique_brand_train) == set() else 'Opps, there are same brand name in train and val.')
        #--------------
        # output
        #--------------    
        df1 = df[df.what_brand_name.isin(testing_brand)]
        df2 = df[df.what_brand_name.isin(training_brand)]
        df3 = df[df.what_brand_name.isin(validating_brand)]

        df1['is_valid'] = 2 # 2:testing
        df2['is_valid'] = 0 # 0:training
        df3['is_valid'] = 1 # 1:validating

        # combination
        df = pd.concat([df1, df2, df3], axis = 0)
        del df1, df2, df3, brand_stat, training_brand, validating_brand   
        gc.collect()
    else:
        print ('FUCK YOU, the dataset u provided is WRONG, please usee Amazon or Shopee')

    return df


def sequence_labeling_w_bio_encoding(row):
    '''
    BIO encoding is a distant supervision approach to automatically generate training data for training machine- earning based model. 
    
    Reference for distant supervision approach: http://deepdive.stanford.edu/distant_supervision
    Reference for BIO : Attribute Extraction from Product Titles in eCommerce.
    Assumption:
        - We assume that one sku only has one brand name.(kind of non-realistic)
    parameters:
    --------------
    df: DataFrame
    if_assumption: str. if True, we assume we only have one-single brand_word in one item_name. 
    Otherwise, we can have multiple token with positive lable in one item_name.
    '''

    # initialize variables
    word_list = []
    tagging = [] # multi-class label, {0:not part of the brand name, 1: intermediate part of the brand name, 2:beginning of the brand name}
    item_name = []
    val = [] 
    #---------------
    # sequential labeling with BIO encoding
    #---------------
    brand_started = False
    b_ix = 0
    brand = row.what_brand_name.iloc[0].split(' ')
    title = clean_name(row['item_name'].iloc[0]).split(' ')
    # filter
    title = [t for t in title if '' != t]
    for w_ix, word in enumerate(title):
        if word.lower() == brand[0].lower():
            tagging.append(2) # B-B: 2
            brand_started = True
            b_ix += 1
        elif (len(brand) > 1) and (brand_started):
            if b_ix >= len(brand):
                # for avoiding . For example, if 'BUMBLE AND BUMBLE by Bumble and Bumble: QUENCHING CONDITIONER 8.5 OZ'
                tagging.append(0) # O: 0
                brand_started = False  
                b_ix = 0                
            else:
                if word.lower() == brand[b_ix].lower():
                    tagging.append(1) # I-B: 1
                    b_ix += 1
                    if b_ix == len(brand):
                        # go back to orginal state because we already marked what we want
                        brand_started = False
                        b_ix = 0
                else:
                    tagging.append(0) # O: 0
                    brand_started = False     
                    # if we need to modified the labeling we priviously marked.
                    if b_ix < len(brand):
                        go_back_to_modified = 0
                        for i in range(b_ix):
                            #print ('w_ix', w_ix) # w_ix 對應的不是整個 tagging的list: 兩個解法, 1.groupby 2.w_ix要一直被加上
                            go_back_to_modified += 1
                            #print ('go back', w_ix - go_back_to_modified)
                            tagging[w_ix - go_back_to_modified] = 0 # O: 0
                        # Once removing privous labeling, we update b_ix to zero
                        b_ix = 0         
        else:
            brand_started = False
            tagging.append(0) # O: 0
        #---------------------------
        # for output dataframe
        #---------------------------
        if NORMALIZED == True:
            word_list.append(word.lower())
        else:
            word_list.append(word)
        item_name.append(clean_name(row['item_name'].iloc[0]))
        val.append(row['is_valid'].iloc[0])
    #---------------------------
    # output
    #---------------------------
    df = pd.DataFrame({'tokens':word_list, 
                'is_brand': tagging,
                'is_valid': val,
                'item_name':item_name})[['item_name','tokens','is_brand','is_valid']]
    return df

if __name__ == '__main__':
    #--------------------------
    # setting
    #--------------------------
    seed = 1030
    NORMALIZED = False
    np.random.seed(seed)

    # log path
    log_dir = 'log/'
    init_logging(log_dir)

    #--------------------------
    # loading data (TV and Laptop) / shopee
    #--------------------------
    # Note the below
    bath_path_1 = '../../../grouping/tv_and_laptop_grouping/output/laptop/2018-08-07/'
    bath_path_2 = '../../../grouping/tv_and_laptop_grouping/output/tv/2018-08-07/'
    df_laptop = pd.read_csv(os.path.join(bath_path_1,'TH-laptop_for_brand_detector.csv'))
    df_tv = pd.read_csv(os.path.join(bath_path_2,'TH-TV_for_brand_detector.csv'))
    # cobine tv and laptop data
    df_all = pd.concat([df_laptop, df_tv], axis = 0)
    # take all the sku that we can find brand from catalogue list as train
    tv_and_laptop = df_all[df_all.if_tokens_of_cleaned_name_is_in_raw_brand == 1]
    # cleaning duplicated item_name
    tv_and_laptop = tv_and_laptop.drop_duplicates(subset = ['item_name'])
    # make the index unique
    tv_and_laptop.index = np.arange(tv_and_laptop.shape[0]) 
    # convert type into str
    tv_and_laptop.item_name = tv_and_laptop.item_name.astype(str)
    tv_and_laptop.what_brand_name = tv_and_laptop.what_brand_name.astype(str)
    # clean memory
    del df_laptop, df_tv, df_all
    gc.collect()

    #--------------------------
    # loading data (personal care and beauty) / shopee
    #--------------------------

    # data path
    bath_path_beauty = '../../../grouping/tv_and_laptop_grouping/raw_data/beauty_personal_care/'
    sheet_name = 'Shopee input'
    personal_care_and_beauty_shopee = pd.read_excel(os.path.join(bath_path_beauty,'Face_Masks_BD.xlsx'),
                                             sheet_name, skiprows = [0])
    # chang column name
    old_columns = personal_care_and_beauty_shopee.columns.tolist()
    new_columns = personal_care_and_beauty_shopee.iloc[0].values.tolist()
    personal_care_and_beauty_shopee = personal_care_and_beauty_shopee.rename(
        columns = dict(zip(old_columns,new_columns))
            )
    # data cleaning
    personal_care_and_beauty_shopee = personal_care_and_beauty_shopee.reset_index()
    # drop the first row by index
    personal_care_and_beauty_shopee.drop([0], inplace = True)
    # select the columns we only required
    personal_care_and_beauty_shopee = personal_care_and_beauty_shopee[['Item name','Brand','Product ']]
    # make the columns consistent with train
    personal_care_and_beauty_shopee = personal_care_and_beauty_shopee.rename(
    columns = {'Item name':'item_name', 'Brand':'what_brand_name', 'Product ': 'what_product_name'}
    )
    # cleaning duplicated item_name
    personal_care_and_beauty_shopee = personal_care_and_beauty_shopee.drop_duplicates(subset = ['item_name'])
    # convert type into str
    personal_care_and_beauty_shopee.item_name = personal_care_and_beauty_shopee.item_name.astype(str)
    personal_care_and_beauty_shopee.what_brand_name = personal_care_and_beauty_shopee.what_brand_name.astype(str)
    # take the sku that has confirmed ground truth as test
    personal_care_and_beauty_shopee = personal_care_and_beauty_shopee[
    personal_care_and_beauty_shopee.what_brand_name != 'Others'
    ]
    # remove the bad case
    personal_care_and_beauty_shopee = personal_care_and_beauty_shopee[personal_care_and_beauty_shopee.what_brand_name != 'beauty mask']
    personal_care_and_beauty_shopee = personal_care_and_beauty_shopee[personal_care_and_beauty_shopee.what_brand_name != "nan"]
    # make the index unique
    personal_care_and_beauty_shopee.index = np.arange(personal_care_and_beauty_shopee.shape[0]) 
    gc.collect()

    logging.info('there is no duplicated item in beauty' if personal_care_and_beauty_shopee.item_name.nunique() == personal_care_and_beauty_shopee.shape[0] else "opps")
    #--------------------------
    # loading data (beauty) / amazon
    #--------------------------
    beauty_amazon_path = '../../../grouping/tv_and_laptop_grouping/raw_data/amazon/beauty_amazon.csv'
    beauty_amazon = pd.read_csv(beauty_amazon_path)
    beauty_amazon.item_name = beauty_amazon.item_name.astype(str)
    beauty_amazon.what_brand_name = beauty_amazon.what_brand_name.astype(str)
    beauty_amazon.dropna(subset = ['item_name'], axis = 0, inplace = True)

    #--------------------------
    # loading data (tv and laptop) / amazon
    #--------------------------
    tv_laptop_amazon_path = '../../../grouping/tv_and_laptop_grouping/raw_data/amazon/tv_laptop_amazon.csv'
    tv_laptop_amazon = pd.read_csv(tv_laptop_amazon_path)
    tv_laptop_amazon.item_name = tv_laptop_amazon.item_name.astype(str)
    tv_laptop_amazon.what_brand_name = tv_laptop_amazon.what_brand_name.astype(str)
    tv_laptop_amazon.dropna(subset = ['item_name'], axis = 0, inplace = True)

    #--------------------------
    # validating strategy: train/val/test split based on the unique brand_name
    #--------------------------

    #-------------
    # tv & laptop / shopee
    #-------------
    tv_and_laptop_df = brand_split_validating_strategy(tv_and_laptop, train_val_rate = 0.5, dataset_source = 'Shopee')
    personal_care_and_beauty_shopee_df = brand_split_validating_strategy(personal_care_and_beauty_shopee, train_val_rate = 0.5, dataset_source = 'Shopee')
    # # check raw data first
    # personal_care_and_beauty_shopee_df.to_csv('data/face_mask_raw_data.csv', index = False)
    logging.info('thers is no bugging in validating strategy on tv_and_laptop/shopee' if tv_and_laptop_df.shape[0] == tv_and_laptop.shape[0] else 'bugging in combination')
    logging.info('thers is no bugging in validating strategy on personal_care_and_beauty_shopee/shopee' if personal_care_and_beauty_shopee_df.shape[0] == personal_care_and_beauty_shopee.shape[0] else 'bugging in combination')
    #--------------------------
    # create_supervised_data
    #--------------------------
    tv_and_laptop_df = tv_and_laptop_df.groupby('item_name').apply(sequence_labeling_w_bio_encoding).reset_index(drop = True)
    personal_care_and_beauty_shopee_df = personal_care_and_beauty_shopee_df.groupby('item_name').apply(sequence_labeling_w_bio_encoding).reset_index(drop = True)
    #--------------------------
    # training data eda report ==> transform to the percentage( unbranded product ratio)
    #--------------------------
    # 3c / shopee
    tv_df_stat = tv_and_laptop_df.groupby(by = 'item_name').is_brand.mean().to_frame().reset_index().sort_values('is_brand') 
    num_item_without_positive_lable = tv_df_stat[tv_df_stat.is_brand == 0].shape[0]
    logging.info('ratio of sku in 3c cannot find brand name given his item_name /shopee : {} '.format(np.round(1.0 * num_item_without_positive_lable / len(tv_df_stat), 4)))
    # beauty / shopee
    beauty_df_stat = personal_care_and_beauty_shopee_df.groupby(by = 'item_name').is_brand.mean().to_frame().reset_index().sort_values('is_brand') 
    num_item_without_positive_lable = beauty_df_stat[beauty_df_stat.is_brand == 0].shape[0]
    logging.info('ratio of sku in beauty cannot find brand name given his item_name / shopee: {}'.format(np.round(1.0 * num_item_without_positive_lable / len(beauty_df_stat), 4)))

    #--------------------------
    # drop all unbranded iten_name ( we want to keep a little bit to make machine can learn these examples) on amazon dataset
    #--------------------------
    tv_and_laptop_df = tv_and_laptop_df \
    [~tv_and_laptop_df.item_name.isin(tv_df_stat[tv_df_stat.is_brand == 0].item_name.unique())]
    personal_care_and_beauty_shopee_df = personal_care_and_beauty_shopee_df \
    [~personal_care_and_beauty_shopee_df.item_name.isin(beauty_df_stat[beauty_df_stat.is_brand == 0].item_name.unique())]

    # preprocessing
    personal_care_and_beauty_shopee_df = personal_care_and_beauty_shopee_df[personal_care_and_beauty_shopee_df.item_name != 'Kiehls Rare Earth Deep Pore Cleansing Masque \r 14gr']
    tv_and_laptop_df.tokens = [np.nan if t == '' else t for t in tv_and_laptop_df.tokens]
    personal_care_and_beauty_shopee_df.tokens = [np.nan if t == '' else t for t in personal_care_and_beauty_shopee_df.tokens]
    tv_and_laptop_df.dropna(subset = ['tokens'], axis = 0, inplace = True)
    personal_care_and_beauty_shopee_df.dropna(subset = ['tokens'], axis = 0, inplace = True)

    #--------------------------
    # training data eda report ==> transform to the percentage( unbranded product ratio)
    #--------------------------
    # 3c / shopee
    tv_df_stat = tv_and_laptop_df.groupby(by = 'item_name').is_brand.mean().to_frame().reset_index().sort_values('is_brand') 
    num_item_without_positive_lable = tv_df_stat[tv_df_stat.is_brand == 0].shape[0]
    logging.info('after drop, ratio of sku in 3c cannot find brand name given his item_name /shopee : {} '.format(np.round(1.0 * num_item_without_positive_lable / len(tv_df_stat), 4)))
    # beauty / shopee
    beauty_df_stat = personal_care_and_beauty_shopee_df.groupby(by = 'item_name').is_brand.mean().to_frame().reset_index().sort_values('is_brand') 
    num_item_without_positive_lable = beauty_df_stat[beauty_df_stat.is_brand == 0].shape[0]
    logging.info('after drop, ratio of sku in beauty cannot find brand name given his item_name / shopee: {}'.format(np.round(1.0 * num_item_without_positive_lable / len(beauty_df_stat), 4)))

    gc.collect()
    #-------------
    # personal_care_and_beauty / amazon
    #-------------
    beauty_amazon_df = brand_split_validating_strategy(beauty_amazon, train_val_rate = 0.8, dataset_source = 'Amazon')
    logging.info('thers is no bugging in validating strategy on beauty/amazon' if beauty_amazon_df.shape[0] == beauty_amazon.shape[0] else 'bugging in combination')
    del beauty_amazon
    gc.collect()

    #-------------
    # tv&laptop / amazon
    #-------------
    tv_laptop_amazon_df = brand_split_validating_strategy(tv_laptop_amazon, train_val_rate = 0.8, dataset_source = 'Amazon')
    logging.info('thers is no bugging in validating strategy on tv_laptop/amazon' if tv_laptop_amazon_df.shape[0] == tv_laptop_amazon.shape[0] else 'bugging in combination')
    del tv_laptop_amazon
    gc.collect()
    #---------------------
    # preprocessing for distantly supervison method
    #---------------------
    beauty_amazon_df['count_of_what_brand_name_popping_out'] = [i_n.lower().count(b_n) for i_n, b_n in zip(beauty_amazon_df.item_name,
                                                                                beauty_amazon_df.what_brand_name)]
    tv_laptop_amazon_df['count_of_what_brand_name_popping_out'] = [i_n.lower().count(b_n) for i_n, b_n in zip(tv_laptop_amazon_df.item_name,
                                                                                tv_laptop_amazon_df.what_brand_name)]

    beauty_amazon_df = beauty_amazon_df[beauty_amazon_df.count_of_what_brand_name_popping_out <= 1]
    beauty_amazon_df.drop(['count_of_what_brand_name_popping_out'], axis = 1, inplace = True)

    tv_laptop_amazon_df = tv_laptop_amazon_df[tv_laptop_amazon_df.count_of_what_brand_name_popping_out <= 1]
    tv_laptop_amazon_df.drop(['count_of_what_brand_name_popping_out'], axis = 1, inplace = True)

    #--------------------------
    # create_supervised_data
    #--------------------------
    beauty_amazon_df = beauty_amazon_df.groupby('item_name').apply(sequence_labeling_w_bio_encoding).reset_index(drop = True)
    tv_laptop_amazon_df = tv_laptop_amazon_df.groupby('item_name').apply(sequence_labeling_w_bio_encoding).reset_index(drop = True)

    #--------------------------
    # training data eda report ==> transform to the percentage( unbranded product ratio)
    #--------------------------
    # 3c / amazone
    tv_df_stat = tv_laptop_amazon_df.groupby(by = 'item_name').is_brand.mean().to_frame().reset_index().sort_values('is_brand') 
    num_item_without_positive_lable_tv = tv_df_stat[tv_df_stat.is_brand == 0].shape[0]
    logging.info('ratio of sku in 3c cannot find brand name given his item_name /amazon : {}'.format(np.round(1.0 * num_item_without_positive_lable / len(tv_df_stat), 4)))
    # beauty / amazone
    beauty_df_stat = beauty_amazon_df.groupby(by = 'item_name').is_brand.mean().to_frame().reset_index().sort_values('is_brand') 
    num_item_without_positive_lable_beauty = beauty_df_stat[beauty_df_stat.is_brand == 0].shape[0]
    logging.info('ratio of sku in beauty cannot find brand name given his item_name / amazon: {}'.format(np.round(1.0 * num_item_without_positive_lable / len(beauty_df_stat), 4)))
    
    #--------------------------
    # preprocessing
    #--------------------------

    tv_laptop_amazon_df = tv_laptop_amazon_df[~tv_laptop_amazon_df.item_name.isin(tv_df_stat[tv_df_stat.is_brand == 0].item_name.unique())]
    beauty_amazon_df = beauty_amazon_df[~beauty_amazon_df.item_name.isin(beauty_df_stat[beauty_df_stat.is_brand == 0].item_name.unique())]
    tv_laptop_amazon_df.dropna(subset = ['tokens'], axis = 0, inplace = True)
    beauty_amazon_df.dropna(subset = ['tokens'], axis = 0, inplace = True)
    beauty_amazon_df = beauty_amazon_df[beauty_amazon_df.item_name != 'Ageless Answer Moisturizing Cream Gary Null 45 oz Cream']
    #--------------------------
    # Make the amazone dataset distribution closer to shopee dataset via cus-off too long sequence on amazon dataset
    #--------------------------
    # TV
    tokens_count_df = tv_and_laptop_df.groupby('item_name').apply(lambda x: x.tokens.size).to_frame('num_count').reset_index()
    shopee_longest_itemname_num = tokens_count_df.num_count.max() # int
    del tokens_count_df
    tokens_count_df_amazon = tv_laptop_amazon_df.groupby('item_name').apply(lambda x: x.tokens.size) \
    .to_frame('num_count').reset_index()
    tv_laptop_amazon_df = pd.merge(tv_laptop_amazon_df, tokens_count_df_amazon, on = 'item_name', how = 'left')
    tv_laptop_amazon_df = tv_laptop_amazon_df[tv_laptop_amazon_df.num_count <= shopee_longest_itemname_num]
    tv_laptop_amazon_df.drop(['num_count'], axis = 1, inplace = True)
    # Beauty
    tokens_count_df = personal_care_and_beauty_shopee_df.groupby('item_name').apply(lambda x: x.tokens.size).to_frame('num_count').reset_index()
    shopee_longest_itemname_num = tokens_count_df.num_count.max() # int
    del tokens_count_df
    tokens_count_df_amazon = beauty_amazon_df.groupby('item_name').apply(lambda x: x.tokens.size) \
    .to_frame('num_count').reset_index()
    beauty_amazon_df = pd.merge(beauty_amazon_df, tokens_count_df_amazon, on = 'item_name', how = 'left')
    beauty_amazon_df = beauty_amazon_df[beauty_amazon_df.num_count <= shopee_longest_itemname_num]
    beauty_amazon_df.drop(['num_count'], axis = 1, inplace = True)

    gc.collect()
    #--------------------------
    # training data eda report ==> transform to the percentage( unbranded product ratio)
    #--------------------------
    # 3c / amazone
    tv_df_stat = tv_laptop_amazon_df.groupby(by = 'item_name').is_brand.mean().to_frame().reset_index().sort_values('is_brand') 
    num_item_without_positive_lable_tv = tv_df_stat[tv_df_stat.is_brand == 0].shape[0]
    logging.info('after drop, ratio of sku in 3c cannot find brand name given his item_name /amazon : {}'.format(np.round(1.0 * num_item_without_positive_lable / len(tv_df_stat), 4)))
    # beauty / amazone
    beauty_df_stat = beauty_amazon_df.groupby(by = 'item_name').is_brand.mean().to_frame().reset_index().sort_values('is_brand') 
    num_item_without_positive_lable_beauty = beauty_df_stat[beauty_df_stat.is_brand == 0].shape[0]
    logging.info('after drop, ratio of sku in beauty cannot find brand name given his item_name / amazon: {}'.format(np.round(1.0 * num_item_without_positive_lable / len(beauty_df_stat), 4)))

    #--------------------------
    # save
    #--------------------------
    output_dir = '../data/preprocessed'
    # date_str = datetime.now().strftime('%Y-%m-%d')
    # output_dir = os.path.join(output_dir, date_str)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # shopee
    tv_and_laptop_df.to_csv(os.path.join(output_dir, 'tv_and_laptop.csv'), index = False)
    personal_care_and_beauty_shopee_df.to_csv(os.path.join(output_dir, 'personal_care_and_beauty.csv'), index = False)

    # amazon
    beauty_amazon_df.to_csv(os.path.join(output_dir, 'beauty_amazon.csv'), index = False)
    tv_laptop_amazon_df.to_csv(os.path.join(output_dir, 'tv_laptop_amazon.csv'), index = False)


