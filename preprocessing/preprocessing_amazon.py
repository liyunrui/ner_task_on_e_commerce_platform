#! /usr/bin/env python3
'''
Created on July 25 2018
@author: Ray

Amazon DataSet refer to https://github.com/etano/productner

'''
import pandas as pd
import numpy as np
import os
#--------------------------
# loading data 
#--------------------------
amazon_dataset_path = '../../../grouping/tv_and_laptop_grouping/raw_data/amazon/products.csv'
amazon_dataset = pd.read_csv( amazon_dataset_path, header = None)
amazon_dataset.rename(columns = {0: 'item_name',
                                 1: 'what_brand_name',
                                 2: 'description',
                                 3: 'category'
                                }, inplace = True)

#--------------------------
# processing 
#--------------------------
amazon_dataset.item_name = amazon_dataset.item_name.astype(str)
amazon_dataset.category = amazon_dataset.category.astype(str)
# covert what_brand_name to lower for creating supervised data.
amazon_dataset.what_brand_name = amazon_dataset.what_brand_name.apply(lambda x: x.lower() if type(x) == str else x)
# filling na with Others
amazon_dataset.what_brand_name = amazon_dataset.what_brand_name.fillna('Others')

#--------------------------
# catalogue selection
#--------------------------
beauty = amazon_dataset[amazon_dataset.category.str.contains('Beauty')]
tv_laptop_amazon = pd.concat([amazon_dataset[amazon_dataset.category.str.contains('Laptop')], 
	amazon_dataset[amazon_dataset.category.str.contains('Electronics / Television & Video / Televisions / LCD TVs')]]
	, axis = 0)

#--------------------------
# data clearning
#--------------------------

# drop duplicated
beauty.drop_duplicates(['item_name'], inplace = True)
tv_laptop_amazon.drop_duplicates(['item_name'], inplace = True)
# clearn the rediculous what_brand_name in beatuy
dirty_brand_name_in_beauty = ['360','4420', '844825074132', '42', '4711', 
'3','11','300960729080','070501025956','322170312043', '075609026232',
 '180','666', '300960731045', '8', '1234', '47338', '.', '309974458146']
beauty = beauty[~beauty.what_brand_name.isin(dirty_brand_name_in_beauty)]
# drop Others
tv_laptop_amazon = tv_laptop_amazon[tv_laptop_amazon.what_brand_name != 'Others']
beauty = beauty[beauty.what_brand_name != 'Others']

#--------------------------
# simple eda or data understanding
#--------------------------
num_beauty = beauty.shape[0]
num_tv_laptop = tv_laptop_amazon.shape[0]
num_sku = amazon_dataset.shape[0]
print ('around {} percent is Beauty catalogue on Amazon dataset'.format(np.round(100.0 * num_beauty / num_sku, 4)))
print ('around {} percent is TV and Laptop catalogue on Amazon dataset'.format(np.round(100.0 * num_tv_laptop / num_sku, 4)))

# beauty
print ('there is no nan in what_brand_name on beauty_amazon' if beauty.what_brand_name.count() == beauty.shape[0] else 'it has nan in the brand name')
print ('there is no duplicated item_name on beauty_amazon' if beauty.item_name.nunique() == beauty.shape[0] else 'oops')
print ('num_unique_brand on Amazon dataset on beauty_amazon: {}'.format(beauty.what_brand_name.nunique()))
print ('num_sku without brand on beauty_amazon: {}'.format(beauty[beauty.what_brand_name == 'Others'].shape[0])) # 把它拿掉
# tv and laptop
print ('there is no nan in what_brand_name on tv_laptop_amazon' if tv_laptop_amazon.what_brand_name.count() == tv_laptop_amazon.shape[0] else 'it has nan in the brand name')
print ('there is no duplicated item_name on tv_laptop_amazon' if tv_laptop_amazon.item_name.nunique() == tv_laptop_amazon.shape[0] else 'oops')
print ('num_unique_brand on Amazon dataset on tv_laptop_amazon: {}'.format(tv_laptop_amazon.what_brand_name.nunique()))
print ('num_sku without brand on tv_laptop_amazon: {}'.format(tv_laptop_amazon[tv_laptop_amazon.what_brand_name == 'Others'].shape[0]))

# 輸出unique brand for 和shopee比較

#--------------------------
# save
#--------------------------
output_base_path = '../../../grouping/tv_and_laptop_grouping/raw_data/amazon'
beauty[['item_name','what_brand_name']].to_csv(os.path.join(output_base_path,'beauty_amazon.csv'), index = False)
tv_laptop_amazon[['item_name','what_brand_name']].to_csv(os.path.join(output_base_path,'tv_laptop_amazon.csv'), index = False)



