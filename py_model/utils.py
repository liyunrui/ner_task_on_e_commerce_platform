#! /usr/bin/env python3

import logging
import os
from datetime import datetime
from operator import itemgetter
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import numpy as np
#--------------------
# thereshold
#--------------------

def get_best_threshold(df, interval = 0.05, measurement = 'f1'):
    '''
    It's the function to determine the best threshold,
    which returns the best threshold according to the measurement you assigned.
    
    parameters:
    -----------------
    df: DataFrame
    measurement: str
    interval: float, range between (0,1)
    ''' 
    for_best_threshold = [] # can be used to visualization
    for th in np.arange(0.0, 1.0, step = interval):
        #------------------
        # threshold
        #------------------
        threshold = th
        df['y_pred'] = df.score.apply(lambda x: 1 if x >= threshold else 0) 
        #------------------
        # choicd of measurement
        #------------------
        if measurement == 'f1':
            performance = f1_score(df.y_true.values, df.y_pred.values)
        elif measurement == 'recall':
            performance = recall_score(df.y_true.values, df.y_pred.values)
        elif measurement == 'precision':
            performance = precision_score(df.y_true.values, df.y_pred.values)
        else:
            print ('fuck u, the choise u have is f1, recall, precision')
        for_best_threshold.append((th, performance))
    #-----------------
    # unit testing by visualizing 
    #-----------------
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    ths = [i[0] for i in for_best_threshold]
    per = [i[1] for i in for_best_threshold]
    plt.plot(ths, per)
    plt.show()
    #-----------------
    # output
    #-----------------
    # get the maximun element given the index from list of tuple
    best_th = max(for_best_threshold, key = itemgetter(1))[0]
    best_performance = max(for_best_threshold, key = itemgetter(1))[1]
    df['threshold'] = best_th
    df['y_pred'] = df.score.apply(lambda x: 1 if x >= best_th else 0)
    return df, best_th, best_performance

#--------------------
# evaluation metric
#--------------------

def get_positive_token(x, flag = 'atual_brand'):
    '''
    It's a helper function to evaluate our system predictive power.
    
    paras:
    ------------
    x: DataFrame
    flag: str. atual_brand or predicted_brand
    '''
    if flag == 'atual_brand':
        return set(x[x.y_true != 0].tokens.tolist())
    elif flag == 'predicted_brand':
        return set(x[x.y_pred != 0].tokens.tolist())
    else:
        print ('fuck u, the choise u have is atual_brand or predicted_brand')

def top_k_accuracy(df, k = 1):
    '''
    It's evaluation metric for brand detector model. 
    Judge whether the targets are in the top K predictions. return 1 if yes else 0
    
    parameters:
    --------------------
    df: DataFrame
    k: int, number of top elements to look at for computing precision
    '''
    top_k_ix_ls = df.prob_that_is_brand.sort_values(ascending= False)[:k].index.tolist()
    df['system_prediction_result'] = 1 if 1 in set(df.label[top_k_ix_ls]) else 0
    return df

def evaluating_system(df):
    '''
    Evaluate our performance of brand detecor using f1, precision, and recall.
        -precision可以反應, 我們系統預測出有brand的的準確度.
        -recall可以反應, 在真實有brand的情況下, 我們能預測出的能力.
    paras:
    -----------------
    df: DataFrame
    '''
    # initialize variables
    acc = 0
    p = 0
    r = 0
    nun_prediction_true = 0
    nun_actual_true = 0
    correct_pred_for_p = 0
    correct_pred_for_r = 0
    #--------------------
    # core
    #--------------------
    for ix, row in df.iterrows():
        if row['acutal_brand'] == row['predicted_brand']:
            acc += 1
        else:
            pass
    for ix, row in df.iterrows():
        if row['predicted_brand'] != set():
            # our system predict there is a brand here
            nun_prediction_true += 1
            if row['acutal_brand'] == row['predicted_brand']:
                correct_pred_for_p += 1
        else:
            pass
    for ix, row in df.iterrows():
        if row['acutal_brand'] != set():
            # there is a actual brand
            nun_actual_true += 1
            if row['acutal_brand'] == row['predicted_brand']:
                correct_pred_for_r += 1
        else:
            pass
    #--------------------
    # output
    #--------------------
    try:        
        accuracy = acc / df.shape[0]  
    except Exception:
        # avoding to divide into zero
        pass
    try:       
        p = correct_pred_for_p / nun_prediction_true
    except Exception:
        # avoding to divide into zero
        pass
    try:          
        r =  correct_pred_for_r / nun_actual_true
    except Exception:
        # avoding to divide into zero
        pass
    try:          
        f1 =  2 * p * r / (p + r)
    except Exception:
        # avoding to divide into zero
        pass   
    return f1, r, p, accuracy

#--------------------
# logging config
#--------------------

def init_logging(log_dir):
    '''
    for recording the experiments.
    
    log_dir: path
    '''
    #--------------
    # setting
    #--------------
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_file = 'log_{}.txt'.format(date_str)
    #--------------
    # config
    #--------------    
    logging.basicConfig(
        filename = os.path.join(log_dir, log_file),
        level = logging.INFO,
        format = '[[%(asctime)s]] %(message)s',
        datefmt = '%m/%d/%Y %I:%M:%S %p'
    )
    logging.getLogger().addHandler(logging.StreamHandler())
