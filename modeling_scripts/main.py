# System
import os
import logging
import warnings
warnings.filterwarnings('ignore')   # suppress warning messages
import multiprocessing as mp         # parallelization
from multiprocessing import Pool

# Externals
import copy
import tqdm
import json
import math
import random
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import richdem as rd
import rasterio as rio
import geopandas as gpd
from shapely import wkt
from datetime import datetime
import matplotlib.pyplot as plt
# import statsmodels.api as sm  
import statsmodels.formula.api as sm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Locals
import config
import data
import models
import train
import time_series_dswe
import misc



# Initialize logging
logger = mp.log_to_stderr(logging.DEBUG)


def by_fold(k_ct):
    # Initialize fold and multiprocessing
    k = config.cv_splits[k_ct]
    print(mp.current_process())
    #logger.info('Received {}'.format(k))
    logger.info(mp.current_process())
    
    # Define and create paths
    save_path = config.save_dir+'_'.join(features)+'/'+'k='+str(k_ct)+'/'
    misc.make_dirs(save_path)

    # Cross validation: training and validation years
    train_yrs = list(np.setdiff1d(config.trn,k))
    val_yrs = k
    # Save train and validation years to file
    d = {'train_yrs': train_yrs, 'val_yrs': val_yrs}
    d['train_yrs'] = list(map(int, d['train_yrs']))
    json.dump(d, open(save_path+"train_val_yrs.txt",'w'))

    
    ### Prepare, clean, and organize dataframes ###
    # Create rain and snow columns based on rain-snow partitioning threshold
    all_sites_df['rain'] = all_sites_df['pr'].where(all_sites_df['temp_partition']=='rain', 0)
    all_sites_df['snow'] = all_sites_df['pr'].where(all_sites_df['temp_partition']=='snow', 0)
    # Add necessary columns to dataframe based on features and lags
    full_features = features.copy()
    if any("pr_t" in s for s in full_features):    # if feautures list contains a lag item
        nlags = int(next(x for x in full_features if 'pr_t' in x)[4])    # get the lag value
        for l in range(1, nlags):
            features.append(f'pr_t{l}')
    if any("temp_t" in s for s in full_features):    # if feautures list contains a lag item
        nlags = int(next(x for x in full_features if 'temp_t' in x)[6])    # get the lag value
        for l in range(1, nlags):
            features.append(f'temp_t{l}')
    if any("tmin_t" in s for s in full_features):    # if feautures list contains a lag item
        nlags = int(next(x for x in full_features if 'tmin_t' in x)[6])    # get the lag value
        for l in range(1, nlags):
            features.append(f'tmin_t{l}')
    if any("tmax_t" in s for s in full_features):    # if feautures list contains a lag item
        nlags = int(next(x for x in full_features if 'tmax_t' in x)[6])    # get the lag value
        for l in range(1, nlags):
            features.append(f'tmax_t{l}')
    full_features = features.copy()
    
    # Subset to train and validation years for this fold
    all_sites_df_TRAIN = all_sites_df[all_sites_df['water_year'].isin(train_yrs)]
    all_sites_df_VAL = all_sites_df[all_sites_df['water_year'].isin(val_yrs)]

    # Clean dataframes
    train_set = data.clean_dataframe(all_sites_df_TRAIN)
    val_set = data.clean_dataframe(all_sites_df_VAL)
    # Remove outliers
    clean_train_set = data.remove_outliers(train_set)
    clean_val_set = data.remove_outliers(val_set)
    # Organize dataframes
    trim_df_train = data.organize_dataframe(clean_train_set, full_features, mode='train')
    trim_df_val = data.organize_dataframe(clean_val_set, full_features, mode='val')
    
    ### Train models ###
    linReg, quadReg, cubReg, rF, nN, sScaler, clean_train_set, clean_val_set = train.train_models(features, trim_df_train, trim_df_val, clean_train_set, clean_val_set, save_path)
    
    # Combine model predictions to single dataframe
    all_sites_df_TRAIN = all_sites_df_TRAIN.merge(clean_train_set[[f'pred_{config.output}_LM',f'pred_{config.output}_P2M',f'pred_{config.output}_P3M',f'pred_{config.output}_RF',f'pred_{config.output}_NN']],how='left', left_index=True, right_index=True)
    all_sites_df_VAL = all_sites_df_VAL.merge(clean_val_set[[f'pred_{config.output}_LM',f'pred_{config.output}_P2M',f'pred_{config.output}_P3M',f'pred_{config.output}_RF',f'pred_{config.output}_NN']],how='left', left_index=True, right_index=True)
    
    # Save full dataframes to file
    all_sites_df_TRAIN.to_csv(save_path+'all_sites_df_TRAIN.csv')
    all_sites_df_VAL.to_csv(save_path+'all_sites_df_VAL.csv')
    
    
    ### Generate continuous SWE time series based on daily SWE predictions from pre-trained models ###
    time_series_dswe.generate_time_series(save_path, train_yrs, val_yrs, all_sites_df_TRAIN, all_sites_df_VAL, 
                                          linReg, quadReg, cubReg, rF, nN, sScaler, mode='train')
    time_series_dswe.generate_time_series(save_path, train_yrs, val_yrs, all_sites_df_TRAIN, all_sites_df_VAL, 
                                          linReg, quadReg, cubReg, rF, nN, sScaler, mode='val')
    
    json.dump('finish run', open(save_path+"finish.txt",'w'))

    
    return k



if __name__ == '__main__':
    
    st = datetime.now()
    
    # Folds in cross validation
    folds = [0,1,2,3]
    
    # Get global variables
    features = config.features
    all_sites_df = config.all_sites_df
    
    # Multiprocessing
    with Pool(4) as pool:        # with Pool() as pool:
        result = pool.map(by_fold, folds)
        print('after result')
        logger.info(result)
    
    # Compute total code run time
    et = datetime.now()
    tt = et - st
    print(tt)
    
