# System
import os

# Externals
import copy
import tqdm
import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
# import statsmodels.api as sm
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Locals
import config
import main
import misc


    
def criteria_dswe(row, model_swe_t1, model_name, Model, nnScaler, features_lst):
    '''
    Predict dSWE based on pre-trained model and physical conditions on given day.
    '''
    # Get datetime of current df row
    row_datetime = pd.to_datetime(row['datetime']).to_pydatetime().replace(tzinfo=None)    # get datetime of current row

    # dSWE = 0 on dry days with no snow on the ground. Under all other physical conditions, predict dSWE using pre-trained model.
    if model_swe_t1 == 0.0 and row['pr'] == 0.0:
        model_dswe_t = 0.0
    else:
        model_covariates = row[features_lst].to_frame().swapaxes("index", "columns")
        if 'swe_t1' in features_lst:
            model_covariates['swe_t1'] = model_swe_t1
        if model_name == 'rf':
            if model_covariates.isnull().values.any():
                model_dswe_t = np.nan
            else:
                model_dswe_t = Model.predict(model_covariates.astype('float64'))[0]
        elif model_name == 'nn':
            Model.eval()
            if model_covariates.isnull().values.any():
                model_dswe_t = np.nan
            else:
                model_dswe_t = Model(torch.tensor(nnScaler.transform(model_covariates.astype('float32').values))).item()
        else:
            if model_covariates.isnull().values.any():
                model_dswe_t = np.nan
            else:
                model_dswe_t = Model.predict(model_covariates.astype('float64')).values[0]
            
    # Accumulate dSWE onto running SWE. If result is negative, make zero.
    if model_swe_t1 + model_dswe_t < 0.0:
        model_swe_t = 0.0
    else:
        model_swe_t = model_swe_t1 + model_dswe_t
    
    
    return model_swe_t, model_dswe_t


    
def combine_dfs(save_dest):
    '''
    Combine dataframes from individual site-years into a single dataframe.
    '''
    # Initialize list of dataframes.
    dfs = []
    # Loop through dataframes for all site-years and append to running list
    for file in sorted(os.listdir(save_dest)):
        filename = os.fsdecode(file)
        if filename.endswith("_df.csv"):
            site_wy_df = pd.read_csv(f'{save_dest}{filename}', index_col=0)
            dfs.append(site_wy_df)
    # Concatenate dataframes in list to single large dataframe
    df_full = pd.concat(dfs)
    # Save full large dataframe to file
    df_full.to_csv(f'{save_dest}/ALL_SITES_WY_DF.csv')
    
    
    
def generate_time_series(save_pth, train_wys, val_wys, all_sites_df_TRAIN, all_sites_df_VAL, linReg, quadReg, cubReg, rF, nN, sScaler, mode):
    '''
    Generate SWE time series based on dSWE model predictions.
    '''
    
    if mode == 'train':
        df = all_sites_df_TRAIN.copy()
        wys = train_wys
    elif mode == 'val':
        df = all_sites_df_VAL.copy()
        wys = val_wys
    
    # Loop through SNOTEL sites and water years, predicting dSWE and SWE on each day of each site-year
    for site in np.unique(df['sitecode']): 
        site_df = df[df['sitecode'] == site]
        for wy in wys:
            if os.path.exists(f'{save_pth}{mode}/{site}_{wy}_df.csv'):      # skip water year if this site-year was already run
                continue 
                
            if not site_df[site_df['water_year'] == wy].empty:
                site_wy_df = site_df[site_df['water_year'] == wy]
                site_wy_df = site_wy_df.dropna(subset=list(set(config.features) | set(config.full_features)), how='any')    # drop rows where at least one feature is nan

                if len(site_wy_df) != 365:       # skip water year if site-year data is incomplete
                    print(len(site_wy_df), site, wy)
                    continue

                # Load UA SWE data
                uaswe_df = xr.open_dataset(f'{config.uaswe_dir}4km_SWE_Depth_WY{wy}_v01.nc')

                # Get coordinates of this site
                site_coords = config.gm_snotel_sites[config.gm_snotel_sites['code'] == site]
                site_lon, site_lat = site_coords['geometry'].x.item(), site_coords['geometry'].y.item()

                for idx, row in site_wy_df.iterrows():
                    # Get datetime of current row
                    row_datetime = pd.to_datetime(row['datetime']).to_pydatetime().replace(tzinfo=None)  
                    # Extract UA SWE at grid cell corresponding to this SNOTEL site
                    uaswe_gt = uaswe_df.SWE.sel(time=row_datetime, lon=site_lon, lat=site_lat, method='nearest').item()    

                    # Predict dSWE and SWE iteratively for each day of water year
                    if row_datetime.month == 10 and row_datetime.day == 1:
                        lm_swe_t, p2m_swe_t, p3m_swe_t, rf_swe_t, nn_swe_t = 0.0, 0.0, 0.0, 0.0, 0.0
                        lm_dswe_t, p2m_dswe_t, p3m_dswe_t, rf_dswe_t, nn_dswe_t = 0.0, 0.0, 0.0, 0.0, 0.0
                    else:
                        lm_swe_t, lm_dswe_t = criteria_dswe(row, lm_swe_t1, 'lm', linReg, None, config.features)
                        p2m_swe_t, p2m_dswe_t = criteria_dswe(row, p2m_swe_t1, 'p2m', quadReg, None, config.features)
                        p3m_swe_t, p3m_dswe_t = criteria_dswe(row, p3m_swe_t1, 'p3m', cubReg, None, config.features)
                        rf_swe_t, rf_dswe_t = criteria_dswe(row, rf_swe_t1, 'rf', rF, None, config.features)
                        nn_swe_t, nn_dswe_t = criteria_dswe(row, nn_swe_t1, 'nn', nN, sScaler, config.features)

                    # Append current date predictions to df
                    site_wy_df.loc[idx, 'UASWE'] = uaswe_gt
                    site_wy_df.loc[idx, 'pred_swe_LM'] = lm_swe_t
                    site_wy_df.loc[idx, 'pred_swe_P2M'] = p2m_swe_t
                    site_wy_df.loc[idx, 'pred_swe_P3M'] = p3m_swe_t
                    site_wy_df.loc[idx, 'pred_swe_RF'] = rf_swe_t
                    site_wy_df.loc[idx, 'pred_swe_NN'] = nn_swe_t

                    site_wy_df.loc[idx, 'pred_dswe_LM'] = lm_dswe_t
                    site_wy_df.loc[idx, 'pred_dswe_P2M'] = p2m_dswe_t
                    site_wy_df.loc[idx, 'pred_dswe_P3M'] = p3m_dswe_t
                    site_wy_df.loc[idx, 'pred_dswe_RF'] = rf_dswe_t
                    site_wy_df.loc[idx, 'pred_dswe_NN'] = nn_dswe_t

                # Reset SWE predictions to zero at the end of the water year
                lm_swe_t1 = 0.0
                p2m_swe_t1 = 0.0
                p3m_swe_t1 = 0.0
                rf_swe_t1 = 0.0
                nn_swe_t1 = 0.0

                # Plot time series of each model prediction for this site-year
                plt.figure(figsize=(15, 4))
                plt.plot(site_wy_df['swe'], color='black', label='SNOTEL SWE')
                plt.plot(site_wy_df['UASWE'], color='grey', label='UA SWE')
                plt.plot(site_wy_df['pred_swe_LM'], color='green', label='LM Pred SWE')
                plt.plot(site_wy_df['pred_swe_P2M'], color='red', label='P2M Pred SWE')
                plt.plot(site_wy_df['pred_swe_P3M'], color='tab:orange', label='P3M Pred SWE')
                plt.plot(site_wy_df['pred_swe_RF'], color='tab:olive', label='RF Pred SWE')
                plt.plot(site_wy_df['pred_swe_NN'], color='purple', label='NN Pred SWE')
                plt.legend()
                plt.title(f'{np.unique(site_wy_df["sitecode"]).item()} {site_wy_df["datetime"].iloc[0]}')
                # plt.show()
                plt.savefig(f'{save_pth}{mode}/{site}_{wy}_SWE.png', dpi=300)

                site_wy_df.to_csv(f'{save_pth}{mode}/{site}_{wy}_df.csv')

    # Combine dataframes from individual site-years into a single dataframe.
    combine_dfs(f'{save_pth}{mode}/')
    