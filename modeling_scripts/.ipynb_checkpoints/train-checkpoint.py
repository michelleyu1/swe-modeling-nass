# Externals
import joblib
import torch
import torch.nn as nn
import numpy as np

# Locals
import config
import models



def train_models(features, reduced_df_train, reduced_df_val, cleaned_train_df, cleaned_val_df, save_pth):
    '''
    Train models and save trained models to file.
    
    Parameters:
    reduced_df_train : trimmed and cleaned version of train_df (subset to key features and dropped NANs)
    reduced_df_val : trimmed and cleaned version of val_df (subset to key features and dropped NANs)
    cleaned_train_df : cleaned train set (after data pre-processing)
    cleaned_val_df : cleaned val set (after data pre-processing)
    '''
    
    # Fit linear model
    lm, train_set, val_set = models.fit_reg(1, reduced_df_train, reduced_df_val, cleaned_train_df, cleaned_val_df, features)
    # Fit quadratic regression
    p2m, train_set, val_set = models.fit_reg(2, reduced_df_train, reduced_df_val, cleaned_train_df, cleaned_val_df, features)
    # Fit cubic regression
    p3m, train_set, val_set = models.fit_reg(3, reduced_df_train, reduced_df_val, cleaned_train_df, vcleaned_al_df, features)
    # Train random forest
    rf, train_set, val_set = models.train_rf(reduced_df_train, reduced_df_val, cleaned_train_df, cleaned_val_df, features)
    # Train neural network
    nnet, sclr, train_set, val_set = models.train_nn(reduced_df_train, reduced_df_val, cleaned_train_df, cleaned_val_df, features, l1_nnodes=256, nepochs=1000)
    
    # Save models to file
    joblib.dump(lm, f'{save_pth}lm.joblib')
    joblib.dump(p2m, f'{save_pth}p2m.joblib')
    joblib.dump(p3m, f'{save_pth}p3m.joblib')
    joblib.dump(rf, f'{save_pth}rf.joblib')
    torch.save(nnet.state_dict(), f'{save_pth}nnet.pth')
    joblib.dump(sclr, f'{save_pth}std_scaler.joblib')

    return lm, p2m, p3m, rf, nnet, sclr, train_set, val_set
    
    
    
def load_trained_models(features, save_pth):
    '''
    Load trained models from file.
    '''
    
    lm = joblib.load(f'{save_pth}lm.joblib')
    p2m = joblib.load(f'{save_pth}p2m.joblib')
    p3m = joblib.load(f'{save_pth}p3m.joblib')
    rf = joblib.load(f'{save_pth}rf.joblib')
    nnet = nn.Sequential(
        nn.Linear(len(features), 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 1),
    )
    nnet.load_state_dict(torch.load(f'{save_pth}nnet.pth'))
    sclr = joblib.load(f'{save_pth}std_scaler.joblib')
    
    return lm, p2m, p3m, rf, nnet, sclr