# Externals
import pandas as pd

# Locals
import config



def clean_dataframe(full_df):
    '''
    Clean dataframe, only keeping days viable for training, based on physical intuition and observation data availability.
    '''
    # Remove all dry days with no snow on the ground
    clean_df = full_df[~((full_df['swe_t1'] == 0.0) & (full_df['pr'] == 0.0))]    
    # Remove rows with NANs in SNOTEL SWE record
    clean_df = clean_df[clean_df['swe_t1'].notnull()]
        
    return clean_df


def remove_outliers(df):
    '''
    Remove outliers idenitified during EDA.
    '''
    nlags = 7
    for l in range(1, nlags+1):
        df = df[df[f'tmin_t{l}'] < 3000]
        df = df[df[f'pr_t{l}'] < 500]
    df = df[df['tmin'] < 3000]
    df = df[df['pr'] < 500]
    df = df[df['swe'] < 500]
    df = df[df['swe_t1'] < 500]
        
    return df


def organize_dataframe(df, features_list, mode):
    '''
    Reduce dataframe size by eliminating miscellaneous columns and missing rows.
    '''
    # subset to covariates of interest
    if mode == 'train':
        clean_df = df[features_list + [config.output, 'acc_dd', 'abl_snw']]
    elif mode == 'val':
        clean_df = df[features_list + [config.output, 'acc_dd']]
    # remove rows with NANs 
    clean_df = clean_df.dropna()
    
    return clean_df


