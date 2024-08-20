import pandas as pd

import config


def clean_dataframe(full_df, modelType):
    clean_df = full_df
    
    if modelType == 'uaswelike':
        clean_df = clean_df[~(clean_df['temp'] < 0.0)]    # Remove cold days
        clean_df = clean_df[~(clean_df['swe_t1'] == 0.0)]    # Remove no SWE days
        clean_df = clean_df[clean_df['swe_t1'].notnull()]
        # clean_df = clean_df[(pd.to_datetime(clean_df['datetime']).dt.month != 7) & (pd.to_datetime(clean_df['datetime']).dt.month != 8) & (pd.to_datetime(clean_df['datetime']).dt.month != 9)]
        
    elif modelType == 'physical':
        #clean_df = clean_df[(clean_df['temp_partition'] != 'snow') | ((clean_df['temp_partition'] == 'snow') & (clean_df['pr'] == 0))]
        #clean_df = clean_df[~(((clean_df['swe'] == 0) & (clean_df['pr'] == 0)) | ((clean_df['swe'] == 0) & (clean_df['pr'].isna())))]
        clean_df = clean_df[~(clean_df['swe_t1'] == 0.0)]   # Remove no SWE days
        clean_df = clean_df[~((clean_df['temp'] < 2.0) & (clean_df['pr'] > 0.0) & (clean_df['swe_t1'] != 0.0))]    # Remove cold- and mid-wet days with SWE
        clean_df = clean_df[clean_df['swe_t1'].notnull()]
        # clean_df = clean_df[(pd.to_datetime(clean_df['datetime']).dt.month != 7) & (pd.to_datetime(clean_df['datetime']).dt.month != 8) & (pd.to_datetime(clean_df['datetime']).dt.month != 9)]
        
    elif modelType == 'full':
        # clean_df = full_df[(pd.to_datetime(full_df['datetime']).dt.month != 7) & (pd.to_datetime(full_df['datetime']).dt.month != 8) & (pd.to_datetime(full_df['datetime']).dt.month != 9)]
        #clean_df = clean_df[~(((clean_df['swe'] == 0) & (clean_df['pr'] == 0)) | ((clean_df['swe'] == 0) & (clean_df['pr'].isna())))]    # 
        clean_df = clean_df[~((clean_df['swe_t1'] == 0.0) & (clean_df['pr'] == 0.0))]    # Remove all dry days with no SWE
        clean_df = clean_df[~((clean_df['temp'] >= 2.0) & (clean_df['pr'] > 0.0) & (clean_df['swe_t1'] == 0.0))]    # Remove warm-wet days with no SWE
        clean_df = clean_df[clean_df['swe_t1'].notnull()]
        # clean_df = clean_df[clean_df['pr'].notnull()]   #?
        # clean_df = clean_df[(pd.to_datetime(clean_df['datetime']).dt.month != 7) & (pd.to_datetime(clean_df['datetime']).dt.month != 8) & (pd.to_datetime(clean_df['datetime']).dt.month != 9)]

    elif modelType == 'dswe':
        clean_df = clean_df[~((clean_df['swe_t1'] == 0.0) & (clean_df['pr'] == 0.0))]    # Remove all dry days with no SWE
        clean_df = clean_df[clean_df['swe_t1'].notnull()]
        
    return clean_df


def organize_dataframe(df, features_list, mode):
    # subset to covariates of interest
    if mode == 'train':
        clean_df = df[features_list + [config.output, 'acc_dd', 'abl_snw']]
    elif mode == 'test':
        clean_df = df[features_list + [config.output, 'acc_dd']]
    # remove rows with NANs 
    clean_df = clean_df.dropna()
    
    '''
    # USE THIS FOR SD EXPERIMENT
    clean_df = df[list((set(config.full_features) | set(features_list)) - set(['swe','sd'])) + ['daily_abl', 'acc_dd', 'abl_snw']]
    # remove rows with NANs 
    clean_df = clean_df.dropna()
    # subset to covariates of interest
    if mode == 'train':
        clean_df = clean_df[features_list + ['daily_abl', 'acc_dd', 'abl_snw']]
    elif mode == 'test':
        clean_df = clean_df[features_list + ['daily_abl', 'acc_dd']]
    '''
    
    return clean_df


def remove_outliers(df):#, features_list):
    # if any("tmin_t" in s for s in features_list):    # if feautures list contains a lag item
    #     nlags = int(next(x for x in features_list if 'tmin_t' in x)[6])    # get the lag value
    #     for l in range(1, nlags+1):
    #         df = df[df[f'tmin_t{l}'] < 3000]
    # if any("pr_t" in s for s in features_list):    # if feautures list contains a lag item
    #     nlags = int(next(x for x in features_list if 'pr_t' in x)[4])    # get the lag value
    #     for l in range(1, nlags+1):
    #         df = df[df[f'pr_t{l}'] < 500]
    # if 'tmin' in features_list:
    #     df = df[df['tmin'] < 3000]
    # if 'pr' in features_list:
    #     df = df[df['pr'] < 500]
    # if 'swe' in features_list:
    #     df = df[df['swe'] < 500]
    # if 'swe_t1' in features_list:
    #     df = df[df['swe_t1'] < 500]
    nlags = 7
    for l in range(1, nlags+1):
        df = df[df[f'tmin_t{l}'] < 3000]
        df = df[df[f'pr_t{l}'] < 500]
    df = df[df['tmin'] < 3000]
    df = df[df['pr'] < 500]
    df = df[df['swe'] < 500]
    df = df[df['swe_t1'] < 500]
        
    return df




if False:
    def clean_dataframe(full_df):   # FULL MODEL
        # remove summer (JJA) dates  
        # clean_df = full_df[(pd.to_datetime(full_df['datetime']).dt.month != 6) & (pd.to_datetime(full_df['datetime']).dt.month != 7) & (pd.to_datetime(full_df['datetime']).dt.month != 8)]
        clean_df = full_df[(pd.to_datetime(full_df['datetime']).dt.month != 7) & (pd.to_datetime(full_df['datetime']).dt.month != 8) & (pd.to_datetime(full_df['datetime']).dt.month != 9)]
        # remove days categorized as "snow"
        # clean_df = clean_df[(clean_df['partition'] != 'snow')]
        # clean_df = clean_df[(clean_df['swe_partition'] != 'snow')]
        # clean_df = clean_df[(clean_df['temp_partition'] != 'snow')]
        # clean_df = clean_df[(clean_df['temp_partition'] != 'snow') | ((clean_df['temp_partition'] == 'snow') & (clean_df['pr'] == 0))]
        # remove days where SWE or SD = 0
        # clean_df = clean_df[~(clean_df['swe'] == 0)]
        # clean_df = clean_df[~((clean_df['swe'] == 0) | (clean_df['sd'] == 0))]
        # clean_df = clean_df[~(((clean_df['swe'] == 0) & (clean_df['pr'] == 0)) | ((clean_df['sd'] == 0) & (clean_df['pr'] == 0)))]
        '''
        clean_df = clean_df[~(((clean_df['swe'] == 0) & (clean_df['pr'] == 0)) | ((clean_df['swe'] == 0) & (clean_df['pr'].isna())) | ((clean_df['sd'] == 0) & (clean_df['pr'] == 0)) | ((clean_df['sd'] == 0) & (clean_df['pr'].isna())))]
        '''
        clean_df = clean_df[~(((clean_df['swe'] == 0) & (clean_df['pr'] == 0)) | ((clean_df['swe'] == 0) & (clean_df['pr'].isna())))]
        # remove days where SWE or SD = NAN
        clean_df = clean_df[clean_df['swe'].notnull()]
        '''
        clean_df = clean_df[clean_df['sd'].notnull()]
        '''

        return clean_df



    def clean_dataframe(full_df):   # PHYSICAL MODEL
        # remove summer (JJA) dates  
        # clean_df = full_df[(pd.to_datetime(full_df['datetime']).dt.month != 6) & (pd.to_datetime(full_df['datetime']).dt.month != 7) & (pd.to_datetime(full_df['datetime']).dt.month != 8)]
        # clean_df = full_df[(pd.to_datetime(full_df['datetime']).dt.month != 7) & (pd.to_datetime(full_df['datetime']).dt.month != 8) & (pd.to_datetime(full_df['datetime']).dt.month != 9)]
        clean_df = full_df
        # remove days categorized as "snow"
        # clean_df = clean_df[(clean_df['partition'] != 'snow')]
        # clean_df = clean_df[(clean_df['swe_partition'] != 'snow')]
        # clean_df = clean_df[(clean_df['temp_partition'] != 'snow')]
        clean_df = clean_df[(clean_df['temp_partition'] != 'snow') | ((clean_df['temp_partition'] == 'snow') & (clean_df['pr'] == 0))]
        # remove days where SWE or SD = 0
        # clean_df = clean_df[~(clean_df['swe'] == 0)]
        # clean_df = clean_df[~((clean_df['swe'] == 0) | (clean_df['sd'] == 0))]
        '''
        clean_df = clean_df[~(((clean_df['swe'] == 0) & (clean_df['pr'] == 0)) | ((clean_df['swe'] == 0) & (clean_df['pr'].isna())) | ((clean_df['sd'] == 0) & (clean_df['pr'] == 0)) | ((clean_df['sd'] == 0) & (clean_df['pr'].isna())))]
        '''
        clean_df = clean_df[~(((clean_df['swe'] == 0) & (clean_df['pr'] == 0)) | ((clean_df['swe'] == 0) & (clean_df['pr'].isna())))]
        # remove days where SWE or SD = NAN
        clean_df = clean_df[clean_df['swe'].notnull()]
        '''
        clean_df = clean_df[clean_df['sd'].notnull()]
        '''

        return clean_df



    def clean_dataframe(full_df):   # UASWE-like MODEL
        # remove summer (JJA) dates  
        # clean_df = full_df[(pd.to_datetime(full_df['datetime']).dt.month != 6) & (pd.to_datetime(full_df['datetime']).dt.month != 7) & (pd.to_datetime(full_df['datetime']).dt.month != 8)]
        # clean_df = full_df[(pd.to_datetime(full_df['datetime']).dt.month != 7) & (pd.to_datetime(full_df['datetime']).dt.month != 8) & (pd.to_datetime(full_df['datetime']).dt.month != 9)]
        clean_df = full_df
        # remove days categorized as "snow"
        # clean_df = clean_df[(clean_df['partition'] != 'snow')]
        # clean_df = clean_df[(clean_df['swe_partition'] != 'snow')]
        # clean_df = clean_df[(clean_df['temp_partition'] != 'snow')]
        # clean_df = clean_df[(clean_df['temp_partition'] != 'snow') | ((clean_df['temp'] < temp_threshold) & (clean_df['temp'] >= 0.0) & (clean_df['pr'] == 0.0))]
        clean_df = clean_df[clean_df['temp'] >= 0.0]
        # remove days where SWE or SD = 0
        # clean_df = clean_df[~(clean_df['swe'] == 0)]
        '''
        clean_df = clean_df[~((clean_df['swe'] == 0) | (clean_df['sd'] == 0))]
        '''
        clean_df = clean_df[~(clean_df['swe'] == 0)]
        # clean_df = clean_df[~(((clean_df['swe'] == 0) & (clean_df['pr'] == 0)) | ((clean_df['swe'] == 0) & (clean_df['pr'].isna())) | ((clean_df['sd'] == 0) & (clean_df['pr'] == 0)) | ((clean_df['sd'] == 0) & (clean_df['pr'].isna())))]
        # remove days where SWE or SD = NAN
        clean_df = clean_df[clean_df['swe'].notnull()]
        '''
        clean_df = clean_df[clean_df['sd'].notnull()]
        '''

        return clean_df


