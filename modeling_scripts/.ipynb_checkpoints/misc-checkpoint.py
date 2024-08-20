import os
import random
import pandas as pd

import config

def partition(list_in, n):
    random.seed(27106)
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def inverse_UASWE_model(df):
    trn_df = df[(pd.to_datetime(df['datetime']).dt.month != 7) & (pd.to_datetime(df['datetime']).dt.month != 8) & (pd.to_datetime(df['datetime']).dt.month != 9)]
    trn_df = trn_df.drop_duplicates(subset=['acc_dd', 'abl_snw'])
    trn_df = trn_df[['acc_dd', 'abl_snw']].dropna()
    x = trn_df['acc_dd']
    y = trn_df['abl_snw']
    return x, y


def make_dirs(root_dir):
    os.makedirs(root_dir+'daily_eval/train/', exist_ok=True)
    os.makedirs(root_dir+'daily_eval/test/', exist_ok=True)
    os.makedirs(root_dir+'cum_swe_ts/train', exist_ok=True)
    os.makedirs(root_dir+'cum_swe_ts/test/', exist_ok=True)
    os.makedirs(root_dir+'cum_eval/aggregate_scheme/train/', exist_ok=True)
    os.makedirs(root_dir+'cum_eval/aggregate_scheme/test/', exist_ok=True)
    os.makedirs(root_dir+'cum_eval/climatological_scheme/train/', exist_ok=True)
    os.makedirs(root_dir+'cum_eval/climatological_scheme/test/', exist_ok=True)
    os.makedirs(root_dir+'cum_eval/site_year_scheme/train/', exist_ok=True)
    os.makedirs(root_dir+'cum_eval/site_year_scheme/test/', exist_ok=True)
    # os.makedirs(root_dir+'daily_on_cum_eval/aggregate_scheme/train/', exist_ok=True)
    # os.makedirs(root_dir+'daily_on_cum_eval/aggregate_scheme/test/', exist_ok=True)
    # os.makedirs(root_dir+'daily_on_cum_eval/climatological_scheme/train/', exist_ok=True)
    # os.makedirs(root_dir+'daily_on_cum_eval/climatological_scheme/test/', exist_ok=True)
    # os.makedirs(root_dir+'daily_on_cum_eval/site_year_scheme/train/', exist_ok=True)
    # os.makedirs(root_dir+'daily_on_cum_eval/site_year_scheme/test/', exist_ok=True)
    os.makedirs(root_dir+'swe_eval/aggregate_scheme/train/', exist_ok=True)
    os.makedirs(root_dir+'swe_eval/aggregate_scheme/test/', exist_ok=True)
    os.makedirs(root_dir+'swe_eval/climatological_scheme/train/', exist_ok=True)
    os.makedirs(root_dir+'swe_eval/climatological_scheme/test/', exist_ok=True)
    os.makedirs(root_dir+'swe_eval/site_year_scheme/train/', exist_ok=True)
    os.makedirs(root_dir+'swe_eval/site_year_scheme/test/', exist_ok=True)
    